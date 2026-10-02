# Copied from the earlier harness: <earlier-harness>/loss/span_loss.py (working
# tree; that repo has no git commit). sha256 of the source file:
# f0c1e4170fbc15e55f6aaca89fc6721e726309acc05dfaea883c1b23770ba1e1; EDGE pin
# 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications: per_frame_terms / weighted_per_frame / LOSS_WEIGHTS unchanged (they work for any
# frame count); span_reduce and diffusion_loss replaced by chunk_loss, which builds the 300-frame
# input (clean or zeroed prefix + noised chunk) and reduces over frames 150..299 only (SPEC 2.1);
# Amendment B (seam terms against the clean prefix frame 149) and Amendment C (root-relative
# window) applied in chunk_loss; Amendment E (displacement chunk) and Amendment F (seam term,
# root channel weight) selected by the loss config; per_frame_terms takes optional channel weights.
"""Per-frame EDGE loss on the 300-frame window, masked to the generated chunk.

per_frame_terms reimplements the arithmetic of edge/model/diffusion.py::GaussianDiffusion.p_losses
(x0 prediction, l2) but keeps the frame axis. Frame-coupled terms (velocity between t-1 and t;
foot sliding between t and t+1 gated by the contact at t) are assigned to the LATER frame.
Amendment B: before the terms are formed, the model's output on frames 0..149 is replaced by the
clean prefix (the pair's true frames 0..149), so the terms at frame 150 are computed against the
given prefix frame 149 and the model's prefix-half output enters no term.
Amendment C: the whole window (prefix input and target) is expressed relative to the prefix's
frame-149 root position; the loss is computed in that frame.
Amendment E ("disp"): the chunk's root channels are scaled displacements; the simple term compares
them directly, the velocity / FK / foot terms use positions reconstructed by cumulative sum from
the same anchor as the target (x0[149, root] in the loss frame).
Amendment F: channel weights in the simple term (root_weight on channels 4:7, mean over 151 kept)
and a seam term seam_lambda x ||(x0_pred[150, root] - x0[150, root]) / scale||^2 (metres) on
conditional rows, averaged over the batch.
"""
import torch

from lattice_smc.model.window import (DEFAULT_LOSS_CFG, ROOT_SCALE, chunk_to_disp, positions_from_disp, root_shift,
                                      shift_window)

from lattice_smc.edge_import import EDGE_DIR  # noqa: F401
from dataset.quaternion import ax_from_6v  # noqa: E402
from model.utils import extract  # noqa: E402

CHUNK_LEN = 150
WINDOW = 300
LOSS_WEIGHTS = {"simple": 0.636, "vel": 2.964, "fk": 0.646, "foot": 10.942}
FOOT_IDX = [7, 8, 10, 11]


def per_frame_terms(model_out, target, t, diffusion, channel_weights=None):
    """model_out, target: [B, S, 151]; t: [B] long. Returns dict of [B, S] tensors.
    channel_weights [151] (optional, Amendment F) multiplies the squared errors of the simple
    term before the mean over the 151 channels."""
    B, S, C = model_out.shape
    w = extract(diffusion.p2_loss_weight, t, (B, 1))  # [B, 1]; all ones for use_p2=False
    zero_col = model_out.new_zeros(B, 1)

    sq = (model_out - target) ** 2
    if channel_weights is not None:
        sq = sq * channel_weights.to(sq.device, sq.dtype)
    simple = sq.mean(-1)

    model_contact, mo = model_out[..., :4], model_out[..., 4:]
    tg = target[..., 4:]
    vel_core = (((mo[:, 1:] - mo[:, :-1]) - (tg[:, 1:] - tg[:, :-1])) ** 2).mean(-1)
    vel = torch.cat([zero_col, vel_core], dim=1)

    model_q = ax_from_6v(mo[..., 3:].reshape(B, S, -1, 6))
    target_q = ax_from_6v(tg[..., 3:].reshape(B, S, -1, 6))
    model_xp = diffusion.smpl.forward(model_q, mo[..., :3])  # [B, S, 24, 3]
    target_xp = diffusion.smpl.forward(target_q, tg[..., :3])
    fk = ((model_xp - target_xp) ** 2).reshape(B, S, -1).mean(-1)

    static = model_contact > 0.95  # [B, S, 4]
    feet = model_xp[:, :, FOOT_IDX]  # [B, S, 4, 3]
    foot_v = feet[:, 1:] - feet[:, :-1]  # index t: velocity between t and t+1
    foot_v = torch.where(static[:, :-1, :, None], foot_v, torch.zeros_like(foot_v))
    foot_core = (foot_v ** 2).reshape(B, S - 1, -1).mean(-1)
    foot = torch.cat([zero_col, foot_core], dim=1)

    return {"simple": simple * w, "vel": vel * w, "fk": fk * w, "foot": foot}


def weighted_per_frame(terms):
    return sum(LOSS_WEIGHTS[k] * v for k, v in terms.items())


def make_inputs(x0, music, uncond):
    """Denoiser conditioning for a batch of windows. x0 [B, 300, 151] clean, music [B, 300, 35],
    uncond [B] bool (unconditional-start rows). Returns (prefix_in [B,150,151], is_prefix [B,300],
    music_in [B,300,35]). Unconditional-start rows get a zero prefix, is_prefix = 0 everywhere and
    zero music on the prefix half, which is exactly what chunk 1 of a generated sequence sees."""
    B = x0.shape[0]
    u = uncond.to(x0.device)[:, None, None]
    prefix_in = torch.where(u, torch.zeros_like(x0[:, :CHUNK_LEN]), x0[:, :CHUNK_LEN])
    is_prefix = torch.zeros(B, WINDOW, device=x0.device, dtype=x0.dtype)
    is_prefix[:, :CHUNK_LEN] = (~uncond.to(x0.device)).to(x0.dtype)[:, None]
    music_in = music.clone()
    music_in[:, :CHUNK_LEN] = torch.where(u, torch.zeros_like(music[:, :CHUNK_LEN]), music[:, :CHUNK_LEN])
    return prefix_in, is_prefix, music_in


def channel_weights_for(cfg, like):
    if float(cfg.get("root_weight", 1.0)) == 1.0:
        return None
    cw = torch.ones(like.shape[-1], device=like.device, dtype=like.dtype)
    cw[4:7] = float(cfg["root_weight"])
    return cw


def masked_loss_from_output(out, x0, t, diffusion, cfg=DEFAULT_LOSS_CFG, uncond=None):
    """Loss from a model output [B, 300, 151] and the clean window x0 in position form (both in
    the same root frame). Amendment B: output frames 0..149 are replaced by x0's before the terms
    are formed. The weighted per-frame terms are summed over frames 150..299 and divided by
    B * 150; Amendment F adds the seam term. Returns (loss, logged terms)."""
    B = x0.shape[0]
    cw = channel_weights_for(cfg, out)
    if cfg["repr"] == "disp":
        # Amendment E: simple term on the displacement channels, the rest on reconstructed positions
        x0_disp = chunk_to_disp(x0)
        out_pos = out.clone()
        out_pos[:, CHUNK_LEN:, 4:7] = positions_from_disp(out[:, CHUNK_LEN:, 4:7], x0[:, CHUNK_LEN - 1, 4:7])
        out_ref = torch.cat([x0[:, :CHUNK_LEN], out_pos[:, CHUNK_LEN:]], dim=1)
        x0_pos = x0
        if cfg.get("fk_metres"):
            # Amendment H: root positions in metres for the FK / velocity / foot terms (the MinMax
            # offset cancels in every difference, so dividing by the scale is enough)
            sc = ROOT_SCALE.to(out.device, out.dtype)
            out_ref, x0_pos = out_ref.clone(), x0.clone()
            out_ref[..., 4:7] = out_ref[..., 4:7] / sc
            x0_pos[..., 4:7] = x0_pos[..., 4:7] / sc
        terms = per_frame_terms(out_ref, x0_pos, t, diffusion, cw)
        sq = (out[:, CHUNK_LEN:] - x0_disp[:, CHUNK_LEN:]) ** 2
        if cw is not None:
            sq = sq * cw
        w = extract(diffusion.p2_loss_weight, t, (B, 1))
        terms["simple"] = torch.cat([out.new_zeros(B, CHUNK_LEN), sq.mean(-1) * w], dim=1)
    else:
        out_ref = torch.cat([x0[:, :CHUNK_LEN], out[:, CHUNK_LEN:]], dim=1)
        terms = per_frame_terms(out_ref, x0, t, diffusion, cw)
    norm = B * CHUNK_LEN
    loss = weighted_per_frame(terms)[:, CHUNK_LEN:].sum() / norm
    logged = {k: v[:, CHUNK_LEN:].sum().detach() / norm for k, v in terms.items()}
    lam = float(cfg.get("seam_lambda", 0.0))
    if lam > 0:
        diff_m = (out[:, CHUNK_LEN, 4:7] - x0[:, CHUNK_LEN, 4:7]) / ROOT_SCALE.to(out.device, out.dtype)
        per_row = (diff_m ** 2).sum(-1)
        if uncond is not None:
            per_row = per_row * (~uncond.to(out.device)).to(per_row.dtype)
        seam = lam * per_row.mean()
        loss = loss + seam
        logged["seam"] = seam.detach()
    return loss, logged


def chunk_loss(diffusion, model, x0, music, t, noise, uncond, cond_drop_prob, keep_mask=None, cfg=DEFAULT_LOSS_CFG):
    """Training loss for one batch with fixed timesteps and noise.
    x0 [B, 300, 151] clean window (absolute root), noise [B, 150, 151] for the chunk half,
    uncond [B] bool. Returns (loss, per-term values for logging, model output in the model's
    representation and the shifted frame)."""
    x0 = shift_window(x0, root_shift(x0, uncond))
    target = chunk_to_disp(x0) if cfg["repr"] == "disp" else x0
    prefix_in, is_prefix, music_in = make_inputs(x0, music, uncond)
    x_t = diffusion.q_sample(x_start=target[:, CHUNK_LEN:], t=t, noise=noise)
    x_in = torch.cat([prefix_in, x_t], dim=1)
    out = model(x_in, is_prefix, music_in, t, cond_drop_prob=cond_drop_prob, keep_mask=keep_mask)
    loss, logged = masked_loss_from_output(out, x0, t, diffusion, cfg, uncond)
    return loss, logged, out
