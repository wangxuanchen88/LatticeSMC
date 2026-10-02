"""SPEC 2.5 loss-masking check (blocking) for the three loss variants:
  relative  Amendments B + C (Phase 1b)
  disp      Amendment E (root-displacement chunk), candidate 1
  seamF     Amendment F (seam term lambda = 50, root channel weight 10), candidate 2
For a fixed batch, the chunk loss equals an independently written reference restricted to frames
150..299 with the seam terms at frame 150 computed against the clean prefix frame 149, float64,
rtol 1e-10. The reference uses EDGE's p_losses arithmetic on frame slices of the root-relative
window and, for "disp", reconstructs positions with an explicit running sum; it never calls
per_frame_terms or masked_loss_from_output. The model's output on frames 0..149 must not enter
the loss at all.
"""
import os

import pytest
import torch

from lattice_smc.data.pairs import CACHE_DIR, PairSet
from lattice_smc.edge_import import EDGE_DIR  # noqa: F401
from lattice_smc.model.loss import (CHUNK_LEN, LOSS_WEIGHTS, chunk_loss, make_inputs, masked_loss_from_output,
                                    per_frame_terms, weighted_per_frame)
from lattice_smc.model.small import build_diffusion, build_model
from lattice_smc.model.window import DISP_FACTOR, ROOT_SCALE, root_shift, shift_window
from dataset.quaternion import ax_from_6v  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
B = 4
VARIANTS = {"relative": {"repr": "relative", "root_weight": 1.0, "seam_lambda": 0.0},
            "disp": {"repr": "disp", "root_weight": 1.0, "seam_lambda": 0.0},
            "seamF": {"repr": "relative", "root_weight": 10.0, "seam_lambda": 50.0},
            "dispH": {"repr": "disp", "root_weight": 1.0, "seam_lambda": 0.0, "fk_metres": True}}


@pytest.fixture(autouse=True)
def _reset_default_dtype():
    yield
    torch.set_default_dtype(torch.float32)


def make_batch(dtype):
    torch.manual_seed(0)
    if os.path.exists(os.path.join(CACHE_DIR, "pairs_train.npz")):
        ps = PairSet("pairs_train", n=B)
        x0, music = ps.motion, ps.music
    else:
        x0 = torch.rand(B, 300, 151) * 2 - 1
        music = torch.randn(B, 300, 35)
    t = torch.tensor([300, 500, 700, 800])
    noise = torch.randn(B, CHUNK_LEN, 151)
    uncond = torch.tensor([False, True, False, False])
    return x0.to(DEV, dtype), music.to(DEV, dtype), t.to(DEV), noise.to(DEV, dtype), uncond.to(DEV)


def make_model(dtype):
    # EDGE's SinusoidalPosEmb builds its frequency table in the default dtype.
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)
    model = build_model({}).to(DEV, dtype).eval()
    diffusion = build_diffusion(model, DEV).to(dtype)
    diffusion.smpl._offsets = diffusion.smpl._offsets.to(dtype)
    return model, diffusion


def reference_loss(out, x0, diffusion, cfg, uncond):
    """EDGE p_losses arithmetic on frames 150..299 of a 300-frame output, seam terms against the
    clean frame 149 of x0 (Amendment B), summed, divided by B*150; plus the variant's extras."""
    Bn = out.shape[0]
    s = CHUNK_LEN
    dt = out.dtype
    cw = torch.ones(151, device=out.device, dtype=dt)
    cw[4:7] = cfg["root_weight"]
    # position-form output: for "disp" rebuild the chunk root with an explicit running sum
    out_pos = out.clone()
    if cfg["repr"] == "disp":
        fac = DISP_FACTOR.to(out.device, dt)
        p = x0[:, s - 1, 4:7].clone()
        for f in range(s, 2 * s):
            p = p + out[:, f, 4:7] * fac
            out_pos[:, f, 4:7] = p
        tgt_disp = x0.clone()
        for f in range(s, 2 * s):
            tgt_disp[:, f, 4:7] = (x0[:, f, 4:7] - x0[:, f - 1, 4:7]) / fac
        simple = (((out[:, s:] - tgt_disp[:, s:]) ** 2) * cw).mean(-1)
    else:
        simple = (((out[:, s:] - x0[:, s:]) ** 2) * cw).mean(-1)
    mo, tg = out_pos[..., 4:], x0[..., 4:]
    if cfg.get("fk_metres"):
        sc = ROOT_SCALE.to(out.device, dt)
        mo, tg = mo.clone(), tg.clone()
        mo[..., :3] = mo[..., :3] / sc
        tg[..., :3] = tg[..., :3] / sc
    mo_prev = torch.cat([tg[:, s - 1:s], mo[:, s:-1]], dim=1)  # frames 149 (clean), 150..298 (model)
    vel = (((mo[:, s:] - mo_prev) - (tg[:, s:] - tg[:, s - 1:-1])) ** 2).mean(-1)

    def fk(v):  # v [B, T, 147] -> [B, T, 24, 3]
        return diffusion.smpl.forward(ax_from_6v(v[..., 3:].reshape(v.shape[0], v.shape[1], 24, 6)), v[..., :3])

    xp_m = fk(mo[:, s:])  # model frames 150..299
    xp_t = fk(tg[:, s - 1:])  # clean frames 149..299
    fkl = ((xp_m - xp_t[:, 1:]) ** 2).reshape(Bn, s, -1).mean(-1)
    feet_m = xp_m[:, :, [7, 8, 10, 11]]
    feet_prev = torch.cat([xp_t[:, :1, [7, 8, 10, 11]], feet_m[:, :-1]], dim=1)  # clean 149, model 150..298
    contact_prev = torch.cat([x0[:, s - 1:s, :4], out[:, s:-1, :4]], dim=1) > 0.95  # clean 149, model 150..298
    fv = feet_m - feet_prev
    fv = torch.where(contact_prev[:, :, :, None], fv, torch.zeros_like(fv))
    foot = (fv ** 2).reshape(Bn, s, -1).mean(-1)
    total = (LOSS_WEIGHTS["simple"] * simple + LOSS_WEIGHTS["vel"] * vel + LOSS_WEIGHTS["fk"] * fkl
             + LOSS_WEIGHTS["foot"] * foot)
    loss = total.sum() / (Bn * s)
    if cfg["seam_lambda"] > 0:
        seam = 0.0
        for b in range(Bn):
            if not bool(uncond[b]):
                d = (out[b, s, 4:7] - x0[b, s, 4:7]) / ROOT_SCALE.to(out.device, dt)
                seam = seam + (d ** 2).sum()
        loss = loss + cfg["seam_lambda"] * seam / Bn
    return loss


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 1e-5), (torch.float64, 1e-10)])
def test_prefix_mask_equals_restricted_per_frame_loss(variant, dtype, rtol):
    cfg = VARIANTS[variant]
    x0, music, t, noise, uncond = make_batch(dtype)
    model, diffusion = make_model(dtype)
    with torch.no_grad():
        loss, terms, out = chunk_loss(diffusion, model, x0, music, t, noise, uncond, 0.0, cfg=cfg)
        x0s = shift_window(x0, root_shift(x0, uncond))  # the frame the loss is computed in (Amendment C)
        ref = reference_loss(out, x0s, diffusion, cfg, uncond)
    assert loss.item() > 0
    assert torch.allclose(loss, ref, rtol=rtol, atol=0), (variant, loss.item(), ref.item())
    logged = sum(LOSS_WEIGHTS[k] * terms[k] for k in LOSS_WEIGHTS) + terms.get("seam", 0)
    assert torch.allclose(logged, loss, rtol=rtol, atol=0)
    if variant == "relative":
        # the earlier harness per-frame terms on the window with the prefix half of the output replaced by
        # the clean prefix (Amendment B), explicitly masked to 150..299
        with torch.no_grad():
            out_ref = torch.cat([x0s[:, :CHUNK_LEN], out[:, CHUNK_LEN:]], dim=1)
            pf = weighted_per_frame(per_frame_terms(out_ref, x0s, t, diffusion))
            mask = torch.zeros(B, 300, device=DEV, dtype=dtype)
            mask[:, CHUNK_LEN:] = 1
            masked = (pf * mask).sum() / mask.sum()
        assert torch.allclose(loss, masked, rtol=rtol, atol=0), (loss.item(), masked.item())


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_model_output_on_prefix_frames_enters_no_term(variant):
    """Amendment B: replacing the model's output on frames 0..149 by anything leaves the loss
    bitwise unchanged."""
    cfg = VARIANTS[variant]
    x0, music, t, noise, uncond = make_batch(torch.float64)
    model, diffusion = make_model(torch.float64)
    with torch.no_grad():
        _, _, out = chunk_loss(diffusion, model, x0, music, t, noise, uncond, 0.0, cfg=cfg)
        x0s = shift_window(x0, root_shift(x0, uncond))
        loss, _ = masked_loss_from_output(out, x0s, t, diffusion, cfg, uncond)
        out_pert = out.clone()
        out_pert[:, :CHUNK_LEN] = torch.randn_like(out_pert[:, :CHUNK_LEN]) * 3
        loss_pert, _ = masked_loss_from_output(out_pert, x0s, t, diffusion, cfg, uncond)
    assert torch.equal(loss, loss_pert)


def test_seam_terms_use_clean_prefix_frame_149():
    """Perturbing the clean frame 149 changes the loss (through the seam terms), perturbing
    clean frames 0..148 does not."""
    cfg = VARIANTS["relative"]
    x0, music, t, noise, uncond = make_batch(torch.float64)
    model, diffusion = make_model(torch.float64)
    with torch.no_grad():
        _, _, out = chunk_loss(diffusion, model, x0, music, t, noise, uncond, 0.0, cfg=cfg)
        x0s = shift_window(x0, root_shift(x0, uncond))
        loss, _ = masked_loss_from_output(out, x0s, t, diffusion, cfg, uncond)
        early = x0s.clone()
        early[:, :CHUNK_LEN - 1] += torch.randn_like(early[:, :CHUNK_LEN - 1])
        loss_early, _ = masked_loss_from_output(out, early, t, diffusion, cfg, uncond)
        seam = x0s.clone()
        seam[:, CHUNK_LEN - 1, 4:] += 0.1
        loss_seam, _ = masked_loss_from_output(out, seam, t, diffusion, cfg, uncond)
    assert torch.equal(loss, loss_early)
    assert not torch.equal(loss, loss_seam)


def test_seam_term_F_is_zero_when_frame150_root_is_exact_and_skips_uncond_rows():
    cfg = VARIANTS["seamF"]
    x0, music, t, noise, uncond = make_batch(torch.float64)
    model, diffusion = make_model(torch.float64)
    with torch.no_grad():
        _, _, out = chunk_loss(diffusion, model, x0, music, t, noise, uncond, 0.0, cfg=cfg)
        x0s = shift_window(x0, root_shift(x0, uncond))
        _, logged = masked_loss_from_output(out, x0s, t, diffusion, cfg, uncond)
        assert logged["seam"].item() > 0
        fixed = out.clone()
        fixed[:, CHUNK_LEN, 4:7] = x0s[:, CHUNK_LEN, 4:7]
        _, logged_fixed = masked_loss_from_output(fixed, x0s, t, diffusion, cfg, uncond)
        assert logged_fixed["seam"].item() == 0.0
        # an unconditional row's frame-150 root does not enter the seam term
        pert = out.clone()
        pert[1, CHUNK_LEN, 4:7] += 1.0
        _, logged_pert = masked_loss_from_output(pert, x0s, t, diffusion, cfg, uncond)
        assert torch.equal(logged_pert["seam"], logged["seam"])


def test_inputs_for_unconditional_start():
    x0, music, t, noise, uncond = make_batch(torch.float32)
    prefix_in, is_prefix, music_in = make_inputs(x0, music, uncond)
    assert torch.all(prefix_in[1] == 0) and torch.all(is_prefix[1] == 0) and torch.all(music_in[1, :CHUNK_LEN] == 0)
    assert torch.equal(prefix_in[0], x0[0, :CHUNK_LEN]) and torch.all(is_prefix[0, :CHUNK_LEN] == 1)
    assert torch.all(is_prefix[:, CHUNK_LEN:] == 0) and torch.equal(music_in[:, CHUNK_LEN:], music[:, CHUNK_LEN:])
