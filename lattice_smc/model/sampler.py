"""DDIM denoising of one chunk given a clean prefix, with exact NFE accounting (SPEC 2.4).

Reimplements edge/model/diffusion.py::GaussianDiffusion.ddim_sample (same time grid, eta = 1,
constant guidance weight 2, x0 clipping to [-1, 1]) because EDGE's version (a) draws its noise
from the global generator, (b) calls the model through guided_forward, which cannot count forwards
or carry the is_prefix channel, and (c) has no notion of a fixed prefix half. Settings:
  sampling_timesteps S = 50 on times linspace(-1, 999, 51).int() reversed, i.e. 999, 979, ..., 19,
  then the last step returns the x0 prediction; eta = 1 (stochastic DDIM); guidance weight 2 at
  every step (EDGE's ddim_sample does not use the guidance clipping of p_mean_variance);
  clip_x_start = True.
Every denoiser call adds its batch size to counters.nfe; the conditional and unconditional halves
of guidance are one batch of 2N rows, so a guided step on N particles adds 2N.
Noise is generated on the CPU from torch.Generator objects seeded per (seed, chunk, particle), so
the noise stream does not depend on the device or on how particles are batched.
"""
import torch

from lattice_smc.model.small import CHUNK_LEN, GUIDANCE_WEIGHT, REPR_DIM, WINDOW
from lattice_smc.model.window import DEFAULT_LOSS_CFG, chunk_from_disp, clip_x0, root_shift, shift_window, unshift_window

DDIM_STEPS = 50
DDIM_ETA = 1.0
CLIP_X0 = True


def ddim_time_pairs(n_timestep=1000, steps=DDIM_STEPS):
    times = torch.linspace(-1, n_timestep - 1, steps=steps + 1).int().tolist()
    times = list(reversed(times))
    return list(zip(times[:-1], times[1:]))


def chunk_noise_seed(seed, chunk, particle):
    """One generator seed per (sequence seed, chunk index, particle index)."""
    assert 0 <= chunk < 64 and 0 <= particle < 4096
    return (int(seed) * 64 + int(chunk)) * 4096 + int(particle)


def chunk_noise(seed, chunk, particles, steps=DDIM_STEPS, device="cpu"):
    """Initial noise and the per-step DDIM noise for each particle: [N, steps, 150, 151]
    (index 0 = x_T, index i >= 1 = noise added after step i; the last step adds none)."""
    out = []
    for p in particles:
        g = torch.Generator(device="cpu").manual_seed(chunk_noise_seed(seed, chunk, p))
        out.append(torch.randn(steps, CHUNK_LEN, REPR_DIM, generator=g))
    return torch.stack(out).to(device)


def guided_x0(model, x_in, is_prefix, music, t, counters, guidance=GUIDANCE_WEIGHT):
    """Classifier-free-guided x0 prediction for the whole window. One batch of 2N rows."""
    N = x_in.shape[0]
    keep = torch.zeros(2 * N, dtype=torch.bool, device=x_in.device)
    keep[:N] = True  # rows 0..N-1 conditional, N..2N-1 unconditional (null music embedding)
    out = model(torch.cat([x_in, x_in]), torch.cat([is_prefix, is_prefix]), torch.cat([music, music]),
                torch.cat([t, t]), keep_mask=keep)
    counters.nfe.add(2 * N)
    cond, unc = out[:N], out[N:]
    return unc + (cond - unc) * guidance


@torch.no_grad()
def denoise_chunk(model, diffusion, prefix, is_prefix, music, noise, counters, guidance=GUIDANCE_WEIGHT,
                  on_x0_pred=None, cfg=DEFAULT_LOSS_CFG):
    """prefix [N, 150, 151] (clean chunk k-1, or zeros for chunk 1), is_prefix [N, 300],
    music [N, 300, 35], noise [N, 50, 150, 151] from chunk_noise. Returns the clean chunk
    [N, 150, 151]. on_x0_pred(step_index, time, x0_pred) is called after every step with the x0
    prediction in the model's representation; it may return parent indices [N] (noise-axis
    resampling), in which case the particles' states x, x0 and prefix rows are permuted before
    the DDIM update (slot s continues with slot s's noise stream)."""
    N = prefix.shape[0]
    dev = prefix.device
    pairs = ddim_time_pairs(diffusion.n_timestep)
    assert noise.shape == (N, len(pairs), CHUNK_LEN, REPR_DIM), noise.shape
    x = noise[:, 0]
    for i, (time, time_next) in enumerate(pairs):
        t = torch.full((N,), time, device=dev, dtype=torch.long)
        x_in = torch.cat([prefix, x], dim=1)
        x0 = guided_x0(model, x_in, is_prefix, music, t, counters, guidance)[:, CHUNK_LEN:]
        if CLIP_X0:
            x0 = clip_x0(x0, cfg)
        if on_x0_pred is not None:
            idx = on_x0_pred(i, time, x0)
            if idx is not None:
                idx = torch.as_tensor(idx, device=dev, dtype=torch.long)
                x, x0, prefix = x[idx], x0[idx], prefix[idx]
        if time_next < 0:
            x = x0
            break
        pred_noise = diffusion.predict_noise_from_start(x, t, x0)
        alpha = diffusion.alphas_cumprod[time]
        alpha_next = diffusion.alphas_cumprod[time_next]
        sigma = DDIM_ETA * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        x = x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise[:, i + 1]
    return x


def window_music(music, k, N):
    """Music for the denoiser window of chunk k (1-based) of a K*150-frame prompt: the previous
    chunk's music on the prefix half (zeros for chunk 1, as in unconditional-start training) and
    chunk k's music on the second half. Returns [N, 300, 35]."""
    cur = music[(k - 1) * CHUNK_LEN: k * CHUNK_LEN]
    prev = music[(k - 2) * CHUNK_LEN: (k - 1) * CHUNK_LEN] if k > 1 else torch.zeros_like(cur)
    return torch.cat([prev, cur])[None].expand(N, WINDOW, music.shape[-1]).contiguous()


def window_prefix(prev_chunk, N, device, dtype=torch.float32):
    """(prefix, is_prefix) for chunk k given clean chunk k-1 [N, 150, 151], or None for chunk 1."""
    if prev_chunk is None:
        return (torch.zeros(N, CHUNK_LEN, REPR_DIM, device=device, dtype=dtype),
                torch.zeros(N, WINDOW, device=device, dtype=dtype))
    is_prefix = torch.zeros(N, WINDOW, device=device, dtype=dtype)
    is_prefix[:, :CHUNK_LEN] = 1
    return prev_chunk, is_prefix


@torch.no_grad()
def sample_chunk(model, diffusion, prev_chunk, music_win, noise, counters, guidance=GUIDANCE_WEIGHT,
                 on_x0_pred=None, cfg=DEFAULT_LOSS_CFG):
    """Chunk k in absolute coordinates from clean chunk k-1 [N, 150, 151] (None for chunk 1):
    Amendment C shift of the prefix, denoising in the root-relative frame, un-shift of the result.
    Under Amendment E (cfg["repr"] == "disp") the denoised chunk's root channels are scaled
    displacements, integrated from the anchor (0 in the relative frame; the origin for chunk 1)
    before the un-shift. on_x0_pred(step_index, time, x0_abs) receives the x0 prediction already
    converted to absolute position form [N, 150, 151] and may return parent indices [N]; the
    per-particle shifts are permuted accordingly. Returns [N, 150, 151] in position form."""
    N = music_win.shape[0]
    dev = music_win.device
    if prev_chunk is None:
        prefix, is_prefix = window_prefix(None, N, dev)
        state = {"s": None}
    else:
        state = {"s": root_shift(prev_chunk)}
        prefix, is_prefix = window_prefix(shift_window(prev_chunk, state["s"]), N, dev)

    def to_abs(x0):
        if cfg["repr"] == "disp":
            x0 = chunk_from_disp(x0, torch.zeros(N, 3, device=dev, dtype=x0.dtype))
        return x0 if state["s"] is None else unshift_window(x0, state["s"])

    hook = None
    if on_x0_pred is not None:
        def hook(i, time, x0):
            idx = on_x0_pred(i, time, to_abs(x0))
            if idx is not None and state["s"] is not None:
                state["s"] = state["s"][torch.as_tensor(idx, device=dev, dtype=torch.long)]
            return idx
    x0 = denoise_chunk(model, diffusion, prefix, is_prefix, music_win, noise, counters, guidance, hook, cfg)
    return to_abs(x0)
