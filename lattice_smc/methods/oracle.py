"""Amendment N: oracle twist. For every particle at chunk boundary k, log psi_k is estimated by
M base-model continuations to the end: log mean_m exp(S_m / alpha), S_m = sum_{j > k} r_j
(the steering reward of the run). Continuation noise seeds are 50000 + (seed * 8 + k) * 64 +
slot, disjoint from every other seed space. The continuations' NFE and reward evaluations go to
ctx.oracle_counters (not compute-matched, reported separately)."""
import numpy as np
import torch

from lattice_smc.methods.common import alpha_of, evaluator
from lattice_smc.model.sampler import chunk_noise, sample_chunk, window_music


def oracle_log_psi(ctx, chunks, music, k, K, seed, counters=None, M=None, batched=False):
    """chunks: list of k tensors [N, 150, 151] (the particles' own prefixes). Returns
    (log psi [N], MC standard error [N]) in nats (nan for M = 1). counters: where the
    continuations' NFE and reward evaluations are counted (default ctx.oracle_counters).
    batched: all N x M continuations in one denoiser batch (same noise per (particle, slot);
    Amendment R rollout mode) instead of one batch of M per particle (Phase 3b oracle)."""
    M = int(getattr(ctx, "oracle_M", 16)) if M is None else int(M)
    counters = ctx.oracle_counters if counters is None else counters
    alpha = alpha_of(ctx)
    N = chunks[0].shape[0]
    ev = evaluator(ctx, music)
    ev.counters = counters
    logpsi, se = np.zeros(N), np.zeros(N)
    if batched:
        cont = [c.repeat_interleave(M, dim=0) for c in chunks]  # row = particle * M + slot
        S = np.zeros((N, M))
        for j in range(k + 1, K + 1):
            noise = torch.cat([chunk_noise(50000 + (int(seed) * 8 + k) * 64 + i, j, range(M), device=ctx.device) for i in range(N)])
            x = sample_chunk(ctx.model, ctx.diffusion, cont[-1], window_music(music, j, N * M), noise, counters, cfg=ctx.loss_cfg)
            cont.append(x)
            S += ev.r_k(ev.joints(torch.cat(cont, dim=1)), j).reshape(N, M)
        z = S / alpha
        zmax = z.max(1, keepdims=True)
        e = np.exp(z - zmax)
        return np.log(e.mean(1)) + zmax[:, 0], (e.std(1, ddof=1) / (np.sqrt(M) * e.mean(1)) if M > 1 else np.full(N, np.nan))
    for i in range(N):
        cont = [c[i:i + 1].expand(M, -1, -1).contiguous() for c in chunks]
        S = np.zeros(M)
        for j in range(k + 1, K + 1):
            noise = chunk_noise(50000 + (int(seed) * 8 + k) * 64 + i, j, range(M), device=ctx.device)
            x = sample_chunk(ctx.model, ctx.diffusion, cont[-1], window_music(music, j, M), noise, counters, cfg=ctx.loss_cfg)
            cont.append(x)
            S += ev.r_k(ev.joints(torch.cat(cont, dim=1)), j)
        z = S / alpha
        zmax = z.max()
        e = np.exp(z - zmax)
        logpsi[i] = np.log(e.mean()) + zmax
        se[i] = e.std(ddof=1) / (np.sqrt(M) * e.mean()) if M > 1 else np.nan
    return logpsi, se
