"""`base`: one particle, K chunks in sequence, no potentials (SPEC 4)."""
import time

import torch

from lattice_smc.model.sampler import chunk_noise, sample_chunk, window_music


def run(ctx, music, K, seed, N=1, **unused):
    """music [K*150, 35] on ctx.device. Returns (motion [K*150, 151] normalized, log dict)."""
    assert N == 1, "base is the N = 1 method"
    model, diffusion, counters, dev = ctx.model, ctx.diffusion, ctx.counters, ctx.device
    chunks, chunk_wall, chunk_nfe = [], [], []
    prev = None
    for k in range(1, K + 1):
        t0 = time.time()
        nfe0 = counters.nfe.n
        noise = chunk_noise(seed, k, range(N), device=dev)
        x0 = sample_chunk(model, diffusion, prev, window_music(music, k, N), noise, counters, cfg=ctx.loss_cfg)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        chunks.append(x0)
        prev = x0
        chunk_wall.append(round(time.time() - t0, 4))
        chunk_nfe.append(counters.nfe.n - nfe0)
    motion = torch.cat(chunks, dim=1)[0]
    return motion, {"chunk_wall": chunk_wall, "chunk_nfe": chunk_nfe}
