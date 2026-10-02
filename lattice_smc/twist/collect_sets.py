"""Amendment R part 1: within-set twist rollouts. For each (segment, seed) set: N = 16 independent
base prefixes at k = 1 (particle slots 0..15 of the base seed), each continued to k = 2 with its
own slot's noise; for every particle at k = 1 and at k = 2, M = 8 base continuations to the end
with distinct seeds; targets log mean exp(S / alpha), S = sum_{j > k} r_j (R_rep).

Segments: 100 of the 317 twist-pool segments drawn with default_rng(0) without replacement; the
last 20 of the draw are held out entirely (both seeds). Seeds: prefix noise 60000 + 2 i + s for
set (segment i, seed s in {0, 1}); continuation noise 70000 + (2 i + s) * 64 + 16 * (k - 1) + slot.

  python -m lattice_smc.twist.collect_sets --ckpt runs/chunk_dispE/ckpt_adopted.pt --alpha 0.02

Writes data/twist_rep/sets/{i:03d}_{s}.npz and data/twist_rep/sets/index.json.
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lattice_smc.data.pairs import DATA_DIR  # noqa: E402
from lattice_smc.generate import Context, set_deterministic  # noqa: E402
from lattice_smc.methods.common import evaluator  # noqa: E402
from lattice_smc.model.sampler import chunk_noise, sample_chunk, window_music  # noqa: E402
from lattice_smc.model.small import load_checkpoint  # noqa: E402
from lattice_smc.twist.collect import load_segments  # noqa: E402

SETS_DIR = os.path.join(DATA_DIR, "twist_rep", "sets")
N_PART, M_CONT, K = 16, 8, 4


def targets(S, alpha):
    z = S / alpha
    zmax = z.max(-1, keepdims=True)
    e = np.exp(z - zmax)
    return np.log(e.mean(-1)) + zmax[..., 0], e.std(-1, ddof=1) / (np.sqrt(e.shape[-1]) * e.mean(-1))


def one_set(ctx, music, seed_prefix, seed_cont, alpha):
    dev = ctx.device
    ev = evaluator(ctx, music)
    # N independent base particles: chunk 1 from slots 0..15, chunk 2 continuing each with its own slot
    c1 = sample_chunk(ctx.model, ctx.diffusion, None, window_music(music, 1, N_PART), chunk_noise(seed_prefix, 1, range(N_PART), device=dev), ctx.counters, cfg=ctx.loss_cfg)
    c2 = sample_chunk(ctx.model, ctx.diffusion, c1, window_music(music, 2, N_PART), chunk_noise(seed_prefix, 2, range(N_PART), device=dev), ctx.counters, cfg=ctx.loss_cfg)
    out = {"chunk1": c1.cpu().numpy(), "chunk2": c2.cpu().numpy()}
    for k, prefix_chunks in ((1, [c1]), (2, [c1, c2])):
        # all N x M continuations in one batch: row = particle * M + m
        cont = [c.repeat_interleave(M_CONT, dim=0) for c in prefix_chunks]
        S = np.zeros((N_PART, M_CONT))
        for j in range(k + 1, K + 1):
            noise = torch.cat([chunk_noise(seed_cont + 16 * (k - 1) + p, j, range(M_CONT), device=dev) for p in range(N_PART)])
            x = sample_chunk(ctx.model, ctx.diffusion, cont[-1], window_music(music, j, N_PART * M_CONT), noise, ctx.counters, cfg=ctx.loss_cfg)
            cont.append(x)
            S += ev.r_k(ev.joints(torch.cat(cont, dim=1)), j).reshape(N_PART, M_CONT)
        t, se = targets(S, alpha)
        out[f"S_k{k}"], out[f"target_k{k}"], out[f"se_k{k}"] = S, t, se
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--n_segments", type=int, default=100)
    ap.add_argument("--n_heldout", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_deterministic()
    dev = torch.device(args.device)
    model, diffusion, ck = load_checkpoint(args.ckpt, dev)
    ctx = Context(model, diffusion, dev, ck["config"], alpha=args.alpha, reward="rep")
    segs, music_all = load_segments(K, os.path.join(DATA_DIR, "twist_rep"))
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(segs), size=args.n_segments, replace=False).tolist()
    heldout = chosen[-args.n_heldout:]
    os.makedirs(SETS_DIR, exist_ok=True)
    json.dump({"segments": chosen, "heldout": heldout, "n_particles": N_PART, "M": M_CONT, "alpha": args.alpha,
               "seeds_per_segment": 2, "pieces": sorted({segs[i]["piece"] for i in chosen})}, open(os.path.join(SETS_DIR, "index.json"), "w"), indent=1)
    t0 = time.time()
    n = 0
    for pos, seg in enumerate(chosen):
        for s in (0, 1):
            path = os.path.join(SETS_DIR, f"{pos:03d}_{s}.npz")
            if os.path.exists(path):
                continue
            ctx.counters.reset()
            res = one_set(ctx, music_all[seg].to(dev), 60000 + 2 * pos + s, 70000 + (2 * pos + s) * 64, args.alpha)
            np.savez(path, segment=seg, position=pos, seed=s, heldout=(seg in heldout), nfe=ctx.counters.nfe.n, **res)
            n += 1
            if n % 20 == 0:
                print(f"{n} sets, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new sets in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
