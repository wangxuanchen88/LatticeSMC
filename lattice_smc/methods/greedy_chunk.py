"""`greedy_chunk` (SPEC 4, Stream-T1 style): after each fully denoised chunk keep the top-M of N
particles by cumulative additive prefix reward (M = max(1, N // 4), ties by index) and give each
kept particle N / M slots that continue with independent noise streams. Degenerate resampling,
no twist, no lookahead. Returned sample: the argmax of the final cumulative reward."""
import numpy as np
import torch

from lattice_smc.methods.common import evaluator, gen_chunk


def run(ctx, music, K, seed, N=1, **unused):
    M = max(1, N // 4)
    if getattr(ctx, "prune_M", None):  # R1 Phase 2 (2026-09-19): M sweep; default unchanged
        M = int(N // 2) if ctx.prune_M == "N/2" else int(ctx.prune_M)
    assert N % M == 0
    ev = evaluator(ctx, music)
    prev, chunks = None, []
    R = np.zeros(N)
    events = []
    for k in range(1, K + 1):
        x = gen_chunk(ctx, prev, music, k, seed, N)
        chunks.append(x)
        R = R + ev.r_k(ev.joints(torch.cat(chunks, dim=1)), k)
        if k < K:
            order = np.argsort(-R, kind="stable")
            keep = order[:M]
            parents = np.repeat(keep, N // M)
            events.append({"chunk": k, "ess": float(M), "kept": keep.tolist(),
                           "n_distinct_parents": int(len(set(parents.tolist()))), "resampled": True})
            pt = torch.as_tensor(parents, device=ctx.device)
            chunks = [c[pt] for c in chunks]
            R = R[parents]
        prev = chunks[-1]
    idx = int(np.argmax(R))
    motion = torch.cat(chunks, dim=1)
    return motion[idx], {"rewards": R.tolist(), "chosen": idx, "final_reward": float(R[idx]),
                         "weighted_mean_reward": float(R.mean()), "M": M,
                         "ess_content": events, "ess_noise": []}
