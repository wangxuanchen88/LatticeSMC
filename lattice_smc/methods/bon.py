"""`bon_argmax` and `bon_is` (SPEC 4): N independent sequences, generated as one batch of N
particles; return the argmax of the full-sequence R_BA, or one drawn with probability
proportional to exp(R_BA / alpha) (the terminal-only Feynman-Kac model)."""
import numpy as np
import torch

from lattice_smc.methods.common import alpha_of, evaluator, gen_chunk, resample_generator
from lattice_smc.rewards.steering import draw_index, normalized_weights


def _run(ctx, music, K, seed, N, mode):
    ev = evaluator(ctx, music)
    prev, chunks = None, []
    for k in range(1, K + 1):
        x = gen_chunk(ctx, prev, music, k, seed, N)
        chunks.append(x)
        prev = x
    motion = torch.cat(chunks, dim=1)
    R = ev.full(ev.joints(motion))
    if mode == "argmax":
        idx = int(np.argmax(R))
        w = np.full(N, 1.0 / N)
    else:
        w = normalized_weights(R / alpha_of(ctx))
        idx = draw_index(w, resample_generator(seed, "bon_is"))
        if getattr(ctx, "return_rule", "draw") == "argmax":  # Amendment U
            idx = int(np.argmax(R))
    return motion[idx], {"rewards": R.tolist(), "weights": w.tolist(), "chosen": idx,
                         "final_reward": float(R[idx]), "weighted_mean_reward": float(np.dot(w, R)),
                         "ess_content": [], "ess_noise": []}


def run_argmax(ctx, music, K, seed, N=1, **unused):
    return _run(ctx, music, K, seed, N, "argmax")


def run_is(ctx, music, K, seed, N=1, **unused):
    return _run(ctx, music, K, seed, N, "is")
