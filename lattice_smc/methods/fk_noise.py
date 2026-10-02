"""`fk_noise` (SPEC 4): Feynman-Kac steering along the noise axis. Within each chunk, N
particles; after DDIM steps 10, 20, 30, 40, 50 (of 50) the potential is
exp((r_hat - r_hat_prev) / alpha), r_hat = the particle's own prefix reward + r_k of the chunk's
x0 prediction at that step (the difference potential; r_hat_prev at a chunk's first event is the
prefix reward), systematic resampling when ESS < N / 2. Particles carry their own prefixes across
chunks, log-weights carry over, no resampling at chunk boundaries; the final sample is drawn by
weight. x0 predictions are free (no extra NFE); reward evaluations are counted."""
import numpy as np
import torch

from lattice_smc.methods.common import alpha_of, evaluator, gen_chunk, resample_generator
from lattice_smc.rewards.steering import draw_index, ess, normalized_weights, systematic_resample

RESAMPLE_STEPS = (9, 19, 29, 39, 49)  # 0-based DDIM step indices: after steps 10, 20, ..., 50


def run(ctx, music, K, seed, N=1, trace=None, **unused):
    """trace (optional list): every potential event is appended as a dict with the per-slot
    r_hat, r_prev, log-potential and the parent indices of the resampling (Amendment L check)."""
    alpha = alpha_of(ctx)
    ev = evaluator(ctx, music)
    g = resample_generator(seed, "fk_noise")
    st = {"chunks": [], "R": np.zeros(N), "logw": np.zeros(N), "r_prev": None, "r_last": None}
    events = []
    for k in range(1, K + 1):
        st["r_prev"] = st["R"].copy()

        def on_x0(i, time, x0_abs, k=k):
            if i not in RESAMPLE_STEPS:
                return None
            J = ev.joints(torch.cat(st["chunks"] + [x0_abs], dim=1))
            r_hat = st["R"] + ev.r_k(J, k)
            logG = (r_hat - st["r_prev"]) / alpha
            st["logw"] = st["logw"] + logG
            e = ess(st["logw"])
            resampled = e < N / 2
            events.append({"chunk": k, "step": i + 1, "ess": e, "resampled": bool(resampled)})
            idx = None
            if resampled:
                idx = systematic_resample(normalized_weights(st["logw"]), g)
            if trace is not None:
                trace.append({"chunk": k, "step": i + 1, "r_hat": r_hat.copy(), "r_prev": st["r_prev"].copy(),
                              "logG": logG.copy(), "parents": None if idx is None else idx.copy()})
            if resampled:
                pt = torch.as_tensor(idx, device=ctx.device)
                st["chunks"] = [c[pt] for c in st["chunks"]]
                st["R"] = st["R"][idx]
                r_hat = r_hat[idx]
                st["logw"] = np.zeros(N)
            st["r_prev"] = r_hat
            st["r_last"] = r_hat
            return idx

        prev = st["chunks"][-1] if st["chunks"] else None
        x = gen_chunk(ctx, prev, music, k, seed, N, on_x0_pred=on_x0)
        st["chunks"].append(x)
        # the x0 prediction at the last step is the returned chunk, so r_last = prefix + r_k(chunk)
        st["R"] = st["r_last"].copy()
    w = normalized_weights(st["logw"])
    idx = draw_index(w, g)
    R = st["R"]
    if getattr(ctx, "return_rule", "draw") == "argmax":  # Amendment U: same particles and draws, argmax returned
        idx = int(np.argmax(R))  # (2026-09-18: R was read before assignment on this branch; the draw path is unchanged)
    motion = torch.cat(st["chunks"], dim=1)
    return motion[idx], {"rewards": R.tolist(), "weights": w.tolist(), "chosen": idx,
                         "final_reward": float(R[idx]), "weighted_mean_reward": float(np.dot(w, R)),
                         "ess_content": [], "ess_noise": events}
