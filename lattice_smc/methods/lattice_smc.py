"""`lattice_smc` and `lattice_smc_notwist` (SPEC 4): N particles; after chunk k the incremental
log-weight is r_k / alpha + log psi_k(prefix_k) - log psi_{k-1}(prefix_{k-1}) with psi_0 =
psi_K = 1; systematic resampling when ESS < N / 2 after chunks 1..K-1; the final sample is drawn
by weight after chunk K. The ablation uses psi = 1 (bootstrap content-axis SMC)."""
import numpy as np
import torch

from lattice_smc.methods.common import alpha_of, evaluator, gen_chunk, resample_generator
from lattice_smc.rewards.steering import draw_index, ess, normalized_weights, systematic_resample
from lattice_smc.methods.oracle import oracle_log_psi
from lattice_smc.twist.model import twist_log_psi


def _run(ctx, music, K, seed, N, use_twist, method_name, trace=None):
    alpha = alpha_of(ctx)
    if use_twist:
        assert ctx.twist is not None, "lattice_smc needs a trained twist (data/twist/twist.pt)"
    ev = evaluator(ctx, music)
    g = resample_generator(seed, method_name)
    prev, chunks = None, []
    R = np.zeros(N)
    logw = np.zeros(N)
    logpsi_prev = np.zeros(N)
    events = []
    oracle_se = []
    for k in range(1, K + 1):
        x = gen_chunk(ctx, prev, music, k, seed, N)
        chunks.append(x)
        r = ev.r_k(ev.joints(torch.cat(chunks, dim=1)), k)
        R = R + r
        if use_twist and k < K:
            mode = getattr(ctx, "twist_mode", "learned")
            if mode == "oracle" and k < K - 1:  # Amendment N: oracle at k < 3, psi_{K-1} = 1
                logpsi, se = oracle_log_psi(ctx, chunks, music, k, K, seed)
                oracle_se.append({"chunk": k, "mc_se_mean_nats": float(np.mean(se)), "mc_se_max_nats": float(np.max(se))})
            elif mode == "rollout" and k < K - 1:  # Amendment R: M continuations per particle, NFE counted in the sequence
                logpsi, se = oracle_log_psi(ctx, chunks, music, k, K, seed, counters=ctx.counters, M=int(getattr(ctx, "rollout_M", 1)), batched=True)
                oracle_se.append({"chunk": k, "mc_se_mean_nats": float(np.mean(se)) if np.all(np.isfinite(se)) else None,
                                  "mc_se_max_nats": float(np.max(se)) if np.all(np.isfinite(se)) else None})
            elif mode in ("oracle", "rollout"):
                logpsi = np.zeros(N)
            else:
                logpsi = float(getattr(ctx, "twist_beta", 1.0)) * twist_log_psi(ctx, chunks[-1], music, k, K)
        else:
            logpsi = np.zeros(N)
        if trace is not None and use_twist and k < K:
            rec = {"chunk": k, "logpsi": logpsi.copy(), "r": r.copy(), "chunks": [c.detach().clone() for c in chunks]}
            if getattr(ctx, "trace_oracle", False) and k < K - 1:
                rec["oracle"], rec["oracle_se"] = oracle_log_psi(ctx, chunks, music, k, K, seed)
            trace.append(rec)
        logw = logw + r / alpha + logpsi - logpsi_prev
        logpsi_prev = logpsi
        e = ess(logw)
        if k < K:
            resampled = e < N / 2
            events.append({"chunk": k, "ess": e, "resampled": bool(resampled)})
            if resampled:
                parents = systematic_resample(normalized_weights(logw), g)
                pt = torch.as_tensor(parents, device=ctx.device)
                chunks = [c[pt] for c in chunks]
                R, logpsi_prev = R[parents], logpsi_prev[parents]
                logw = np.zeros(N)
        prev = chunks[-1]
    w = normalized_weights(logw)
    idx = draw_index(w, g)
    if getattr(ctx, "return_rule", "draw") == "argmax":  # Amendment U: same particles and draws, argmax returned
        idx = int(np.argmax(R))
    events.append({"chunk": K, "ess": ess(logw), "resampled": False, "final_draw": True})
    motion = torch.cat(chunks, dim=1)
    log = {"rewards": R.tolist(), "weights": w.tolist(), "chosen": idx,
           "final_reward": float(R[idx]), "weighted_mean_reward": float(np.dot(w, R)),
           "ess_content": events, "ess_noise": [], "twist_mode": getattr(ctx, "twist_mode", "learned") if use_twist else "none",
           "twist_beta": float(getattr(ctx, "twist_beta", 1.0)) if use_twist else None}
    if oracle_se:
        log["oracle_mc_se"] = oracle_se
        log["oracle_nfe"] = ctx.oracle_counters.nfe.n
        log["oracle_reward_evals"] = ctx.oracle_counters.reward_evals.n
        log["rollout_M"] = int(getattr(ctx, "rollout_M", 0)) if getattr(ctx, "twist_mode", "") == "rollout" else None
    return motion[idx], log


def run(ctx, music, K, seed, N=1, trace=None, **unused):
    return _run(ctx, music, K, seed, N, True, "lattice_smc", trace)


def run_notwist(ctx, music, K, seed, N=1, **unused):
    return _run(ctx, music, K, seed, N, False, "lattice_smc_notwist")
