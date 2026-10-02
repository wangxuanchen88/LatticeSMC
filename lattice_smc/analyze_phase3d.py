"""Phase 3d (Amendment R): (a) the within-set twist (samples_p3_rep_wstwist/lattice_smc) against
lattice_smc_notwist (samples_p3_rep_a002) at every N, paired bootstrap over prompts, the
pre-registered question at N = 32; (b) rollout lookahead lattice_smc_rollout(M), M in {1, 2, 4}
at N in {4, 8} (samples_p3_rep_roll_M{M}/lattice_smc) against lattice_smc_notwist at the
NFE-matched N' = N (4 + 5M) / 4 (samples_p3_rep_match/lattice_smc_notwist), same 40 prompts x 4
seeds, with the counted NFE of both sides checked equal, and the per-particle Monte Carlo error
of log psi in nats per M (from the method logs for M > 1; for every M also from the M = 8
within-set rollouts, data/twist_rep/sets, by subsampling). Writes results/phase3d_{analysis.json,tables.md}.

  python -m lattice_smc.analyze_phase3d
"""
import glob
import json
import os

import numpy as np

from lattice_smc.analyze_phase2 import NS, RESULTS_DIR, md_table
from lattice_smc.analyze_phase3b import boot, load, paired, summarize
from lattice_smc.edge_import import ROOT

MATCH = {(4, 1): 9, (4, 2): 14, (4, 4): 24, (8, 1): 18, (8, 2): 28, (8, 4): 48}  # N (4 + 5M) / 4, all exact


def mc_se_from_sets(M, rng, n_sub=64):
    """Per-particle MC standard error of log mean exp(S / alpha) with M continuations, estimated by
    drawing M of the 8 stored continuations of every particle in every within-set rollout set."""
    out = {1: [], 2: []}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "twist_rep", "sets", "*.npz"))):
        z = np.load(f)
        alpha = 0.02
        for k in (1, 2):
            zz = z[f"S_k{k}"] / alpha  # [16, 8]
            full = np.log(np.mean(np.exp(zz - zz.max(1, keepdims=True)), 1)) + zz.max(1)
            if M >= zz.shape[1]:
                sub = full[None]
            else:
                idx = np.stack([rng.permutation(zz.shape[1])[:M] for _ in range(n_sub)])  # [n_sub, M]
                s = zz[:, idx]  # [16, n_sub, M]
                sub = (np.log(np.mean(np.exp(s - s.max(2, keepdims=True)), 2)) + s.max(2)).T  # [n_sub, 16]
            out[k].append(float(np.sqrt(np.mean((sub - full[None]) ** 2))))  # RMS deviation from the M = 8 estimate
    return {str(k): float(np.mean(v)) for k, v in out.items()}


def main():
    rng = np.random.default_rng(0)
    notwist = {N: load("samples_p3_rep_a002", "lattice_smc_notwist", N) for N in NS}
    ws = {N: load("samples_p3_rep_wstwist", "lattice_smc", N) for N in NS}
    rows, pairs = [], []
    for N in NS:
        if notwist[N]:
            rows.append({"variant": "notwist", "N": N, **summarize(notwist[N], N, rng)})
        if ws[N]:
            rows.append({"variant": "within-set twist", "N": N, **summarize(ws[N], N, rng)})
            pairs.append({"variant": "within-set twist", "vs": "notwist", "N": N, **paired(ws[N], notwist[N], rng)})
    q32 = next((p for p in pairs if p["N"] == 32), None)
    question = {"statement": "does lattice_smc with the within-set twist beat lattice_smc_notwist at N = 32 with an interval excluding zero",
                "N32": q32, "met": bool(q32 and q32["separates"] and q32["diff"] > 0)}
    # rollout lookahead at matched NFE
    roll_rows, roll_pairs = [], []
    for M in (1, 2, 4):
        for N in (4, 8):
            d = load(f"samples_p3_rep_roll_M{M}", "lattice_smc", N)
            Nm = MATCH[(N, M)]
            m = load("samples_p3_rep_match", "lattice_smc_notwist", Nm)
            if not d or not m:
                continue
            rr = {"variant": f"rollout M={M}", "N": N, **summarize(d, N, rng)}
            rr["nfe_expected"] = 100 * N * (4 + 5 * M)
            mr = {"variant": f"notwist matched (N'={Nm})", "N": Nm, **summarize(m, Nm, rng)}
            mr["nfe_expected"] = 400 * Nm
            roll_rows += [rr, mr]
            pr = paired(d, m, rng)
            pr.update({"variant": f"rollout M={M} N={N}", "vs": f"notwist N'={Nm}", "nfe_rollout": rr["nfe"], "nfe_matched": mr["nfe"],
                       "nfe_equal": rr["nfe"] == mr["nfe"] and len(rr["nfe"]) == 1})
            roll_pairs.append(pr)
    mc = {f"M={M}": mc_se_from_sets(M, rng) for M in (1, 2, 4, 8)}
    out = {"rows": rows, "pairs": pairs, "question": question, "rollout_rows": roll_rows, "rollout_pairs": roll_pairs,
           "mc_se_nats_from_sets_rms_vs_M8": mc, "matched_N": {f"N={n},M={m}": v for (n, m), v in MATCH.items()}}
    json.dump(out, open(os.path.join(RESULTS_DIR, "phase3d_analysis.json"), "w"), indent=1)
    cols = ["variant", "N", "n_sequences", "R_mean", "ci_lo", "ci_hi", "frac_resampled_k1", "frac_resampled_k2", "frac_resampled_k3",
            "ess_over_N_k1", "ess_over_N_k2", "ess_over_N_k3", "final_particle_mean", "final_particle_max", "twist_evals", "wall_s"]
    fmt = {"frac_resampled_k1": "{:.2f}", "frac_resampled_k2": "{:.2f}", "frac_resampled_k3": "{:.2f}", "ess_over_N_k1": "{:.3f}",
           "ess_over_N_k2": "{:.3f}", "ess_over_N_k3": "{:.3f}", "twist_evals": "{:.0f}", "wall_s": "{:.2f}", "oracle_nfe_per_sequence": "{:.0f}"}
    md = "# Phase 3d tables (Amendment R, R_rep, alpha = 0.02)\n\n## Within-set twist vs notwist (40 prompts x 4 seeds)\n\n" + md_table(rows, cols, fmt)
    md += "\n## Paired differences (within-set twist - notwist, bootstrap over prompts)\n\n" + md_table(pairs, ["variant", "vs", "N", "n_prompts", "n_sequences", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Pre-registered question\n\n```\n" + json.dumps(question, indent=1) + "\n```\n"
    md += "\n## Rollout lookahead at matched NFE (40 prompts x 4 seeds; nfe = counted per sequence, all continuation NFE included)\n\n"
    md += md_table(roll_rows, ["variant", "N", "n_sequences", "nfe", "nfe_expected", "R_mean", "ci_lo", "ci_hi", "frac_resampled_k1", "frac_resampled_k2", "frac_resampled_k3",
                               "final_particle_mean", "final_particle_max", "oracle_mc_se_nats_mean_k1", "oracle_mc_se_nats_mean_k2", "oracle_nfe_per_sequence", "wall_s"], fmt)
    md += "\n## Paired differences (rollout - matched notwist, same sequences)\n\n" + md_table(roll_pairs, ["variant", "vs", "n_prompts", "n_sequences", "nfe_rollout", "nfe_matched", "nfe_equal", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Per-particle MC error of log psi (nats): RMS deviation of the M-continuation estimate from the M = 8 estimate on the within-set rollouts\n\n```\n" + json.dumps(mc, indent=1) + "\n```\n"
    open(os.path.join(RESULTS_DIR, "phase3d_tables.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
