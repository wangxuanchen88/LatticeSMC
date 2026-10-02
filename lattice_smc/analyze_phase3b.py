"""Phase 3b (Amendment N): twist variants on R_rep at alpha = 0.02, from the method logs.
For R_rep the method's `final_reward` is exactly the returned sample's R_rep (additive == full).

  python -m lattice_smc.analyze_phase3b

Variants: learned (samples_p3_rep_a002/lattice_smc), tempered beta 0.5 / 0.25 / 0.1
(samples_p3_rep_tw*/lattice_smc), notwist (samples_p3_rep_a002/lattice_smc_notwist), and the
oracle (samples_p3_rep_oracle/lattice_smc, first 10 prompts, N in {8, 32}). Per variant and N:
mean R_rep with a 1000-resample bootstrap over prompts, fraction of events resampled per chunk,
mean ESS / N before events per chunk, final-particle mean and max reward, twist evaluations;
paired differences vs notwist (bootstrap over prompts); the oracle compared on its own 40
sequences against the learned and notwist samples of the same (prompt, seed); the oracle's
Monte Carlo standard error and continuation NFE. Writes results/phase3b_{analysis.json,tables.md}.
"""
import glob
import json
import os

import numpy as np

from lattice_smc.analyze_phase2 import NS, RESULTS_DIR, md_table
from lattice_smc.edge_import import ROOT

VARIANTS = {"notwist": ("samples_p3_rep_a002", "lattice_smc_notwist"), "learned (beta 1)": ("samples_p3_rep_a002", "lattice_smc"),
            "tempered beta 0.5": ("samples_p3_rep_tw050", "lattice_smc"), "tempered beta 0.25": ("samples_p3_rep_tw025", "lattice_smc"),
            "tempered beta 0.1": ("samples_p3_rep_tw010", "lattice_smc"), "oracle (M = 16)": ("samples_p3_rep_oracle", "lattice_smc"),
            "ensemble mean (5)": ("samples_p3_rep_ens_mean", "lattice_smc"), "ensemble shrunk (5, c)": ("samples_p3_rep_ens_shrunk", "lattice_smc")}
N_BOOT, SEED = 1000, 0


def load(root, method, N):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, root, method, str(N), "*.npz"))):
        z = np.load(f)
        log = json.loads(str(z["log"]))
        out[(log["prompt_id"], int(log["seed"]))] = {"R": float(log["final_reward"]), "rewards": np.array(log["rewards"]),
                                                    "weights": np.array(log["weights"]), "ess": log["ess_content"],
                                                    "twist_evals": int(log.get("twist_evals", 0)), "nfe": int(z["nfe"]),
                                                    "oracle_se": log.get("oracle_mc_se"), "oracle_nfe": log.get("oracle_nfe"),
                                                    "wall": float(z["wall"])}
    return out


def boot(values_by_prompt, rng):
    v = np.asarray(values_by_prompt, float)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_prompt(d, keys=None):
    by = {}
    for (p, s), r in d.items():
        if keys is None or (p, s) in keys:
            by.setdefault(p, []).append(r["R"])
    return {p: float(np.mean(v)) for p, v in by.items()}


def summarize(d, N, rng, keys=None):
    sel = {k: v for k, v in d.items() if keys is None or k in keys}
    pp = per_prompt(sel)
    mean, lo, hi = boot(list(pp.values()), rng)
    row = {"n_sequences": len(sel), "n_prompts": len(pp), "R_mean": mean, "ci_lo": lo, "ci_hi": hi,
           "final_particle_mean": float(np.mean([r["rewards"].mean() for r in sel.values()])),
           "final_particle_max": float(np.mean([r["rewards"].max() for r in sel.values()])),
           "twist_evals": float(np.mean([r["twist_evals"] for r in sel.values()])), "nfe": sorted({r["nfe"] for r in sel.values()}),
           "wall_s": float(np.mean([r["wall"] for r in sel.values()]))}
    for k in (1, 2, 3):
        ev = [e for r in sel.values() for e in r["ess"] if e["chunk"] == k and not e.get("final_draw")]
        row[f"frac_resampled_k{k}"] = float(np.mean([e["resampled"] for e in ev])) if ev else float("nan")
        row[f"ess_over_N_k{k}"] = float(np.mean([e["ess"] for e in ev]) / N) if ev else float("nan")
    se = [s for r in sel.values() if r["oracle_se"] for s in r["oracle_se"]]
    if se:  # M = 1 rollouts (Amendment R) log None: no standard error from one continuation
        def _m(vals, f=np.mean):
            vals = [v for v in vals if v is not None]
            return float(f(vals)) if vals else float("nan")
        row["oracle_mc_se_nats_mean_k1"] = _m([s["mc_se_mean_nats"] for s in se if s["chunk"] == 1])
        row["oracle_mc_se_nats_mean_k2"] = _m([s["mc_se_mean_nats"] for s in se if s["chunk"] == 2])
        row["oracle_mc_se_nats_max"] = _m([s["mc_se_max_nats"] for s in se], np.max)
        row["oracle_nfe_per_sequence"] = float(np.mean([r["oracle_nfe"] for r in sel.values()]))
    return row


def paired(a, b, rng, keys=None):
    common = [k for k in a if k in b and (keys is None or k in keys)]
    by = {}
    for k in common:
        by.setdefault(k[0], []).append(a[k]["R"] - b[k]["R"])
    d = [float(np.mean(v)) for v in by.values()]
    mean, lo, hi = boot(d, rng)
    return {"n_prompts": len(d), "n_sequences": len(common), "diff": mean, "ci_lo": lo, "ci_hi": hi, "separates": bool(lo > 0 or hi < 0)}


def main():
    rng = np.random.default_rng(SEED)
    data = {name: {N: load(root, m, N) for N in NS} for name, (root, m) in VARIANTS.items()}
    rows, pairs = [], []
    for name in VARIANTS:
        for N in NS:
            d = data[name][N]
            if not d:
                continue
            rows.append({"variant": name, "N": N, **summarize(d, N, rng)})
            if name != "notwist":
                keys = set(d) if name.startswith("oracle") else None
                pr = paired(d, data["notwist"][N], rng, keys)
                pairs.append({"variant": name, "vs": "notwist", "N": N, **pr})
                if name.startswith("oracle"):
                    pairs.append({"variant": name, "vs": "learned (beta 1)", "N": N, **paired(d, data["learned (beta 1)"][N], rng, keys)})
    # notwist and learned restricted to the oracle's 40 sequences, for the same-sequence table
    oracle_rows = []
    for N in (8, 32):
        keys = set(data["oracle (M = 16)"][N])
        if not keys:
            continue
        for name in ("oracle (M = 16)", "learned (beta 1)", "tempered beta 0.1", "notwist"):
            oracle_rows.append({"variant": name, "N": N, **summarize(data[name][N], N, rng, keys)})
    q = {"statement": "does any twist variant beat lattice_smc_notwist at N = 32 with an interval excluding zero, and does the oracle",
         "variants_beating_notwist_N32": [p["variant"] for p in pairs if p["N"] == 32 and p["vs"] == "notwist" and p["separates"] and p["diff"] > 0],
         "variants_worse_than_notwist_N32": [p["variant"] for p in pairs if p["N"] == 32 and p["vs"] == "notwist" and p["separates"] and p["diff"] < 0],
         "oracle_vs_notwist_N32": next((p for p in pairs if p["N"] == 32 and p["variant"].startswith("oracle") and p["vs"] == "notwist"), None),
         "oracle_vs_notwist_N8": next((p for p in pairs if p["N"] == 8 and p["variant"].startswith("oracle") and p["vs"] == "notwist"), None)}
    q["met_any_variant"] = len(q["variants_beating_notwist_N32"]) > 0
    q["met_oracle"] = bool(q["oracle_vs_notwist_N32"] and q["oracle_vs_notwist_N32"]["separates"] and q["oracle_vs_notwist_N32"]["diff"] > 0)
    out = {"rows": rows, "pairs": pairs, "oracle_same_sequences": oracle_rows, "question": q}
    json.dump(out, open(os.path.join(RESULTS_DIR, "phase3b_analysis.json"), "w"), indent=1)
    cols = ["variant", "N", "n_sequences", "R_mean", "ci_lo", "ci_hi", "frac_resampled_k1", "frac_resampled_k2", "frac_resampled_k3",
            "ess_over_N_k1", "ess_over_N_k2", "ess_over_N_k3", "final_particle_mean", "final_particle_max", "twist_evals", "wall_s"]
    fmt = {"frac_resampled_k1": "{:.2f}", "frac_resampled_k2": "{:.2f}", "frac_resampled_k3": "{:.2f}", "ess_over_N_k1": "{:.3f}",
           "ess_over_N_k2": "{:.3f}", "ess_over_N_k3": "{:.3f}", "twist_evals": "{:.0f}", "wall_s": "{:.2f}"}
    md = "# Phase 3b tables (twist variants on R_rep, alpha = 0.02)\n\n## All variants, all N (full 40 prompts x 4 seeds, oracle on its 10 prompts x 4 seeds)\n\n"
    md += md_table(rows, cols, fmt)
    md += "\n## Paired differences vs lattice_smc_notwist (variant - notwist, same sequences, bootstrap over prompts)\n\n"
    md += md_table(pairs, ["variant", "vs", "N", "n_prompts", "n_sequences", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Oracle diagnostic: the same 40 sequences (10 prompts x 4 seeds)\n\n"
    md += md_table(oracle_rows, cols + ["oracle_mc_se_nats_mean_k1", "oracle_mc_se_nats_mean_k2", "oracle_mc_se_nats_max", "oracle_nfe_per_sequence"],
                   {**fmt, "oracle_nfe_per_sequence": "{:.0f}"})
    md += "\n## Pre-registered question\n\n```\n" + json.dumps(q, indent=1) + "\n```\n"
    open(os.path.join(RESULTS_DIR, "phase3b_tables.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
