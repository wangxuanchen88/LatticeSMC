"""R1 Phase 2 analysis: chunk pruning with M in {1, N/4 (stored), N/2} at N = 32, argmax return, all four rewards.
Reward, held-out metrics, paired differences against the stored M = N/4 row and against the boundary-schedule argmax row
(paired bootstrap over conditions, 1000 resamples, seed 0). Writes results/r1/pruning_sweep.{json,tex}.

  .venv/bin/python scripts/r1_phase2_analysis.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/scripts")
import numpy as np  # noqa: E402

from analyze_u import boot, dance_logs, met_rows, metrics_for, music_logs, paired, per_prompt, seed_div  # noqa: E402

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
R1 = os.path.join(ROOT, "results", "r1")
rng = np.random.default_rng(0)
rows, pairs = [], []


def mean_ci(d):
    m, lo, hi = boot(list(per_prompt(d).values()), rng)
    return {"mean": m, "ci": [lo, hi]}


# ---------------------------------------------------------------------------------------------- dance
for reward, key, base_tag, argmax_tag in (("beat alignment", "ba", "phase2b_a002", "u_argmax_p2_a002"), ("repetition", "rep", "phase3_rep_a002", "u_argmax_p3_rep_a002")):
    full_key = f"{key}_full"
    rowsets = {"M=8 (N/4, stored)": met_rows(metrics_for(base_tag), "greedy_chunk", 32), "M=1": met_rows(metrics_for(f"r1_prune_{key}_M1"), "greedy_chunk", 32),
               "M=16 (N/2)": met_rows(metrics_for(f"r1_prune_{key}_M16"), "greedy_chunk", 32)}
    boundary = met_rows(metrics_for(argmax_tag), "lattice_smc_notwist", 32)
    for name, mr in rowsets.items():
        if not mr:
            print("missing", reward, name); continue
        full = {k: r[full_key] for k, r in mr.items()}
        r = {"reward": reward, "method": f"chunk pruning {name}", "N": 32, "rule": "argmax", "n": len(mr), "reward_full": mean_ci(full),
             "pfc": float(np.mean([x["pfc"] for x in mr.values()])), "realism_w1": float(np.mean([x["realism_w1"] for x in mr.values()])), "travel_m": float(np.mean([x["travel_m"] for x in mr.values()])),
             "nfe": sorted({x["nfe"] for x in mr.values()}), "source": f"results/{base_tag if 'stored' in name else 'r1_prune_' + key + '_M' + name.split()[0][2:]}_metrics.json"}
        # diversity across seeds (root-relative), from the logs' motions is a GPU computation in analyze_u; use the stored
        # diversity for the stored row and compute for the new roots with root_rel_diversity when a GPU is free
        rows.append(r)
        if "stored" not in name:
            pairs.append({"reward": reward, "A": f"chunk pruning {name}", "B": "chunk pruning M=8 (N/4, stored)", "N": 32, **paired(full, {k: x[full_key] for k, x in rowsets["M=8 (N/4, stored)"].items()}, rng)})
        pairs.append({"reward": reward, "A": f"chunk pruning {name}", "B": "boundary schedule argmax (stored)", "N": 32, **paired(full, {k: x[full_key] for k, x in boundary.items()}, rng)})
# ---------------------------------------------------------------------------------------------- music
S = lambda p: {r["key"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else {}  # noqa: E731
for reward, grid, tag in (("prompt adherence", "m3", "clap"), ("motif recurrence", "m4", "motif")):
    stored_root = os.path.join(ROOT, "results", grid, "samples")
    rkey = "clap_terminal_recomputed" if tag == "clap" else "motif_recomputed"
    stored = music_logs(stored_root, "greedy_chunk", 32)
    boundary_root = os.path.join(ROOT, "results", grid, "samples_argmax")
    bnd = music_logs(boundary_root, "lattice_smc_prefix_b1", 32)
    sets = {"M=8 (N/4, stored)": (stored, S(os.path.join(ROOT, "results", grid, "scores.json")), S(os.path.join(ROOT, "results", grid, "tempo.json")), S(os.path.join(ROOT, "results", grid, "aesthetics.json")))}
    for M, mt in (("1", "M1"), ("16 (N/2)", "MN2")):
        root = os.path.join(R1, f"samples_prune_{tag}_{mt}")
        sets[f"M={M}"] = (music_logs(root, "greedy_chunk", 32), S(os.path.join(R1, f"prune_{tag}_{mt}_scores.json")), S(os.path.join(R1, f"prune_{tag}_{mt}_tempo.json")), S(os.path.join(R1, f"prune_{tag}_{mt}_aesthetics.json")))
    for name, (lg, sc, tp, ae) in sets.items():
        if not lg:
            print("missing", reward, name); continue
        rew = {k: v["draw"] for k, v in lg.items()}  # greedy returns its argmax; final_reward is exact
        held = {}
        for lab, src, field in (("tempo_reward", tp, "tempo_reward_additive"), ("frac_energy_above_8k", sc, "frac_energy_above_8k"), ("seam_ratio", sc, "seam_mean_ratio"), ("aes_PQ", ae, "PQ"), ("clap_terminal", sc, "clap_terminal_recomputed")):
            vals = [src[v["key"]][field] for v in lg.values() if v["key"] in src and src[v["key"]].get(field) is not None]
            if vals:
                held[lab] = float(np.mean(vals)); held[lab + "_n"] = len(vals)
        rows.append({"reward": reward, "method": f"chunk pruning {name}", "N": 32, "rule": "argmax", "n": len(lg), "reward_full": mean_ci(rew), "seed_diversity": seed_div(lg),
                     "decode_calls": float(np.mean([v["decode_calls"] for v in lg.values()])), "nfe": sorted({v["nfe"] for v in lg.values()}), **held,
                     "source": "results/{}/samples/greedy_chunk/32 logs".format(grid) if "stored" in name else f"results/r1/samples_prune_{tag}_{'M1' if name == 'M=1' else 'MN2'} logs"})
        if "stored" not in name:
            pairs.append({"reward": reward, "A": f"chunk pruning {name}", "B": "chunk pruning M=8 (N/4, stored)", "N": 32, **paired(rew, {k: v["draw"] for k, v in stored.items()}, rng)})
        pairs.append({"reward": reward, "A": f"chunk pruning {name}", "B": "boundary schedule argmax (stored)", "N": 32, **paired(rew, {k: v["argmax"] for k, v in bnd.items()}, rng)})
json.dump({"n_boot": 1000, "seed": 0, "rows": rows, "pairs": pairs}, open(os.path.join(R1, "pruning_sweep.json"), "w"), indent=1)
tex = ["% chunk pruning M sweep at N = 32 (argmax return): reward [95% CI over conditions], paired differences (* = interval excludes zero)", "\\begin{tabular}{llcll}", "\\toprule",
       "Reward & M & Reward [95\\% CI] & $-$ M = N/4 & $-$ boundary argmax \\\\", "\\midrule"]
for r in rows:
    d1 = next((p for p in pairs if p["reward"] == r["reward"] and p["A"] == r["method"] and "M=8" in p["B"]), None)
    d2 = next((p for p in pairs if p["reward"] == r["reward"] and p["A"] == r["method"] and "boundary" in p["B"]), None)
    f = lambda p: "" if p is None else f"${p['diff']:+.3f}$ $[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]${'$^*$' if p['separates'] else ''}"  # noqa: E731
    tex.append(f"{r['reward']} & {r['method'].replace('chunk pruning ', '')} & ${r['reward_full']['mean']:.3f}$ $[{r['reward_full']['ci'][0]:.3f}, {r['reward_full']['ci'][1]:.3f}]$ & {f(d1)} & {f(d2)} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(R1, "pruning_sweep.tex"), "w").write("\n".join(tex) + "\n")
for r in rows:
    print(r["reward"], r["method"], round(r["reward_full"]["mean"], 4), r["reward_full"]["ci"], {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
for p in pairs:
    print(p["reward"], p["A"], "-", p["B"], f"{p['diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]", p["separates"])
