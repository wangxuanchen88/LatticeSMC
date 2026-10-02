"""R1 Phase 1: paired intervals missing from the paper's Table 1 (no new generation). Dense schedule = fk_noise, boundary
schedule = lattice_smc_notwist (dance) / lattice_smc_prefix_b1 (music), pruning = greedy_chunk. Full reward where a
returned sequence exists (metrics caches, incl. the fk_noise argmax regenerations of 2026-09-18), exact logged values on
R_rep and music. Paired bootstrap over the 40 conditions (1000 resamples, seed 0). Writes results/r1/schedule_pairs.{json,tex}.

  .venv/bin/python scripts/r1_phase1.py
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/scripts")
import numpy as np  # noqa: E402

from analyze_u import dance_logs, met_rows, metrics_for, music_logs, paired  # noqa: E402

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
OUT = os.path.join(ROOT, "results", "r1")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
rows = []


def add(reward, model, A, B, a, b, scale, src):
    p = paired(a, b, rng)
    rows.append({"reward": reward, "model": model, "N": 32, "A": A, "B": B, "scale": scale, "source": src, **p})
    print(f"{reward:16s} {model:4s} {A} - {B}: {p['diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] n={p['n_sequences']} sep={p['separates']}")


def full(tag, method, N=32, key="ba_full"):
    return {k: r[key] for k, r in met_rows(metrics_for(tag), method, N).items()}


# beat alignment, full reward of the returned sequence (all three rows regenerated / stored sequences)
for model, greedy_tag, notwist_tag, fk_tag in (("4M", "phase2b_a002", "u_argmax_p2_a002", "verify_fkargmax_p2_a002"), ("38M", "L_ba_a002", "L_argmax_ba_a002", "verify_fkargmax_L_ba_a002")):
    fk, gr, nt = full(fk_tag, "fk_noise"), full(greedy_tag, "greedy_chunk"), full(notwist_tag, "lattice_smc_notwist")
    src = f"results/{fk_tag}_metrics.json, results/{greedy_tag}_metrics.json, results/{notwist_tag}_metrics.json"
    add("beat alignment", model, "dense argmax", "chunk pruning", fk, gr, "full reward", src)
    add("beat alignment", model, "dense argmax", "boundary argmax", fk, nt, "full reward", src)
# repetition (additive == full; argmax rows exact from the logs)
for model, root, greedy_tag, notwist_tag in (("4M", "samples_p3_rep_a002", "phase3_rep_a002", "u_argmax_p3_rep_a002"), ("38M", "samples_L_rep_a002", "L_rep_a002", "L_argmax_rep_a002")):
    fk = {k: v["argmax"] for k, v in dance_logs(root, "fk_noise", 32).items()}
    gr = full(greedy_tag, "greedy_chunk", key="rep_full")
    nt = full(notwist_tag, "lattice_smc_notwist", key="rep_full")
    src = f"{root}/fk_noise/32 logs (max of per-particle rewards), results/{greedy_tag}_metrics.json, results/{notwist_tag}_metrics.json"
    add("repetition", model, "dense argmax", "chunk pruning", fk, gr, "exact (additive == full)", src)
    add("repetition", model, "dense argmax", "boundary argmax", fk, nt, "exact (additive == full)", src)
# music: dense minus boundary under both rules, exact from the logs
for reward, root in (("prompt adherence", os.path.join(ROOT, "results", "m3", "samples")), ("motif recurrence", os.path.join(ROOT, "results", "m4", "samples"))):
    fk, lat = music_logs(root, "fk_noise", 32), music_logs(root, "lattice_smc_prefix_b1", 32)
    for rule in ("draw", "argmax"):
        add(reward, "music", f"dense {rule}", f"boundary {rule}", {k: v[rule] for k, v in fk.items()}, {k: v[rule] for k, v in lat.items()}, "exact (logs)", f"{os.path.relpath(root, ROOT)}/{{fk_noise,lattice_smc_prefix_b1}}/32 logs")
json.dump({"n_boot": 1000, "seed": 0, "rows": rows}, open(os.path.join(OUT, "schedule_pairs.json"), "w"), indent=1)
tex = ["% paired differences at N = 32, mean [95% bootstrap over 40 conditions]; * = interval excludes zero",
       "\\begin{tabular}{llll}", "\\toprule", "Reward & Model & Comparison (A $-$ B) & Difference [95\\% CI] \\\\", "\\midrule"]
for r in rows:
    tex.append(f"{r['reward']} & {r['model']} & {r['A']} $-$ {r['B']} & ${r['diff']:+.3f}$ $[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]${'$^*$' if r['separates'] else ''} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "schedule_pairs.tex"), "w").write("\n".join(tex) + "\n")
