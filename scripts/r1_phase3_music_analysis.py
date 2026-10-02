"""R1 Phase 3 (music) analysis: K = 8 (80 s, rolling 40 s context) prompt adherence at N in {8, 32}; best-of-N (draw /
argmax), chunk pruning, boundary schedule (draw / argmax, the argmax clip saved alongside the draw); denoiser rows,
decoder and CLAP calls, wall, held-out (tempo, 8 kHz, aesthetics, seam, diversity), paired differences at N = 32, and the
K = 4 rows of the same prompts from the paper grid. Writes results/r1/music_k8.{json,tex} and results/r1/fig_k8.pdf (two
panels: dance beat alignment K = 4 / 6 / 8 and music prompt adherence K = 4 / 8, reward against denoiser evaluations).

  .venv/bin/python scripts/r1_phase3_music_analysis.py
"""
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/scripts")
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analyze_u import boot, met_rows, metrics_for, music_logs, paired, per_prompt, seed_div  # noqa: E402

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
R1 = os.path.join(ROOT, "results", "r1")
K8 = os.path.join(R1, "samples_music_k8")
rng = np.random.default_rng(0)
S = lambda p: {r["key"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else {}  # noqa: E731


def rows_music(root, scores, tempo, aes, methods, label):
    rows, seqs = [], {}
    for N in (8, 32):
        for name, m, rule in methods:
            lg = music_logs(root, m, N)
            if not lg:
                continue
            rew = {k: v[rule] for k, v in lg.items()}
            mean, lo, hi = boot(list(per_prompt(rew).values()), rng)
            embs = {}
            if rule == "argmax" and m == "lattice_smc_prefix_b1":
                for k, v in lg.items():
                    p = os.path.join(os.path.dirname(v["file"]) if "file" in v else os.path.join(root, v["key"]), "clap_embed_argmax.npy")
                    if os.path.exists(p):
                        embs[k] = {"emb": np.load(p)}
            held = {}
            src_key = lambda v: v["key"]  # noqa: E731
            for lab, src, field in (("tempo_reward", tempo, "tempo_reward_additive"), ("tempo_whole_within5", tempo, "tempo_whole_within_5pct"), ("frac_energy_above_8k", scores, "frac_energy_above_8k"), ("seam_ratio", scores, "seam_mean_ratio"), ("aes_PQ", aes, "PQ")):
                vals = [src[src_key(v)][field] for v in lg.values() if src_key(v) in src and src[src_key(v)].get(field) is not None]
                if vals:
                    held[lab] = float(np.mean(vals)); held[lab + "_n"] = len(vals)
            rows.append({"grid": label, "method": name, "N": N, "rule": rule, "n": len(lg), "reward": mean, "ci": [lo, hi], "nfe": sorted({v["nfe"] for v in lg.values()}),
                         "decode_calls": float(np.mean([v["decode_calls"] for v in lg.values()])), "clap_calls": float(np.mean([v["clap_calls"] for v in lg.values()])),
                         "wall_s": float(np.mean([v["wall"] for v in lg.values()])), "seed_diversity": seed_div(embs) if embs else seed_div(lg), **held})
            seqs[(name, N)] = rew
    return rows, seqs


M = [("best-of-N (draw)", "bon_is", "draw"), ("best-of-N (argmax)", "bon_argmax", "argmax"), ("chunk pruning", "greedy_chunk", "argmax"), ("boundary (draw)", "lattice_smc_prefix_b1", "draw"), ("boundary (argmax)", "lattice_smc_prefix_b1", "argmax")]
rows8, seqs8 = rows_music(K8, S(os.path.join(K8, "scores.json")), S(os.path.join(K8, "tempo.json")), S(os.path.join(K8, "aesthetics.json")), M, "K = 8")
m3 = os.path.join(ROOT, "results", "m3")
rows4, seqs4 = rows_music(os.path.join(m3, "samples"), S(os.path.join(m3, "scores.json")), S(os.path.join(m3, "tempo.json")), S(os.path.join(m3, "aesthetics.json")), M, "K = 4")
pairs = []
for label, seqs in (("K = 8", seqs8), ("K = 4", seqs4)):
    for A, B in (("boundary (argmax)", "chunk pruning"), ("boundary (argmax)", "best-of-N (argmax)"), ("boundary (draw)", "best-of-N (draw)"), ("boundary (draw)", "chunk pruning"), ("boundary (argmax)", "boundary (draw)")):
        if (A, 32) in seqs and (B, 32) in seqs:
            pairs.append({"grid": label, "A": A, "B": B, "N": 32, **paired(seqs[(A, 32)], seqs[(B, 32)], rng)})
qual = json.load(open(os.path.join(R1, "qual_k8", "qualification.json"))) if os.path.exists(os.path.join(R1, "qual_k8", "qualification.json")) else None
json.dump({"n_boot": 1000, "seed": 0, "alpha": json.load(open(os.path.join(ROOT, "results", "m2", "alpha.json")))["alpha"], "context_sec": 40, "qualification": qual, "rows_k8": rows8, "rows_k4": rows4, "pairs_N32": pairs},
          open(os.path.join(R1, "music_k8.json"), "w"), indent=1)
tex = ["% music prompt adherence, K = 8 (80 s, rolling 40 s context) vs K = 4; reward [95% CI over 40 prompts]", "\\begin{tabular}{llcccccc}", "\\toprule",
       "Grid & Method & $N$ & Reward [95\\% CI] & dec. calls & wall (s) & tempo & seam \\\\", "\\midrule"]
for r in rows8 + rows4:
    tex.append(f"{r['grid']} & {r['method']} & {r['N']} & ${r['reward']:.3f}$ $[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]$ & {r['decode_calls']:.0f} & {r['wall_s']:.0f} & {r.get('tempo_reward', float('nan')):.3f} & {r.get('seam_ratio', float('nan')):.3f} \\\\")
tex.append("\\midrule")
for p in pairs:
    tex.append(f"{p['grid']} & {p['A']} $-$ {p['B']} & {p['N']} & ${p['diff']:+.3f}$ $[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]${'$^*$' if p['separates'] else ''} & & & & \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(R1, "music_k8.tex"), "w").write("\n".join(tex) + "\n")
for r in rows8 + rows4:
    print(r["grid"], r["method"], r["N"], round(r["reward"], 4), [round(x, 4) for x in r["ci"]], "dec", r["decode_calls"], "wall", round(r["wall_s"]))
for p in pairs:
    print(p["grid"], p["A"], "-", p["B"], f"{p['diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]", p["separates"])

# ------------------------------------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False, "lines.linewidth": 1.0, "lines.markersize": 3.2})
COL = {"bon": "#C0504D", "greedy": "#C9A227", "lattice": "#0E8A8A", "fk": "#8E6BC4", "base": "#7F7F7F"}
RULE = {"draw": dict(ls="-", marker="o"), "argmax": dict(ls="--", marker="s")}
KSTYLE = {4: 1.0, 6: 0.65, 8: 0.35}  # line alpha by horizon
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
# dance panel: beat alignment K = 4 (all 40, paper grid), K = 6 (40), K = 8 (16) at N in {8, 32} (K = 4 also at 1..32)
ax = axes[0]
u = json.load(open(os.path.join(ROOT, "results", "u", "u_analysis.json")))
for m, key, rule in (("bon_argmax", "bon", "argmax"), ("bon_is", "bon", "draw"), ("greedy_chunk", "greedy", "argmax"), ("lattice_smc_notwist", "lattice", "draw"), ("lattice_smc_notwist", "lattice", "argmax"), ("fk_noise", "fk", "draw")):
    pts = sorted((r["N"], r["reward_full"] if r["reward_full"] == r["reward_full"] else r["reward_additive"]) for r in u["dance_rows"] if r["grid"] == "R_BA alpha 0.02" and r["method"] == m and r["rule"] == rule)
    ax.plot([400 * n for n, _ in pts], [v for _, v in pts], color=COL[key], alpha=KSTYLE[4], **RULE[rule])
for K in (6, 8):
    p = os.path.join(R1, f"dance_k{K}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))["rewards"]["ba"]["rows"]
    for name, key, rule in (("best-of-N (argmax)", "bon", "argmax"), ("best-of-N (draw)", "bon", "draw"), ("chunk pruning", "greedy", "argmax"), ("boundary (draw)", "lattice", "draw"), ("boundary (argmax)", "lattice", "argmax"), ("dense (draw)", "fk", "draw")):
        pts = sorted((r["N"], r["reward_full"]) for r in d if r["method"] == name)
        if pts:
            ax.plot([100 * K * n for n, _ in pts], [v for _, v in pts], color=COL[key], alpha=KSTYLE[K], **RULE[rule])
ax.set_xscale("log"); ax.set_xlabel("denoiser evaluations"); ax.set_ylabel("beat alignment (full reward)"); ax.set_title("dance, K = 4 / 6 / 8 (darker = shorter)", fontsize=8)
# music panel
ax = axes[1]
for rows, K in ((rows4, 4), (rows8, 8)):
    for name, key, rule in (("best-of-N (argmax)", "bon", "argmax"), ("best-of-N (draw)", "bon", "draw"), ("chunk pruning", "greedy", "argmax"), ("boundary (draw)", "lattice", "draw"), ("boundary (argmax)", "lattice", "argmax")):
        pts = sorted((r["N"], r["reward"]) for r in rows if r["method"] == name)
        if pts:
            ax.plot([8 * K * n for n, _ in pts], [v for _, v in pts], color=COL[key], alpha=KSTYLE[K], **RULE[rule])
ax.set_xscale("log"); ax.set_xlabel("denoiser evaluations"); ax.set_ylabel("terminal CLAP"); ax.set_title("music, K = 4 / 8 (darker = shorter)", fontsize=8)
handles = [Line2D([], [], color=COL[k], marker="o", label=n) for k, n in (("bon", "best-of-N"), ("greedy", "chunk pruning"), ("fk", "FK steering"), ("lattice", "LatticeSMC"))]
handles += [Line2D([], [], color="0.2", ls="-", marker="o", label="sampling (draw)"), Line2D([], [], color="0.2", ls="--", marker="s", mfc="white", label="search (argmax)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.78, wspace=0.35)
fig.savefig(os.path.join(R1, "fig_k8.pdf"))
print("wrote fig_k8.pdf")
