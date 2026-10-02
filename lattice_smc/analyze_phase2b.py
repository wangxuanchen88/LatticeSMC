"""Phase 2b (Amendment K): combine the per-alpha analyses (results/<tag>_analysis.json from
analyze_phase2.py) into the pre-registered comparison table, the "resampling active" summary,
and one combined figure (reward vs NFE per alpha; diversity vs reward at N = 32 across methods
and alphas).

  python -m lattice_smc.analyze_phase2b --runs 0.2:phase2,0.02:phase2b_a002,0.05:phase2b_a005
"""
import argparse
import json
import os

import numpy as np

from lattice_smc.analyze_phase2 import METHODS, NS, RESULTS_DIR, md_table

PRIMARY_SETS = {"phase2b": [("lattice_smc", "fk_noise"), ("lattice_smc", "lattice_smc_notwist"), ("lattice_smc", "greedy_chunk"),
                            ("bon_is", "bon_argmax")],
                "phase3": [("lattice_smc", "lattice_smc_notwist"), ("lattice_smc", "greedy_chunk"), ("lattice_smc", "fk_noise"),
                           ("lattice_smc", "bon_argmax")]}
PRIMARY = PRIMARY_SETS["phase2b"]
PALETTE = {"base": "#6b6b6b", "bon_argmax": "#4e79a7", "bon_is": "#76b7b2", "fk_noise": "#e15759",
           "greedy_chunk": "#f28e2b", "lattice_smc": "#59a14f", "lattice_smc_notwist": "#b6992d"}


def pair_row(analysis, a, b, N):
    for p in analysis["pairwise_all_N"]:
        if p["N"] == N and {p["A"], p["B"]} == {a, b}:
            sign = 1 if p["A"] == a else -1
            return {"diff": sign * p["diff_mean"], "lo": min(sign * p["ci_lo"], sign * p["ci_hi"]),
                    "hi": max(sign * p["ci_lo"], sign * p["ci_hi"]), "separates": p["separates"]}
    return None


def resampling_summary(analysis):
    out = {}
    for row in analysis["ess_content"]:
        if row["method"] in ("lattice_smc", "lattice_smc_notwist") and row["chunk"] < 4:
            key = (row["method"], row["N"])
            out.setdefault(key, []).append(row["frac_resampled"])
    for row in analysis["ess_noise"]:
        out.setdefault(("fk_noise", row["N"]), []).append(row["frac_resampled"])
    return {f"{m}|{N}": float(np.mean(v)) for (m, N), v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0.2:phase2,0.02:phase2b_a002,0.05:phase2b_a005")
    ap.add_argument("--primary", default="phase2b", choices=sorted(PRIMARY_SETS))
    ap.add_argument("--out", default="phase2b", help="results/<out>_combined.json, <out>_tables.md, fig_<out>_combined.png")
    ap.add_argument("--reward_label", default="R_BA")
    args = ap.parse_args()
    global PRIMARY
    PRIMARY = PRIMARY_SETS[args.primary]
    runs = []
    for item in args.runs.split(","):
        alpha, tag = item.split(":")
        path = os.path.join(RESULTS_DIR, f"{tag}_analysis.json")
        if os.path.exists(path):
            runs.append((float(alpha), tag, json.load(open(path))))
    runs.sort(key=lambda r: -r[0])
    # pre-registered comparisons
    rows = []
    for alpha, tag, an in runs:
        for a, b in PRIMARY:
            for N in (8, 32):
                r = pair_row(an, a, b, N)
                if r:
                    rows.append({"alpha": alpha, "pair": f"{a} - {b}", "N": N, "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                                 "separates": r["separates"]})
    # resampling activity
    act = []
    for alpha, tag, an in runs:
        rs = resampling_summary(an)
        spread = {(s["method"], s["N"]): s for s in an.get("particle_reward_spread", [])}
        for m in ("lattice_smc", "lattice_smc_notwist", "fk_noise"):
            for N in NS[1:]:
                sp = spread.get((m, N), {})
                act.append({"alpha": alpha, "method": m, "N": N, "frac_events_resampled": rs.get(f"{m}|{N}", float("nan")),
                            "mean_reward_range": sp.get("mean_range", float("nan")),
                            "max_weight_ratio": sp.get("max_weight_ratio_exp_range_over_alpha", float("nan"))})
    # curves per alpha
    curves = {}
    for alpha, tag, an in runs:
        for c in an["curve"]:
            curves.setdefault(alpha, []).append({"method": c["method"], "N": c["N"], "mean": c["ba_full_mean"], "lo": c["ci_lo"], "hi": c["ci_hi"]})
    # diversity vs reward at N = 32
    scatter = []
    for alpha, tag, an in runs:
        div = {(d["method"], d["N"]): d for d in an["diversity"]}
        for c in an["curve"]:
            if c["N"] == 32 or c["method"] == "base":
                d = div.get((c["method"], 32 if c["method"] != "base" else 1))
                if d:
                    scatter.append({"alpha": alpha, "method": c["method"], "reward": c["ba_full_mean"],
                                    "diversity_root_relative": d["pairwise_joint_dist_root_relative_m"]})
    first_active = None
    for alpha, tag, an in sorted(runs, key=lambda r: -r[0]):
        rs = resampling_summary(an)
        if any(v > 0 for v in rs.values()):
            first_active = {"alpha": alpha, "fractions": rs}
            break
    out = {"runs": [(a, t) for a, t, _ in runs], "primary_comparisons": rows, "resampling_activity": act,
           "curves": {str(k): v for k, v in curves.items()}, "scatter_N32": scatter,
           "sharpest_alpha_with_any_resampling": first_active}
    json.dump(out, open(os.path.join(RESULTS_DIR, f"{args.out}_combined.json"), "w"), indent=1)

    md = f"# {args.out} combined tables\n\n## Pre-registered comparisons (A - B, mean {args.reward_label}, bootstrap over prompts)\n\n"
    md += md_table(rows, ["alpha", "pair", "N", "diff", "ci_lo", "ci_hi", "separates"], {"alpha": "{:g}"})
    md += "\n## Resampling activity (fraction of events with ESS < N/2), particle reward range and implied max weight ratio\n\n"
    md += md_table(act, ["alpha", "method", "N", "frac_events_resampled", "mean_reward_range", "max_weight_ratio"], {"alpha": "{:g}", "frac_events_resampled": "{:.3f}", "max_weight_ratio": "{:.3g}"})
    md += "\n## Reward vs NFE per alpha (mean [CI])\n\n"
    for alpha in sorted(curves, reverse=True):
        md += f"\n### alpha = {alpha:g}\n\n" + md_table(curves[alpha], ["method", "N", "mean", "lo", "hi"])
    md += "\n## Diversity (root-relative, m) vs reward at N = 32\n\n" + md_table(scatter, ["alpha", "method", "reward", "diversity_root_relative"], {"alpha": "{:g}"})
    open(os.path.join(RESULTS_DIR, f"{args.out}_tables.md"), "w").write(md)
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(runs)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 4.6))
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes[:n], sorted(curves, reverse=True)):
        for m in METHODS:
            pts = sorted([c for c in curves[alpha] if c["method"] == m], key=lambda c: c["N"])
            if not pts:
                continue
            x = [400 * c["N"] for c in pts]
            if m == "base":
                ax.axhline(pts[0]["mean"], color=PALETTE[m], lw=1.2, ls="--", label="base")
                ax.axhspan(pts[0]["lo"], pts[0]["hi"], color=PALETTE[m], alpha=0.10, lw=0)
            else:
                ax.plot(x, [c["mean"] for c in pts], marker="o", ms=3.5, lw=1.5, color=PALETTE[m], label=m)
                ax.fill_between(x, [c["lo"] for c in pts], [c["hi"] for c in pts], color=PALETTE[m], alpha=0.14, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([400 * k for k in NS])
        ax.set_xticklabels([f"{400 * k}\nN={k}" for k in NS], fontsize=7)
        ax.set_title(f"alpha = {alpha:g}")
        ax.set_xlabel("NFE per sequence")
        ax.grid(True, axis="y", color="#dddddd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"mean {args.reward_label} (40 prompts x 4 seeds)")
    ymin = min(c["lo"] for v in curves.values() for c in v) - 0.005
    ymax = max(c["hi"] for v in curves.values() for c in v) + 0.005
    for ax in axes[:n]:
        ax.set_ylim(ymin, ymax)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    ax = axes[n]
    markers = {0.2: "o", 0.05: "s", 0.02: "^"}
    for s in scatter:
        ax.scatter(s["reward"], s["diversity_root_relative"], color=PALETTE[s["method"]], marker=markers.get(s["alpha"], "o"),
                   s=42, edgecolor="white", lw=0.6)
    for a, mk in markers.items():
        if any(s["alpha"] == a for s in scatter):
            ax.scatter([], [], color="#444444", marker=mk, s=42, label=f"alpha = {a:g}")
    for m in METHODS:
        if any(s["method"] == m for s in scatter):
            ax.scatter([], [], color=PALETTE[m], marker="o", s=42, label=m)
    ax.set_xlabel(f"mean {args.reward_label} at N = 32 (base at N = 1)")
    ax.set_ylabel("root-relative diversity across 4 seeds (m)")
    ax.set_title("diversity vs reward, N = 32")
    ax.grid(True, color="#dddddd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"fig_{args.out}_combined.png"), dpi=140)


if __name__ == "__main__":
    main()
