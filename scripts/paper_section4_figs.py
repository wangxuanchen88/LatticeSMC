"""Section 4 data figures for the paper (fig4_curves, fig5_crossover, fig6_cost, fig7_temperature) as vector PDFs under
paper/figures/. Numbers are read only from the stored analysis outputs (results/u/u_analysis.json, results/w/w_analysis.json,
results/x/x_analysis.json, results/m4/m4_analysis.json, results/phase2b_combined.json); nothing is recomputed from samples.
Every plotted value is also written to paper/figures/NUMBERS_CHECK.md with its source file and key.

  .venv/bin/python scripts/paper_section4_figs.py
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    "lines.linewidth": 1.0, "lines.markersize": 3.2,
})
W = 5.5
COL = {"bon": "#C0504D", "greedy": "#C9A227", "fk": "#8E6BC4", "lattice": "#0E8A8A", "base": "#7F7F7F"}
NAME = {"bon": "best-of-N", "greedy": "chunk pruning", "fk": "FK steering", "lattice": "LatticeSMC"}
RULE = {"draw": dict(ls="-", marker="o", mfc=None), "argmax": dict(ls="--", marker="s", mfc="white")}
NS = [1, 2, 4, 8, 16, 32]
BAND = 0.16
LOG = []  # (figure, panel, series, N or x, value, lo, hi, source)


def J(name):
    return json.load(open(os.path.join(RES, name)))


U, WA, X, M4, P2B = J("u/u_analysis.json"), J("w/w_analysis.json"), J("x/x_analysis.json"), J("m4/m4_analysis.json"), J("phase2b_combined.json")


# ----------------------------------------------------------------------------------------------- data access
def dance_series(grid, method, rule):
    """(N, value, lo, hi, scale) from u_analysis dance_rows: full reward of the returned sequence where it exists
    (draw rows; regenerated argmax rows; bon_is-argmax = bon_argmax), else the logged argmax value (additive scale)."""
    out = []
    for r in U["dance_rows"]:
        if r["grid"] != grid or r["method"] != method or r["rule"] != rule:
            continue
        if not (isinstance(r["reward_full"], float) and np.isnan(r["reward_full"])):
            out.append((r["N"], r["reward_full"], r["full_lo"], r["full_hi"], "full"))
        else:
            out.append((r["N"], r["reward_additive"], r["add_lo"], r["add_hi"], "additive (logs)"))
    return sorted(out)


def music_series(method, rule):
    return sorted((r["N"], r["reward"], r["lo"], r["hi"], "exact") for r in U["music_rows"] if r["method"] == method and r["rule"] == rule)


def motif_series(method, rule):
    name = method + ("-argmax" if rule == "argmax" and method not in ("bon_argmax", "greedy_chunk") else "")
    return sorted((r["N"], r["R_mean"], r["ci_lo"], r["ci_hi"], "exact") for r in M4["rows"] if r["method"] == name)


def log(fig, panel, series, s, source):
    for N, v, lo, hi, scale in s:
        LOG.append((fig, panel, series, N, v, lo, hi, f"{source} [{scale}]"))


# ----------------------------------------------------------------------------------------------- helpers
def nfe_axis(ax, per_n, Ns=NS, label=True):
    ax.set_xscale("log")
    ticks = [per_n * n for n in Ns]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) if i % 2 == 0 else "" for i, t in enumerate(ticks)])  # every other label: the N row below carries the rest
    ax.set_xticks([], minor=True)
    if label:
        ax.set_xlabel("denoiser evaluations")
    sec = ax.secondary_xaxis(-0.30)
    sec.set_xscale("log")
    sec.set_xticks(ticks)
    sec.set_xticklabels([f"N={n}" if i == 0 else str(n) for i, n in enumerate(Ns)])
    sec.set_xticks([], minor=True)
    sec.tick_params(length=0, labelsize=6.5, pad=1)
    sec.spines["bottom"].set_visible(False)
    return sec


def plot_series(ax, s, key, rule, per_n, band=True, z=2):
    x = np.array([per_n * t[0] for t in s])
    y, lo, hi = (np.array([t[i] for t in s]) for i in (1, 2, 3))
    st = RULE[rule]
    ax.plot(x, y, color=COL[key], ls=st["ls"], marker=st["marker"], mfc=COL[key] if st["mfc"] is None else "white", mec=COL[key], mew=0.8, zorder=z)
    if band:
        ax.fill_between(x, lo, hi, color=COL[key], alpha=BAND, lw=0, zorder=1)


def shared_legend(fig, keys, rules=("draw", "argmax"), base=False, ncol=None, y=1.0):
    handles = [Line2D([], [], color=COL[k], ls="-", marker="o", mec=COL[k], label=NAME[k]) for k in keys]
    if base:
        handles.append(Line2D([], [], color=COL["base"], ls=":", label="base model (N = 1)"))
    for r in rules:
        st = RULE[r]
        handles.append(Line2D([], [], color="0.2", ls=st["ls"], marker=st["marker"], mfc="0.2" if st["mfc"] is None else "white", mec="0.2",
                              label="sampling (weighted draw)" if r == "draw" else "search (argmax)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles), handlelength=2.2, columnspacing=1.2, handletextpad=0.5)


# ----------------------------------------------------------------------------------------------- Figure 4
def fig4():
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.05))
    panels = [("beat alignment (4M, α = 0.02)", "R_BA alpha 0.02", 400, "dance"), ("repetition (4M, α = 0.02)", "R_rep alpha 0.02", 400, "dance"),
              ("prompt adherence (music)", None, 32, "clap"), ("motif recurrence (music)", None, 32, "motif")]
    ylabels = ["beat alignment R_BA", "repetition R_rep", "terminal CLAP", "motif recurrence R_motif"]
    for ax, (title, grid, per_n, kind), yl in zip(axes, panels, ylabels):
        if kind == "dance":
            get = lambda m, r: dance_series(grid, m, r)  # noqa: E731
            src = "u_analysis.json dance_rows"
            lat = "lattice_smc_notwist"
        elif kind == "clap":
            get = music_series
            src = "u_analysis.json music_rows"
            lat = "lattice_smc_prefix_b1"
        else:
            get = motif_series
            src = "m4_analysis.json rows"
            lat = "lattice_smc_prefix_b1"
        base = get("base", "-")
        ax.axhline(base[0][1], color=COL["base"], ls=":", lw=0.9, zorder=1)
        log("fig4", title, "base", base, src)
        for m, key, rule in [("bon_is", "bon", "draw"), ("bon_argmax", "bon", "argmax"), ("greedy_chunk", "greedy", "argmax"), ("fk_noise", "fk", "draw"),
                             ("fk_noise", "fk", "argmax"), (lat, "lattice", "draw"), (lat, "lattice", "argmax")]:
            s = get(m, rule)
            s = [t for t in s if t[0] >= 1]
            if not s:
                continue
            # N = 1 of every method equals base (bitwise): start the curves at N = 1 where the row exists
            plot_series(ax, s, key, rule, per_n, band=(m != "bon_argmax" or True))
            log("fig4", title, f"{m} ({rule})", s, src)
        nfe_axis(ax, per_n)
        ax.set_ylabel(yl, labelpad=2)
        ax.tick_params(axis="y", pad=1.5)
    shared_legend(fig, ["bon", "greedy", "fk", "lattice"], base=True, ncol=4, y=1.02)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.80, wspace=0.55)
    fig.savefig(os.path.join(OUT, "fig4_curves.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- Figure 5
def fig5():
    C6 = {"R_BA": "#E69F00", "R_rep": "#56B4E9", "CLAP": "#009E73", "motif": "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.25), gridspec_kw={"width_ratios": [1, 1]})
    lines = [  # (panel, label, colour, dashed, series)
        (0, "beat alignment, 4M", C6["R_BA"], False, WA["crossover"]["R_BA alpha 0.02"], "lattice_smc_notwist|argmax|", "w_analysis.json crossover R_BA alpha 0.02"),
        (0, "beat alignment, 38M", C6["R_BA"], True, X["R_BA alpha 0.02"]["L (37.6 M)"]["crossover"], "lattice_smc_notwist|argmax|", "x_analysis.json R_BA alpha 0.02 / L crossover"),
        (0, "repetition, 4M", C6["R_rep"], False, WA["crossover"]["R_rep alpha 0.02"], "lattice_smc_notwist|argmax|", "w_analysis.json crossover R_rep alpha 0.02"),
        (0, "repetition, 38M", C6["R_rep"], True, X["R_rep alpha 0.02"]["L (37.6 M)"]["crossover"], "lattice_smc_notwist|argmax|", "x_analysis.json R_rep alpha 0.02 / L crossover"),
        (1, "prompt adherence", C6["CLAP"], False, WA["crossover"]["music terminal CLAP"], "lattice_smc_prefix_b1|argmax|", "w_analysis.json crossover music terminal CLAP"),
        (1, "motif recurrence", C6["motif"], False, WA["crossover"]["music R_motif"], "lattice_smc_prefix_b1|argmax|", "w_analysis.json crossover music R_motif"),
    ]
    handles = [[], []]
    for panel, label, col, dashed, cs, prefix, src in lines:
        per_n = 400 if panel == 0 else 32
        pts = [(N, cs[f"{prefix}{N}"]["diff"], cs[f"{prefix}{N}"]["ci_lo"], cs[f"{prefix}{N}"]["ci_hi"], "paired diff") for N in NS[1:] if f"{prefix}{N}" in cs]
        x = np.array([per_n * p[0] for p in pts])
        y, lo, hi = (np.array([p[i] for p in pts]) for i in (1, 2, 3))
        ax = axes[panel]
        ax.plot(x, y, color=col, ls="--" if dashed else "-", marker="s" if dashed else "o", mfc="white" if dashed else col, mec=col, mew=0.8, zorder=2)
        ax.fill_between(x, lo, hi, color=col, alpha=BAND, lw=0, zorder=1)
        handles[panel].append(Line2D([], [], color=col, ls="--" if dashed else "-", marker="s" if dashed else "o", mfc="white" if dashed else col, mec=col, label=label))
        log("fig5", "dance" if panel == 0 else "music", label, pts, src)
    for ax, per_n in zip(axes, (400, 32)):
        ax.axhline(0, color="0.3", lw=0.6, zorder=0)
        nfe_axis(ax, per_n, Ns=NS[1:])
    axes[0].set_ylabel("LatticeSMC (argmax) − pruning")
    axes[1].set_ylabel("LatticeSMC (argmax) − pruning")
    fig.legend(handles=handles[0] + handles[1], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, handlelength=2.2, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.82, wspace=0.35)
    fig.savefig(os.path.join(OUT, "fig5_crossover.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- Figure 6
def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.9))
    rows = {}
    for m, key, rule in [("bon_argmax", "bon", "argmax"), ("greedy_chunk", "greedy", "argmax"), ("fk_noise", "fk", "argmax"), ("lattice_smc_prefix_b1", "lattice", "argmax")]:
        rows[key] = sorted([r for r in U["music_rows"] if r["method"] == m and r["rule"] == rule], key=lambda r: r["N"])
    axes_spec = [("nfe", "denoiser evaluations"), ("total_calls", "total model calls"), ("wall_s", "wall-clock per sequence (s)")]
    ymin = min(r["lo"] for rs in rows.values() for r in rs)
    ymax = max(r["hi"] for rs in rows.values() for r in rs)
    pad = 0.04 * (ymax - ymin)
    for ax, (xkey, xlabel) in zip(axes, axes_spec):
        for key, rs in rows.items():
            x = np.array([r[xkey] for r in rs])
            y, lo, hi = (np.array([r[k] for r in rs]) for k in ("reward", "lo", "hi"))
            ax.plot(x, y, color=COL[key], ls="--", marker="s", mfc="white", mec=COL[key], mew=0.8, zorder=2)
            ax.fill_between(x, lo, hi, color=COL[key], alpha=BAND, lw=0, zorder=1)
            log("fig6", xlabel, f"{key} (argmax) x={xkey}", [(r["N"], r["reward"], r["lo"], r["hi"], f"x={r[xkey]:.6g}") for r in rs], "u_analysis.json music_rows")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylim(ymin - pad, ymax + pad)
        if xkey == "nfe":
            nfe_axis(ax, 32)
        else:
            ax.set_xticks([], minor=True)
    axes[0].set_ylabel("terminal CLAP (argmax rule)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    shared_legend(fig, ["bon", "greedy", "fk", "lattice"], rules=("argmax",), ncol=5, y=1.02)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.30, top=0.86, wspace=0.18)
    fig.savefig(os.path.join(OUT, "fig6_cost.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- Figure 7
def fig7():
    alphas = [0.2, 0.05, 0.02]
    grids = {0.2: "R_BA alpha 0.2", 0.05: "R_BA alpha 0.05", 0.02: "R_BA alpha 0.02"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.25))
    ax = axes[0]
    for m, key, rule in [("bon_is", "bon", "draw"), ("fk_noise", "fk", "draw"), ("lattice_smc_notwist", "lattice", "draw"), ("lattice_smc_notwist", "lattice", "argmax"), ("greedy_chunk", "greedy", "argmax")]:
        pts = []
        for a in alphas:
            s = [t for t in dance_series(grids[a], m, rule) if t[0] == 32]
            pts.append((a, s[0][1], s[0][2], s[0][3], s[0][4]))
        x = np.array([p[0] for p in pts])
        y, lo, hi = (np.array([p[i] for p in pts]) for i in (1, 2, 3))
        st = RULE[rule]
        ax.plot(x, y, color=COL[key], ls=st["ls"], marker=st["marker"], mfc=COL[key] if st["mfc"] is None else "white", mec=COL[key], mew=0.8, zorder=2)
        ax.fill_between(x, lo, hi, color=COL[key], alpha=BAND, lw=0, zorder=1)
        log("fig7", "reward vs alpha, N = 32", f"{m} ({rule})", pts, "u_analysis.json dance_rows (N = 32)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("α (sharper tilt →)")
    ax.set_ylabel("beat alignment R_BA at N = 32")
    ax = axes[1]
    for m, key in [("lattice_smc_notwist", "lattice"), ("fk_noise", "fk")]:
        pts = []
        for a in alphas:
            r = [x for x in P2B["resampling_activity"] if x["alpha"] == a and x["method"] == m and x["N"] == 32][0]
            pts.append((a, r["frac_events_resampled"], r["frac_events_resampled"], r["frac_events_resampled"], "fraction of events with ESS < N/2"))
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        ax.plot(x, y, color=COL[key], ls="-", marker="o", mec=COL[key], zorder=2)
        log("fig7", "resampling activity, N = 32", f"{m} fraction of events resampled", pts, "phase2b_combined.json resampling_activity")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_xticks([], minor=True)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("α (sharper tilt →)")
    ax.set_ylabel("fraction of resampling events fired")
    shared_legend(fig, ["bon", "greedy", "fk", "lattice"], ncol=3, y=1.02)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.82, wspace=0.35)
    fig.savefig(os.path.join(OUT, "fig7_temperature.pdf"))
    plt.close(fig)


def write_log():
    lines = ["# NUMBERS_CHECK: values plotted in the Section 4 figures, with their sources", "",
             "The paper draft (`paper/` with its Section 4 text, Tables 2-4, seven captions and the three PowerPoint figures) was not present in the "
             "repository or anywhere under the home directory when this file was generated (2026-09-18), so the text/table/caption comparison the task asked "
             "for could not be performed. This file lists every number that the four data figures draw, read from the analysis files named in each row, so that the "
             "comparison can be made against the draft once it is available. Nothing was recomputed from samples.", "",
             "Conventions: dance 'full' = full reward of the returned sequence; 'additive (logs)' = the logged argmax value on the steering scale (used only where no "
             "regenerated sequence exists, i.e. FK steering argmax at every N and LatticeSMC argmax at N not in {8, 32} on R_BA; identical to full on R_rep). "
             "Music values are exact from the logs. Intervals are 95 % bootstrap over the 40 prompts. LatticeSMC on the dance panels is the no-twist ablation "
             "(lattice_smc_notwist, the method of the pre-registered Amendment U / W / X cells, psi = 1); on the music panels it is lattice_smc_prefix(beta = 1). "
             "The fig7 right panel shows the fraction of resampling events fired only: the mean ESS / N before each boundary per alpha is not in "
             "`results/phase2b_combined.json` or `results/phase2b_tables.md` (it is in the per-alpha `phase2b_*_analysis.json` files, outside the allowed set), so it was not drawn.", "",
             "| figure | panel | series | N or x | value | lo | hi | source |", "|---|---|---|---|---|---|---|---|"]
    for fig, panel, series, N, v, lo, hi, src in LOG:
        lines.append(f"| {fig} | {panel} | {series} | {N} | {v:.4f} | {lo:.4f} | {hi:.4f} | {src} |")
    open(os.path.join(OUT, "NUMBERS_CHECK.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig4()
    fig5()
    fig6()
    fig7()
    write_log()
    print("wrote", sorted(os.listdir(OUT)))
