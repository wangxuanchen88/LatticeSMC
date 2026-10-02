"""Amendment W: consolidated tables and crossover figures. For every reward (R_BA at alpha 0.2 / 0.05 / 0.02, R_rep at
0.02 / 0.05, music terminal CLAP, music R_motif) one table with both return rules for every method at N in {8, 32}, the
held-out columns, diversity and the pre-registered pairwise cells; one crossover figure per reward (lattice_smc(-argmax)
minus greedy_chunk against N, paired bootstrap over prompts); the corrected GPU-hours per phase in the same file.

Every number is read from the stored analysis files of the phase that produced it (phase2/2b/3 analysis JSONs, results/u,
results/m3, results/m4) so the consolidated tables repeat the reported values; only the crossover curves at N not in
{8, 32} are computed here, from the stored logs (dance: the additive steering scale of the logs for both rules, exact for
R_rep; music: exact), with the regenerated full-reward cells overlaid at N = 8 and 32. Writes results/w/consolidated.md,
results/w/w_analysis.json, results/w/fig_crossover_<tag>.{pdf,png}. Nothing under the earlier outputs is modified.

  .venv/bin/python scripts/analyze_w.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", "."))
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/scripts")
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_u import GRIDS, dance_logs, music_logs, paired  # noqa: E402
from lattice_smc.analyze_phase2 import RESULTS_DIR  # noqa: E402

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
OUT = os.path.join(RESULTS_DIR, "w")
os.makedirs(OUT, exist_ok=True)
SEED = 0
NS = [1, 2, 4, 8, 16, 32]
DANCE_METHODS = ["base", "bon_argmax", "bon_is", "greedy_chunk", "fk_noise", "lattice_smc", "lattice_smc_notwist"]
MUSIC_METHODS = ["base", "bon_argmax", "bon_is", "greedy_chunk", "fk_noise", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"]
TAG = {"R_BA alpha 0.2": "rba_a02", "R_BA alpha 0.05": "rba_a005", "R_BA alpha 0.02": "rba_a002", "R_rep alpha 0.02": "rrep_a002", "R_rep alpha 0.05": "rrep_a005",
       "music terminal CLAP": "music_clap", "music R_motif": "music_motif"}


def J(p):
    return json.load(open(p))


def f(x, d=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(m, lo, hi, d=4):
    return "-" if m is None or (isinstance(m, float) and np.isnan(m)) else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------- dance
def dance_table(label, tag, u):
    ana = J(os.path.join(RESULTS_DIR, f"{tag}_analysis.json"))
    curve = {(c["method"], c["N"]): c for c in ana["curve"]}
    held = {(h["method"], h["N"]): h for N in ana["heldout"] for h in ana["heldout"][N]}
    div = {(d["method"], d["N"]): d["pairwise_joint_dist_root_relative_m"] for d in ana["diversity"]}
    urows = {(r["method"], r["N"], r["rule"]): r for r in u["dance_rows"] if r["grid"] == label}
    base = curve[("base", 1)]
    hb, db = held[("base", 1)], div[("base", 1)]
    rows = [{"method": "base", "N": 1, "rule": "-", "reward (full)": ci(base["ba_full_mean"], base["ci_lo"], base["ci_hi"]), "reward (additive, logs)": f(base["ba_additive_mean"]),
             "PFC": f(hb["pfc"]["mean"], 2), "realism W1": f(hb["realism_w1"]["mean"], 5), "travel m": f(hb["travel_m"]["mean"], 2), "diversity (root-rel.)": f(db)}]
    for N in (8, 32):
        for m in DANCE_METHODS[1:]:
            c, h = curve[(m, N)], held[(m, N)]
            rule = "argmax (by definition)" if m in ("bon_argmax", "greedy_chunk") else "draw"
            rows.append({"method": m, "N": N, "rule": rule, "reward (full)": ci(c["ba_full_mean"], c["ci_lo"], c["ci_hi"]), "reward (additive, logs)": f(c["ba_additive_mean"]),
                         "PFC": f(h["pfc"]["mean"], 2), "realism W1": f(h["realism_w1"]["mean"], 5), "travel m": f(h["travel_m"]["mean"], 2), "diversity (root-rel.)": f(div.get((m, N)))})
            if m in ("bon_is", "fk_noise", "lattice_smc", "lattice_smc_notwist"):
                r = urows.get((m, N, "argmax"))
                if r is None:
                    continue
                regen = r.get("regenerated", False)
                if m == "bon_is":  # bon_is-argmax is bon_argmax's returned sample (same particle set, Phase U)
                    cb, hb2 = curve[("bon_argmax", N)], held[("bon_argmax", N)]
                    rows.append({"method": m, "N": N, "rule": "argmax (= bon_argmax)", "reward (full)": ci(cb["ba_full_mean"], cb["ci_lo"], cb["ci_hi"]), "reward (additive, logs)": f(cb["ba_additive_mean"]),
                                 "PFC": f(hb2["pfc"]["mean"], 2), "realism W1": f(hb2["realism_w1"]["mean"], 5), "travel m": f(hb2["travel_m"]["mean"], 2), "diversity (root-rel.)": f(div.get(("bon_argmax", N)))})
                    continue
                rows.append({"method": m, "N": N, "rule": "argmax" + (" (regenerated)" if regen else " (logs only)"),
                             "reward (full)": ci(r["reward_full"], r["full_lo"], r["full_hi"]) if regen else "-",
                             "reward (additive, logs)": ci(r["reward_additive"], r["add_lo"], r["add_hi"]),
                             "PFC": f(r.get("pfc"), 2), "realism W1": f(r.get("realism_w1"), 5), "travel m": "-", "diversity (root-rel.)": f(r.get("diversity_root_rel"))})
    pairs = []
    for p in ana["pairwise_N8_N32"]:
        if p.get("primary"):  # stored as A - B; presented as lattice_smc - other
            flip = p["B"] == "lattice_smc"
            pairs.append({"pair": f"lattice_smc - {p['A']}" if flip else f"{p['A']} - {p['B']}", "rule / scale": "draw, full", "N": p["N"],
                          "diff [95 % CI]": ci(-p["diff_mean"], -p["ci_hi"], -p["ci_lo"]) if flip else ci(p["diff_mean"], p["ci_lo"], p["ci_hi"]),
                          "separates": p["separates"], "source": f"{tag}_analysis.json"})
    for p in u["dance_pairs"]:
        if p["grid"] == label:
            pairs.append({"pair": f"{p['A']} - {p['B']}", "rule / scale": p["scale"], "N": p["N"], "diff [95 % CI]": ci(p["diff"], p["ci_lo"], p["ci_hi"]), "separates": p["separates"], "source": "u_analysis.json"})
    pairs.sort(key=lambda r: (r["pair"], r["N"]))
    return rows, pairs, {"gt_reference": ana.get("gt_reference")}


def dance_crossover(label, root, key, rng):
    """Per N: paired per-prompt difference lattice - greedy on the logs' steering scale (additive R_BA, exact R_rep), both
    rules, for lattice_smc_notwist and lattice_smc; plus the regenerated full-reward difference at N = 8, 32 (from u)."""
    out = {}
    for N in NS[1:]:
        g = {k: v["draw"] for k, v in dance_logs(root, "greedy_chunk", N).items()}
        for m in ("lattice_smc_notwist", "lattice_smc"):
            lg = dance_logs(root, m, N)
            for rule in ("draw", "argmax"):
                out[(m, rule, N)] = paired({k: v[rule] for k, v in lg.items()}, g, rng)
    return out


# ------------------------------------------------------------------------------------------- music
def music_table(kind, u, m4):
    if kind == "clap":
        ana = J(os.path.join(RESULTS_DIR, "m3", "m3_analysis.json"))
        rows_src = {(r["method"], r["N"]): r for r in ana["rows"]}
        arows = {(r["method"], r["N"]): r for r in u["music_rows"] if r["rule"] == "argmax"}
        rkey = "R_mean"
    else:
        ana = m4
        rows_src = {(r["method"], r["N"]): r for r in ana["rows"] if r["rule"] != "argmax" or r["method"] in ("bon_argmax", "greedy_chunk")}
        arows = {(r["method"].replace("-argmax", ""), r["N"]): r for r in ana["rows"] if r["rule"] == "argmax" and r["method"] not in ("bon_argmax", "greedy_chunk")}
        rkey = "R_mean"

    def row(m, N, r, rule):
        d = {"method": m, "N": N, "rule": rule, "reward": ci(r.get(rkey, r.get("reward")), r.get("ci_lo", r.get("lo")), r.get("ci_hi", r.get("hi"))),
             "tempo (n)": f"{f(r.get('tempo_reward'), 3)} ({r.get('tempo_reward_n', '-')})", "8 kHz frac.": f(r.get("frac_energy_above_8k"), 5),
             "aesthetics CE / CU / PC / PQ": " / ".join(f(r.get(k), 2) for k in ("aes_CE", "aes_CU", "aes_PC", "aes_PQ")), "seam ratio": f(r.get("seam_ratio"), 3),
             "seed diversity": f(r.get("seed_diversity_cos_dist", r.get("diversity")))}
        if kind == "motif":
            d["terminal CLAP (held out)"] = f(r.get("clap_terminal"))
            d["s_3"] = f(r.get("s3_chunk3_to_chunk1"), 3)
        return d
    rows = [row("base", 1, rows_src[("base", 1)], "-")]
    for N in (8, 32):
        for m in MUSIC_METHODS[1:]:
            rule = "argmax (by definition)" if m in ("bon_argmax", "greedy_chunk") else "draw"
            rows.append(row(m, N, rows_src[(m, N)], rule))
            if m == "bon_is":
                rows.append(row(m, N, rows_src[("bon_argmax", N)], "argmax (= bon_argmax)"))
            elif m not in ("bon_argmax", "greedy_chunk"):
                r = arows.get((m, N))
                if r is not None:
                    regen = bool(r.get("regenerated", r.get("tempo_reward") is not None))
                    rows.append(row(m, N, r, "argmax" + (" (regenerated)" if regen else " (logs only)")))
    pairs = []
    if kind == "clap":
        for p in ana["pairs"]:
            if p.get("primary"):
                pairs.append({"pair": f"{p['A']} - {p['B']}", "rule": "draw", "N": p["N"], "diff [95 % CI]": ci(p["diff"], p["ci_lo"], p["ci_hi"]), "separates": p["separates"], "source": "m3_analysis.json"})
        for p in u["music_pairs"]:
            pairs.append({"pair": f"{p['A']} - {p['B']}", "rule": "argmax", "N": p["N"], "diff [95 % CI]": ci(p["diff"], p["ci_lo"], p["ci_hi"]), "separates": p["separates"], "source": "u_analysis.json"})
    else:
        for k, p in ana["pre_registered"].items():
            name, N = k.split(" @ N=")
            pairs.append({"pair": name, "rule": "argmax" if "argmax" in name else "draw", "N": int(N), "diff [95 % CI]": ci(p["diff"], p["ci"][0], p["ci"][1]), "separates": bool(p["ci"][0] > 0 or p["ci"][1] < 0), "source": "m4_analysis.json"})
    pairs.sort(key=lambda r: (r["pair"], r["N"]))
    return rows, pairs


def music_crossover(root, rng):
    out = {}
    for N in NS[1:]:
        g = {k: v["draw"] for k, v in music_logs(root, "greedy_chunk", N).items()}
        lg = music_logs(root, "lattice_smc_prefix_b1", N)
        for rule in ("draw", "argmax"):
            out[("lattice_smc_prefix_b1", rule, N)] = paired({k: v[rule] for k, v in lg.items()}, g, rng)
        fk = music_logs(root, "fk_noise", N)
        for rule in ("draw", "argmax"):
            out[("fk_noise", rule, N)] = paired({k: v[rule] for k, v in fk.items()}, g, rng)
    return out


# ----------------------------------------------------------------------------------------- figures
def crossover_figure(tag, title, series, ylabel, overlay=None):
    """series: {label: [(N, diff, lo, hi)]}; overlay: {label: [(N, diff, lo, hi)]} drawn as markers only."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    styles = {"draw": dict(ls="-", marker="o"), "argmax": dict(ls="--", marker="s")}
    colors = {"lattice_smc_notwist": "#9467bd", "lattice_smc": "#8c564b", "lattice_smc_prefix_b1": "#9467bd", "fk_noise": "#2ca02c"}
    for (m, rule), pts in series.items():
        pts = [p for p in pts if p[1] is not None]
        x = [p[0] for p in pts]
        y = np.array([p[1] for p in pts])
        lo = np.array([p[2] for p in pts])
        hi = np.array([p[3] for p in pts])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=colors[m], capsize=2.5, lw=1.4, ms=4.5, label=f"{m} ({rule})", **styles[rule])
    if overlay:
        for (m, rule), pts in overlay.items():
            pts = [p for p in pts if p[1] is not None]
            ax.errorbar([p[0] * 1.08 for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], color="k", fmt="D", ms=5, capsize=2.5, lw=1.2,
                        label=f"{m} ({rule}), full reward, regenerated")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS[1:])
    ax.set_xticklabels([str(n) for n in NS[1:]])
    ax.set_xlabel("N (NFE = 400 N dance, 32 N music)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_crossover_{tag}.{ext}"), dpi=200)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    u = J(os.path.join(RESULTS_DIR, "u", "u_analysis.json"))
    m4 = J(os.path.join(RESULTS_DIR, "m4", "m4_analysis.json"))
    md_out = ["# Consolidated tables (Amendment W, 2026-09-16)", "",
              "One table per reward: every method at N in {8, 32} under both return rules (the weighted draw of the final particles, and the argmax of the "
              "final particles; bon_argmax and greedy_chunk are argmax by definition, bon_is-argmax is bon_argmax's returned sample), the held-out columns "
              "(never used for steering), diversity and the pre-registered pairwise cells. Values are those of the phase analyses (`results/phase*_analysis.json`, "
              "`results/u/u_analysis.json`, `results/m3/m3_analysis.json`, `results/m4/m4_analysis.json`), bootstrap intervals over the 40 prompts (1000 resamples, seed 0). "
              "Dance: 160 sequences per cell (4 seeds); full reward is the reward of the returned sequence, the additive reward is the steering quantity logged per "
              "particle (for R_rep both coincide); argmax rows marked 'logs only' have no regenerated sequence, so their full reward, held-out values and diversity "
              "are not available (Phase U). Music: 80 sequences per cell (2 seeds), rewards exact from the logs; argmax rows of lattice_smc_prefix(1) and fk_noise "
              "regenerated at N in {8, 32} (Phase U for terminal CLAP, Amendment W for R_motif), so their held-out values and diversity are measured.", ""]
    analysis = {"tables": {}, "pairs": {}, "crossover": {}}
    cross_series = {}
    # dance
    dance_cols = ["method", "N", "rule", "reward (full)", "reward (additive, logs)", "PFC", "realism W1", "travel m", "diversity (root-rel.)"]
    pair_cols = ["pair", "rule / scale", "N", "diff [95 % CI]", "separates", "source"]
    for label, root, tag, key, alpha in GRIDS:
        rows, pairs, extra = dance_table(label, tag, u)
        analysis["tables"][label], analysis["pairs"][label] = rows, pairs
        gt = extra["gt_reference"] or {}
        md_out += [f"## {label} (dance; `{root}`; ground truth: {key}_full {f(gt.get(key + '_full', gt.get('ba_full')))}, PFC {f(gt.get('pfc'), 2)}, W1 {f(gt.get('realism_w1_leave_one_out'), 5)})", "",
                   md(rows, dance_cols), "", "Pre-registered pairwise cells (A - B, per-prompt paired difference):", "", md(pairs, pair_cols), ""]
        cs = dance_crossover(label, root, key, rng)
        analysis["crossover"][label] = {f"{m}|{rule}|{N}": v for (m, rule, N), v in cs.items()}
        series = {(m, rule): [(N, cs[(m, rule, N)]["diff"], cs[(m, rule, N)]["ci_lo"], cs[(m, rule, N)]["ci_hi"]) for N in NS[1:] if cs.get((m, rule, N))]
                  for m in ("lattice_smc_notwist", "lattice_smc") for rule in ("draw", "argmax")}
        overlay = {}
        regen = [p for p in u["dance_pairs"] if p["grid"] == label and p["A"].endswith("(regenerated)") and p["B"] == "greedy_chunk"]
        if regen:
            overlay[("lattice_smc_notwist", "argmax")] = [(p["N"], p["diff"], p["ci_lo"], p["ci_hi"]) for p in regen]
        crossover_figure(TAG[label], f"{label}: lattice_smc minus greedy_chunk", series,
                         f"{'R_BA (additive)' if key == 'ba' else 'R_rep'} difference", overlay)
        cross_series[label] = series
    # music
    music_cols = ["method", "N", "rule", "reward", "tempo (n)", "8 kHz frac.", "aesthetics CE / CU / PC / PQ", "seam ratio", "seed diversity"]
    pair_cols_m = ["pair", "rule", "N", "diff [95 % CI]", "separates", "source"]
    for kind, label, root in (("clap", "music terminal CLAP", os.path.join(RESULTS_DIR, "m3", "samples")), ("motif", "music R_motif", os.path.join(RESULTS_DIR, "m4", "samples"))):
        rows, pairs = music_table(kind, u, m4)
        analysis["tables"][label], analysis["pairs"][label] = rows, pairs
        cols = music_cols if kind == "clap" else music_cols[:4] + ["terminal CLAP (held out)", "s_3"] + music_cols[4:]
        md_out += [f"## {label} (Stable Audio 3 harness B2; `{os.path.relpath(root, ROOT)}`)", "", md(rows, cols), "",
                   "Pre-registered pairwise cells (A - B, per-prompt paired difference):", "", md(pairs, pair_cols_m), ""]
        cs = music_crossover(root, rng)
        analysis["crossover"][label] = {f"{m}|{rule}|{N}": v for (m, rule, N), v in cs.items()}
        series = {(m, rule): [(N, cs[(m, rule, N)]["diff"], cs[(m, rule, N)]["ci_lo"], cs[(m, rule, N)]["ci_hi"]) for N in NS[1:] if cs.get((m, rule, N))]
                  for m in ("lattice_smc_prefix_b1", "fk_noise") for rule in ("draw", "argmax")}
        crossover_figure(TAG[label], f"{label}: lattice_smc_prefix(1) and fk_noise minus greedy_chunk", series, f"{'terminal CLAP' if kind == 'clap' else 'R_motif'} difference")
        cross_series[label] = series
    # crossover summary table
    md_out += ["## Crossover: lattice_smc(-argmax) minus greedy_chunk by N (`fig_crossover_<tag>.{pdf,png}`)", "",
               "Paired per-prompt difference with the 95 % bootstrap interval; dance on the logs' steering scale (additive R_BA, exact R_rep; the regenerated "
               "full-reward cells at N = 8, 32 are in the tables above and overlaid in the figures), music exact. '*' marks an interval excluding zero.", ""]
    crows = []
    for label, series in cross_series.items():
        for (m, rule), pts in series.items():
            r = {"reward": label, "method": m, "rule": rule}
            for N, d, lo, hi in pts:
                r[f"N={N}"] = f"{d:+.4f}{'*' if lo > 0 or hi < 0 else ''}"
            crows.append(r)
    md_out += [md(crows, ["reward", "method", "rule"] + [f"N={N}" for N in NS[1:]]), ""]
    # GPU-hours
    import gpu_hours
    gpu_hours.main()
    gh = open(os.path.join(RESULTS_DIR, "GPU_HOURS.md")).read().split("## Per-job walls")[0].strip().split("\n", 1)[1].strip()
    md_out += ["## GPU-hours per phase (corrected; `scripts/gpu_hours.py`, full per-job table in `results/GPU_HOURS.md`)", "", gh, ""]
    open(os.path.join(OUT, "consolidated.md"), "w").write("\n".join(md_out))
    json.dump(analysis, open(os.path.join(OUT, "w_analysis.json"), "w"), indent=1)
    print("\n".join(md_out[-40:]))


if __name__ == "__main__":
    main()
