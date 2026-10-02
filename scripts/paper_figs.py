"""Paper figure set (vector PDF) and LaTeX booktabs tables under results/paper_figs/, read from the existing analysis
JSON files only (nothing under results/ is modified). Caption drafts, stating what was held fixed, go to captions.md.

  .venv/bin/python scripts/paper_figs.py            # PREVIEW_DIR=<dir> also writes PNG previews there
"""
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "paper_figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.25,
                     "lines.linewidth": 1.3, "lines.markersize": 3.5})
COL = {"base": "#7f7f7f", "bon_argmax": "#1f77b4", "bon_is": "#17becf", "fk_noise": "#2ca02c", "greedy_chunk": "#d62728",
       "lattice_smc": "#9467bd", "lattice_smc_notwist": "#8c564b", "lattice_smc_prefix_b1": "#9467bd", "lattice_smc_prefix_b0.5": "#c5b0d5",
       "lattice_smc_prefix_b0.25": "#e377c2", "oracle": "#000000", "rollout": "#ff7f0e"}
LAB = {"base": "base (N = 1)", "bon_argmax": "best-of-N argmax", "bon_is": "best-of-N IS", "fk_noise": "FK noise axis", "greedy_chunk": "greedy chunk (top N/4)",
       "lattice_smc": "lattice SMC, learned twist", "lattice_smc_notwist": "lattice SMC, no twist", "lattice_smc_prefix_b1": "lattice SMC, prefix twist β = 1",
       "lattice_smc_prefix_b0.5": "lattice SMC, β = 0.5", "lattice_smc_prefix_b0.25": "lattice SMC, β = 0.25"}
J = lambda *p: json.load(open(os.path.join(R, *p)))  # noqa: E731
captions = []


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")), dpi=130)


def legend_below(fig, axes, ncol, row_h=0.055):
    """One de-duplicated legend under the panels; the panels are then laid out above it (no overlap with x labels)."""
    seen, handles, labels = set(), [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    nrows = -(-len(labels) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, row_h * nrows + 0.03, 1, 1))


def curve(ax, rows, x_of, methods, key_mean="mean", key_lo="lo", key_hi="hi", key_m="method", ls=None):
    for m in methods:
        pts = sorted([r for r in rows if r[key_m] == m], key=lambda r: r["N"])
        if not pts:
            continue
        x = [x_of(r) for r in pts]
        y = [r[key_mean] for r in pts]
        lo = [r[key_mean] - r[key_lo] for r in pts]
        hi = [r[key_hi] - r[key_mean] for r in pts]
        ax.errorbar(x, y, yerr=[lo, hi], color=COL.get(m, None), marker="o", capsize=1.5, elinewidth=0.7, label=LAB.get(m, m), ls=(ls or {}).get(m, "-"))
    ax.set_xscale("log", base=2)


def base_band(ax, mean, lo, hi):
    ax.axhspan(lo, hi, color="#bbbbbb", alpha=0.35, lw=0)
    ax.axhline(mean, color="#7f7f7f", lw=0.8, label="base (N = 1)")


# ---------------------------------------------------------------- 1. R_BA vs NFE at three alphas
def fig_rba():
    c = J("phase2b_combined.json")["curves"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    methods = ["bon_argmax", "bon_is", "fk_noise", "greedy_chunk", "lattice_smc", "lattice_smc_notwist"]
    for ax, a in zip(axes, ["0.2", "0.05", "0.02"]):
        rows = c[a]
        base = next(r for r in rows if r["method"] == "base")
        base_band(ax, base["mean"], base["lo"], base["hi"])
        curve(ax, rows, lambda r: 400 * r["N"], methods, ls={"lattice_smc_notwist": "--"})
        ax.set_title(f"α = {a}")
        ax.set_xlabel("NFE per sequence")
    axes[0].set_ylabel("R_BA (beat alignment)")
    legend_below(fig, axes, 4)
    save(fig, "fig_rba_alpha.pdf")
    captions.append(("fig_rba_alpha.pdf", "R_BA against NFE at the three tilts (α = 0.2, 0.05, 0.02), mean over 40 prompts × 4 seeds with 1000-resample bootstrap "
                     "intervals over prompts (shaded band: base, N = 1). Held fixed across every curve and panel: the base model (candidate E, `runs/chunk_dispE/ckpt_adopted.pt`, "
                     "step 12000), K = 4 chunks × 150 frames, the DDIM sampler (50 steps, η = 1, guidance 2, ±5 displacement clip), the 40 prompts of `data/prompts.json` "
                     "and seeds 2000–2003, NFE = 400 N denoiser rows for every method, systematic resampling at ESS < N/2, greedy M = max(1, N/4), fk_noise events after DDIM "
                     "steps 10, 20, 30, 40, 50, and the per-(seed, method) resampling generators. Only α and the twist retrained for it (Phase 2a / 2b rollouts, same "
                     "architecture and checkpoint rule) change between panels. Sources: `results/phase2b_combined.json` (Phases 2 and 2b)."))


# ---------------------------------------------------------------- 2. R_rep with oracle and rollout rows
def fig_rrep():
    c = J("phase3_rep_combined.json")["curves"]["0.02"]
    o = J("phase3b_analysis.json")
    d = J("phase3d_analysis.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6))
    ax = axes[0]
    base = next(r for r in c if r["method"] == "base")
    base_band(ax, base["mean"], base["lo"], base["hi"])
    curve(ax, c, lambda r: 400 * r["N"], ["bon_argmax", "bon_is", "fk_noise", "greedy_chunk", "lattice_smc", "lattice_smc_notwist"], ls={"lattice_smc_notwist": "--"})
    ax.set_xlim(300, 20000)
    ax.set_xlabel("NFE per sequence")
    ax.set_ylabel("R_rep (repetition reward)")
    ax.set_title("(a) grid, 40 prompts × 4 seeds")
    ax = axes[1]
    for var, mk, col, lab in (("oracle (M = 16)", "*", COL["oracle"], "oracle twist, M = 16 continuations (continuation NFE not charged)"),
                              ("notwist", "o", COL["lattice_smc_notwist"], "lattice SMC, no twist (same 40 sequences)"),
                              ("learned (beta 1)", "s", COL["lattice_smc"], "lattice SMC, learned twist (same 40 sequences)")):
        pts = sorted([r for r in o["oracle_same_sequences"] if r["variant"] == var], key=lambda r: r["N"])
        ax.errorbar([400 * r["N"] for r in pts], [r["R_mean"] for r in pts], yerr=[[r["R_mean"] - r["ci_lo"] for r in pts], [r["ci_hi"] - r["R_mean"] for r in pts]],
                    fmt=mk + "-", color=col, ms=8 if mk == "*" else 4, capsize=1.5, elinewidth=0.7, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlim(2000, 25000)
    ax.set_xlabel("NFE per sequence (budget only)")
    ax.set_title("(b) oracle twist, 10 prompts × 4 seeds")
    ax = axes[2]
    for M, mk in ((1, "s"), (2, "D"), (4, "^")):
        rr = sorted([r for r in d["rollout_rows"] if r["variant"] == f"rollout M={M}"], key=lambda r: r["N"])
        want = {4 * (4 + 5 * M) // 4, 8 * (4 + 5 * M) // 4}
        mm = sorted([r for r in d["rollout_rows"] if r["variant"].startswith("notwist matched") and int(r["variant"].split("N'=")[1].rstrip(")")) in want], key=lambda r: r["N"])
        nfe = lambda r: r["nfe"][0] if isinstance(r["nfe"], list) else r["nfe"]  # noqa: E731
        ax.errorbar([nfe(r) for r in rr], [r["R_mean"] for r in rr], yerr=[[r["R_mean"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["R_mean"] for r in rr]], fmt=mk + "-",
                    color=COL["rollout"], capsize=1.5, elinewidth=0.7, label=f"rollout lookahead M = {M} (N = 4, 8)")
        ax.errorbar([nfe(r) for r in mm], [r["R_mean"] for r in mm], yerr=[[r["R_mean"] - r["ci_lo"] for r in mm], [r["ci_hi"] - r["R_mean"] for r in mm]], fmt=mk + "--",
                    color=COL["lattice_smc_notwist"], capsize=1.5, elinewidth=0.7, mfc="white", label=f"no twist at matched N' = {', '.join(str(r['N']) for r in mm)}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE per sequence, continuations charged")
    ax.set_title("(c) lookahead vs matched N'")
    legend_below(fig, axes, 3)
    save(fig, "fig_rrep.pdf")
    captions.append(("fig_rrep.pdf", "R_rep against NFE at α = 0.02. (a) The compute-matched grid (40 prompts × 4 seeds, NFE = 400 N, bootstrap intervals over prompts) with the "
                     "learned twist (`data/twist_rep/twist_a002.pt`, MLP on the last 30 root-relative prefix frames, trained on 4,000 twist-pool prefixes × 16 continuations) and the "
                     "no-twist ablation. (b) The oracle twist (M = 16 base-model continuations per particle at chunks 1 and 2, ψ₃ = 1) on the first 10 prompts × 4 seeds, with the "
                     "no-twist and learned-twist runs restricted to the same 40 sequences (these prompts have a lower R_rep level than the 40-prompt mean); the oracle's continuation "
                     "NFE (410–610× the sequence budget) is not charged on the x axis. (c) Rollout lookahead with M ∈ {1, 2, 4} continuations per particle at N ∈ {4, 8}, every "
                     "continuation counted in the sequence NFE (100 N (4 + 5M)), against the no-twist ablation at the NFE-matched particle count N' = N (4 + 5M)/4 (exact) on the "
                     "same prompts and seeds. Held fixed everywhere: base model E, K = 4, DDIM 50 steps, α = 0.02, τ = 0.1706 m in R_rep, ESS < N/2 systematic resampling, seeds "
                     "2000–2003, prompts, generators. Sources: `results/phase3_rep_combined.json`, `results/phase3b_analysis.json`, `results/phase3d_analysis.json` (Phases 3, 3b, 3d)."))


# ---------------------------------------------------------------- 3. music terminal CLAP with the beta family
def fig_music():
    rows = J("m3", "m3_analysis.json")["rows"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    base = next(r for r in rows if r["method"] == "base")
    base_band(ax, base["R_mean"], base["ci_lo"], base["ci_hi"])
    curve(ax, rows, lambda r: 32 * r["N"], ["bon_argmax", "bon_is", "fk_noise", "greedy_chunk", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"],
          key_mean="R_mean", key_lo="ci_lo", key_hi="ci_hi", ls={"lattice_smc_prefix_b0.5": "--", "lattice_smc_prefix_b0.25": ":"})
    ax.set_xlabel("NFE per sequence (transformer rows)")
    ax.set_ylabel("terminal CLAP")
    legend_below(fig, ax, 2)
    save(fig, "fig_music_clap.pdf")
    captions.append(("fig_music_clap.pdf", "Music testbed: terminal CLAP of the returned 40 s clip against NFE, mean over 40 prompts × 2 seeds (2000, 2001) with bootstrap intervals over "
                     "prompts; the lattice SMC family uses the handcrafted prefix twist s_k = prefix CLAP with strength β ∈ {1, 0.5, 0.25} (potentials exp(β (s_k − s_{k−1})/α), "
                     "terminal exp((R − β s_{K−1})/α)). Held fixed: harness B2 (Stable Audio 3 medium, 8 ping-pong steps, cfg 1.0, caller-side replacement inpainting on the latent, "
                     "decoder seed pinned per (prompt, seed), raw decode), K = 4 × 10 s, α = 0.006762 from `results/m2/alpha.json` (base samples only, never revisited), NFE = 32 N "
                     "transformer rows for every method, ESS < N/2 systematic resampling, greedy M = max(1, N/4), fk_noise events after steps 2, 4, 6, 8 of each chunk with the "
                     "CLAP of the decoded clamped x0 prediction, per-slot noise seeds shared by all methods, the CLAP checkpoint (LAION music_audioset_epoch_15_esc_90.14). "
                     "Decoder and CLAP calls (N for best-of-N, 4N for greedy and lattice, 16N for fk_noise) are not charged on the x axis. Seeds 2002–2003 were dropped by the "
                     "pre-registered 24 GPU-hour rule. Source: `results/m3/m3_analysis.json` (Phase M3); every terminal CLAP was recomputed from the saved audio (max difference 0)."))


# ---------------------------------------------------------------- 4. ESS traces
def fig_ess():
    runs = {"R_BA, α = 0.2": J("phase2_rerun_analysis.json"), "R_BA, α = 0.05": J("phase2b_a005_analysis.json"), "R_BA, α = 0.02": J("phase2b_a002_analysis.json"),
            "R_rep, α = 0.02": J("phase3_rep_a002_analysis.json")}
    m3 = J("m3", "m3_analysis.json")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9))
    ax = axes[0]
    for (name, a), mk in zip(runs.items(), ("o", "s", "D", "^")):
        for m, ls in (("lattice_smc", "-"), ("lattice_smc_notwist", "--")):
            pts = sorted([r for r in a["ess_content"] if r["method"] == m and r["N"] == 32], key=lambda r: r["chunk"])
            if pts:
                ax.plot([r["chunk"] for r in pts], [r["mean_ess_over_N"] for r in pts], marker=mk, ls=ls, label=f"{name}, {'learned twist' if m == 'lattice_smc' else 'no twist'}")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("chunk boundary k")
    ax.set_ylabel("mean ESS / N before the event (N = 32)")
    ax.set_title("(a) content axis, dance")
    ax.set_xticks([1, 2, 3])
    ax = axes[1]
    for (name, a), mk in zip(runs.items(), ("o", "s", "D", "^")):
        pts = sorted([r for r in a["ess_noise"] if r["method"] == "fk_noise" and r["N"] == 32], key=lambda r: (r["chunk"], r["step"]))
        if pts:
            ax.plot([(r["chunk"] - 1) * 50 + r["step"] for r in pts], [r["mean_ess_over_N"] for r in pts], marker=mk, ms=2.5, lw=0.9, label=f"{name}, FK noise axis")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    for k in (50, 100, 150):
        ax.axvline(k, color="#999999", lw=0.5)
    ax.set_xlabel("DDIM step (50 per chunk)")
    ax.set_title("(b) noise axis, dance")
    ax = axes[2]
    for m, mk in (("lattice_smc_prefix_b1", "o"), ("lattice_smc_prefix_b0.5", "s"), ("lattice_smc_prefix_b0.25", "D"), ("greedy_chunk", "x")):
        r = next(x for x in m3 if x["method"] == m and x["N"] == 32)
        ax.plot([1, 2, 3], [r["ess_over_N_k1"], r["ess_over_N_k2"], r["ess_over_N_k3"]], marker=mk, color=COL[m], label="music: " + LAB[m] + (" (degenerate)" if m == "greedy_chunk" else ""))
    r = next(x for x in m3 if x["method"] == "fk_noise" and x["N"] == 32)
    ax.plot([1, 2, 3, 4], [r[f"noise_ess_over_N_k{k}"] for k in (1, 2, 3, 4)], marker="^", color=COL["fk_noise"], ls="--", label="music: FK noise axis (mean over the 4 events of a chunk)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xlabel("chunk k")
    ax.set_title("(c) music, N = 32")
    ax.set_xticks([1, 2, 3, 4])
    legend_below(fig, axes, 3)
    save(fig, "fig_ess.pdf")
    captions.append(("fig_ess.pdf", "Effective sample size before each resampling event, mean over the 160 (dance) or 80 (music) sequences at N = 32, as a fraction of N; the dotted line is the "
                     "ESS < N/2 threshold at which every proper method resamples systematically. (a) Content axis (lattice SMC with and without the twist) on R_BA at α = 0.2, 0.05, "
                     "0.02 and on R_rep at α = 0.02; on R_rep the ablation has uniform weights at chunks 1–2 (r₁ = r₂ = 0) while the learned twist collapses the set there. (b) "
                     "Noise axis, fk_noise at its five events per chunk. (c) Music: prefix-twist lattice at β = 1, 0.5, 0.25, greedy (degenerate, ESS = N/4 by construction) and "
                     "fk_noise (per-chunk mean over its four events). Held fixed: everything listed for the corresponding reward curve (base model, sampler, prompts, seeds, NFE, "
                     "α, defaults); the ESS is read from the method logs of the same runs, no re-generation. Sources: `results/phase2_rerun_analysis.json`, "
                     "`results/phase2b_a00{5,2}_analysis.json`, `results/phase3_rep_a002_analysis.json`, `results/m3/m3_analysis.json`."))


# ---------------------------------------------------------------- 5. twist calibration and within-set correlation
def fig_twist():
    ba = J("twist_calibration_a002.json")
    rep = J("rep_twist_calibration_a002.json")
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.7))
    for ax, (name, t) in zip(axes[:2], (("R_BA", ba), ("R_rep", rep))):
        b = t["binned_learned"]
        ax.errorbar([x["pred_mean"] for x in b], [x["real_mean"] for x in b], yerr=[x["real_se"] for x in b], fmt="o-", color=COL["lattice_smc"], capsize=1.5,
                    label="learned twist, 8 prediction bins (mean ± s.e. of the realized target)")
        lo = min(min(x["pred_mean"] for x in b), min(x["real_mean"] for x in b))
        hi = max(max(x["pred_mean"] for x in b), max(x["real_mean"] for x in b))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.6, ls=":", label="identity")
        f = t["learned_twist_vs_future_target"]
        pk = t["learned_twist_per_k"]
        ax.set_title(f"{name}, α = 0.02\nR² {f['r2']:.2f}; within k: {pk['1']['r2']:.2f} / {pk['2']['r2']:.2f} / {pk['3']['r2']:.2f}", fontsize=7)
        ax.set_xlabel("predicted log ψ")
        ax.set_ylabel("realized log mean exp(S/α)")
    ax = axes[2]
    steps = [45, 35, 25, 15, 5]
    for name, t, ls in (("R_BA", ba, "--"), ("R_rep", rep, "-")):
        ax.plot(steps, [t["plugin"][str(s)]["vs_next_chunk_reward_same_continuation"]["r2"] for s in steps], marker="o", color=COL["fk_noise"], ls=ls,
                label=f"{name}: plug-in vs its own continuation's next-chunk reward")
        ax.plot(steps, [t["plugin"][str(s)]["vs_future_target_k3_only"]["r2"] for s in steps], marker="s", color=COL["lattice_smc"], ls=ls,
                label=f"{name}: plug-in vs realized future at k = 3")
        ax.axhline(t["learned_twist_per_k"]["3"]["r2"], color=COL["lattice_smc"], lw=0.7, ls=":" if name == "R_BA" else "-.",
                   label=f"{name}: learned twist vs realized future at k = 3")
    ax.set_xlabel("DDIM step of the x0 prediction")
    ax.set_ylabel("R²")
    ax.set_ylim(-1.05, 1.05)
    ax.invert_xaxis()
    ax.set_title("plug-in noise-axis estimate", fontsize=7.5)
    ax = axes[3]
    files = sorted(glob.glob(os.path.join(ROOT, "data", "twist_rep", "oracle_sets", "*.npz")))
    rs = {1: [], 2: []}
    for f in files:
        z = np.load(f)
        k = int(z["k"])
        l, o = z["logpsi_learned"], z["logpsi_oracle"]
        rs[k].append(float(np.corrcoef(l, o)[0, 1]))
        ax.scatter(o - o.mean(), l - l.mean(), s=6, alpha=0.6, color=COL["lattice_smc"] if k == 1 else COL["greedy_chunk"])
    for k, c in ((1, COL["lattice_smc"]), (2, COL["greedy_chunk"])):
        ax.scatter([], [], s=12, color=c, label=f"same particles, k = {k}: mean within-set r = {np.mean(rs[k]):+.2f}")
    ax.set_xlabel("oracle log ψ (centred, nats)")
    ax.set_ylabel("learned log ψ (centred, nats)")
    ax.set_title("learned vs oracle log ψ", fontsize=7.5)
    legend_below(fig, axes, 2)
    save(fig, "fig_twist_calibration.pdf")
    captions.append(("fig_twist_calibration.pdf", "Twist calibration and the within-set diagnostic. Panels 1–2: binned calibration of the learned twist on the 500 fresh prefixes "
                     "(fresh base sequences on twist-pool music, fresh continuations: 8 for R_BA, 16 for R_rep; predicted-bin mean against the realized log mean exp(S/α), "
                     "error bars = standard error of the bin mean), α = 0.02, with the overall and within-chunk-index R². Panel 3: R² of the plug-in noise-axis estimate "
                     "r̂_t/α from the x0 prediction at DDIM step t of the next chunk (continuation 0 of the same fresh prefixes) against that continuation's own next-chunk "
                     "reward and against the realized future at k = 3 (the equal-horizon case); horizontal lines: the learned twist's within-k = 3 R². Panel 4: the Phase 3 "
                     "learned twist against the oracle twist (M = 16 continuations, ~0.6 nats delta-method error) evaluated on the same 32 particles of the same lattice runs "
                     "(first 5 prompts, seed 2000, chunks 1 and 2; each set centred), whose within-set correlation is −0.04 at k = 1 and +0.35 at k = 2. Held fixed: base model E, "
                     "twist architecture (4,638 → 512 → 512 → 1) and checkpoint rule (lowest validation MSE), α = 0.02, the fresh set, the oracle seeds. Sources: "
                     "`results/twist_calibration_a002.json`, `results/rep_twist_calibration_a002.json`, `data/twist_rep/oracle_sets/*.npz` (Phases 2b, 3, 3c/3d)."))


# ---------------------------------------------------------------- 6. held-out tables (LaTeX booktabs)
def tex_table(path, caption, label, header, rows, colspec):
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open(path, "w").write("\n".join(lines))


def tables():
    esc = lambda s: s.replace("_", "\\_")  # noqa: E731
    for tag, fname, reward_key, rname, phase in (("phase2b_a002", "tab_heldout_rba.tex", "ba_full", "$R_{BA}$", "Phase 2b"),
                                                 ("phase3_rep_a002", "tab_heldout_rrep.tex", "rep_full", "$R_{\\mathrm{rep}}$", "Phase 3")):
        a = J(f"{tag}_analysis.json")
        gt = a["gt_reference"]
        rep = reward_key == "rep_full"
        rows, seen = [], set()
        for N in ("8", "32"):
            for r in a["heldout"][N]:
                key = (r["method"], r["N"])
                if key in seen:
                    continue
                seen.add(key)
                row = [esc(r["method"]), str(r["N"]), f"{r[reward_key]['mean']:.3f}"]
                if rep:
                    row.append(f"{r['ba_full']['mean']:.3f}")
                row += [f"{r['pfc']['mean']:.2f} [{r['pfc']['ci_lo']:.2f}, {r['pfc']['ci_hi']:.2f}]", f"{1000 * r['realism_w1']['mean']:.2f}", f"{r['travel_m']['mean']:.2f}",
                        " / ".join(f"{x:.1f}" for x in r["mean_disp_mm"]), f"{r['root_oob_frac']:.3f}"]
                rows.append(row)
        rows.sort(key=lambda r: (r[1] != "1", int(r[1]), r[0]))
        gtrow = ["ground truth", "--", f"{gt['rep_full']:.3f}" if rep else f"{gt['ba_full']:.3f}"]
        if rep:
            gtrow.append(f"{gt['ba_full']:.3f}")
        gtrow += [f"{gt['pfc']:.2f}", f"{1000 * gt['realism_w1_leave_one_out']:.2f}", f"{gt['travel_m']:.2f}", " / ".join(f"{x:.1f}" for x in gt["mean_disp_mm"]), "0"]
        header = ["method", "$N$", rname] + (["$R_{BA}$ (held out)"] if rep else []) + ["PFC $\\times 10^4$ [CI]", "realism $W_1 \\times 10^3$", "travel (m)", "mean disp.\\ x/y/z (mm/frame)", "root o.o.r."]
        cap = (f"Held-out rewards (never used for steering) at $\\alpha = 0.02$ on the dance testbed, {rname} steering, $N = 8$ and $32$ (40 prompts $\\times$ 4 seeds; base at $N = 1$). "
               "Held fixed: base model E, $K = 4$, DDIM 50 steps, NFE $= 400N$, the prompts, seeds, defaults and generators of the corresponding grid; PFC and the realism statistic "
               "use the Amendment I pipeline (no root clip); travel and mean displacement are the Amendment J covariates; root o.o.r.\\ is the fraction of frames whose integrated root "
               "leaves the normalizer range. Ground truth: the 40 prompts' recorded motion.")
        tex_table(os.path.join(OUT, fname), cap, "tab:heldout_rba" if not rep else "tab:heldout_rrep", header, [gtrow] + rows, "l" + "r" * (len(header) - 1))
        captions.append((fname, cap + f" Source: `results/{tag}_analysis.json` ({phase})."))
    rows = [r for r in J("m3", "m3_analysis.json")["rows"] if r["N"] in (1, 8, 32) and not (r["N"] == 1 and r["method"] != "base")]
    out = []
    for r in rows:
        f = lambda k, fmt="{:.3f}": (fmt.format(r[k]) if k in r else "--")  # noqa: E731
        out.append([esc(r["method"]), str(r["N"]), f("R_mean"), f("tempo_reward") + (f" ({r.get('tempo_reward_n', '')})" if "tempo_reward" in r else ""), f("tempo_whole_within_5pct"),
                    f("frac_energy_above_8k", "{:.4f}"), " / ".join(f("aes_" + k, "{:.2f}") for k in ("CE", "CU", "PC", "PQ")), f("seam_ratio"), f("seed_diversity_cos_dist", "{:.4f}")])
    header = ["method", "$N$", "terminal CLAP", "tempo reward ($n$)", "clip tempo $\\pm 5\\%$", "energy $> 8$ kHz", "aesthetics CE / CU / PC / PQ", "seam ratio", "seed diversity"]
    cap = ("Music testbed, held-out quantities (never used for steering) at $N = 1, 8, 32$ (40 prompts $\\times$ 2 seeds). Tempo reward: beat\\_this on the four 10 s windows of the "
           "returned clip, fraction of chunks within 5\\% of the prompted BPM, $n$ = sequences with all four chunks measurable; aesthetics: audiobox-aesthetics; seam ratio: spectral-flux "
           "ratio at the three chunk seams of the raw decode; seed diversity: mean cosine distance between the terminal CLAP embeddings of the two seeds of a prompt. Held fixed: harness B2, "
           "$\\alpha = 0.006762$, NFE $= 32N$, prompts, seeds, defaults and per-slot noise seeds of the Phase M3 grid.")
    tex_table(os.path.join(OUT, "tab_heldout_music.tex"), cap, "tab:heldout_music", header, out, "lrrrrrrrr")
    captions.append(("tab_heldout_music.tex", cap + " Source: `results/m3/m3_analysis.json` (Phase M3)."))


def main():
    fig_rba()
    fig_rrep()
    fig_music()
    fig_ess()
    fig_twist()
    tables()
    md = ["# Paper figure set: caption drafts", "", "Every caption states what was held fixed. Files are vector PDFs (fonttype 42) and LaTeX booktabs tables produced by "
          "`scripts/paper_figs.py` from the analysis JSON files; nothing under `results/` other than this directory was written.", ""]
    for name, cap in captions:
        md += [f"## `{name}`", "", cap, ""]
    open(os.path.join(OUT, "captions.md"), "w").write("\n".join(md))
    print("\n".join(f"{n}: {len(c)} chars" for n, c in captions))


if __name__ == "__main__":
    main()
