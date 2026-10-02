"""Phase 2 analysis (SPEC 6), read-only over samples_p2/ and the twist calibration.

  python -m lattice_smc.analyze_phase2 --samples samples_p2 [--metrics_only]

Step 1 computes per-sequence metrics once (resumable cache results/phase2_metrics.json): R_BA
full and additive (Amendment I pipeline: no root clip), PFC, the realism statistic (1-Wasserstein
between the sample's pooled per-frame joint-acceleration magnitudes and the held-out ground-truth
pool), Amendment J covariates (net root travel, mean per-frame displacement per axis), and the
method logs (NFE, reward / twist evaluations, ESS events, final and weighted-mean rewards).
Step 2 writes the tables: reward vs NFE with 1000-resample bootstrap intervals over prompts,
pairwise differences at N = 8 and N = 32 (lattice_smc vs each other method = primary), held-out
rewards at N = 8 and 32, diversity across seeds, ESS traces on both axes, additive-vs-full gap,
and the two pre-stated conditions of SPEC Section 0. Held-out rewards are read here only.
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import torch

from lattice_smc.data.clips import load_normalizer
from lattice_smc.data.pairs import CHUNK_LEN, PromptSet
from lattice_smc.edge_import import EDGE_DIR, ROOT  # noqa: F401
from lattice_smc.rewards.ba import ba_additive, ba_full
from lattice_smc.rewards.kinematics import accel_magnitude, realism_w1, root_covariates, to_joints
from lattice_smc.rewards.pfc import pfc_scaled
from lattice_smc.rewards.rep import TAU_PATH, chunk_distance, load_tau
from vis import SMPLSkeleton  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
REWARD = {"key": "ba_full", "add": "ba_additive"}  # set to rep_full / rep_additive by --reward rep
METHODS = ["base", "bon_argmax", "bon_is", "fk_noise", "greedy_chunk", "lattice_smc", "lattice_smc_notwist"]
NS = [1, 2, 4, 8, 16, 32]
N_BOOT = 1000
BOOT_SEED = 0


# ----------------------------------------------------------------------------- metrics
def compute_metrics(samples_root, device, tag="phase2"):
    """Per-sequence metrics for every npz under samples_root, cached in results/<tag>_metrics.json."""
    cache = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    prompts = PromptSet()
    K = prompts.K
    smpl = SMPLSkeleton(device)
    scale, mn = load_normalizer()
    gt_joints = to_joints(prompts.gt_motion.to(device), smpl, scale, mn, clip_root=True).cpu().numpy()
    gt_pool = np.concatenate([accel_magnitude(j) for j in gt_joints])
    tau = load_tau() if os.path.exists(TAU_PATH) else float("nan")
    files = sorted(glob.glob(os.path.join(samples_root, "*", "*", "*.npz")))
    t0 = time.time()
    n_new = 0
    for f in files:
        key = os.path.relpath(f, samples_root)
        if key in done:
            continue
        z = np.load(f)
        log = json.loads(str(z["log"]))
        method, N = key.split(os.sep)[0], int(key.split(os.sep)[1])
        pid = log["prompt_id"]
        pi = prompts.index(pid)
        beats = prompts.beat_frames[pi]
        motion = torch.from_numpy(z["motion"])[None].to(device)
        J = to_joints(motion, smpl, scale, mn, clip_root=False).cpu().numpy()[0]
        travel, disp = root_covariates(motion, scale, mn)
        full = ba_full(J, beats)
        add, r_k = ba_additive(J, beats, K)
        rep_terms = [0.0, 0.0] + [float(np.exp(-chunk_distance(J[None], k)[0] / tau)) for k in (3, 4)]
        done[key] = {"method": method, "N": N, "prompt_id": pid, "seed": int(log["seed"]), "nfe": int(z["nfe"]),
                     "rep_full": float(sum(rep_terms)), "rep_additive": float(sum(rep_terms)), "rep_terms": rep_terms,
                     "reward_evals": int(log["reward_evals"]), "twist_evals": int(log.get("twist_evals", 0)),
                     "wall": float(z["wall"]), "ba_full": full, "ba_additive": add, "r_k": r_k,
                     "pfc": pfc_scaled(J), "realism_w1": realism_w1(accel_magnitude(J), gt_pool),
                     "travel_m": float(travel[0]), "mean_disp_mm": disp[0].tolist(),
                     "final_reward": log.get("final_reward"), "weighted_mean_reward": log.get("weighted_mean_reward"),
                     "ess_content": log.get("ess_content", []), "ess_noise": log.get("ess_noise", []),
                     "root_oob_frac": float((motion[0, :, 4:7].abs() > 1).any(-1).float().mean())}
        n_new += 1
        if n_new % 500 == 0:
            json.dump(done, open(cache, "w"))
            print(f"metrics: {n_new} new, {len(done)} total, {time.time() - t0:.0f}s", flush=True)
    json.dump(done, open(cache, "w"))
    print(f"metrics: {n_new} new, {len(done)} total, {time.time() - t0:.0f}s", flush=True)
    return done, gt_joints, prompts


def gt_reference(gt_joints, prompts):
    K = prompts.K
    scale, mn = load_normalizer()
    travel, disp = root_covariates(prompts.gt_motion, scale, mn)
    gt_pool = np.concatenate([accel_magnitude(j) for j in gt_joints])
    tau = load_tau() if os.path.exists(TAU_PATH) else float("nan")
    return {"ba_full": float(np.mean([ba_full(gt_joints[i], prompts.beat_frames[i]) for i in range(len(prompts))])),
            "rep_full": float(np.mean([sum(np.exp(-chunk_distance(gt_joints[i:i + 1], k)[0] / tau) for k in (3, 4)) for i in range(len(prompts))])),
            "pfc": float(np.mean([pfc_scaled(j) for j in gt_joints])),
            "realism_w1_leave_one_out": float(np.mean([realism_w1(accel_magnitude(gt_joints[i]), gt_pool) for i in range(len(prompts))])),
            "travel_m": float(travel.mean()), "mean_disp_mm": disp.mean(0).tolist()}


# ----------------------------------------------------------------------------- statistics
def per_prompt_means(rows, key):
    """rows -> {prompt_id: mean over seeds of rows[key]}"""
    by = {}
    for r in rows:
        v = r[key]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        by.setdefault(r["prompt_id"], []).append(v)
    return {p: float(np.mean(v)) for p, v in by.items()}


def bootstrap_mean(values_by_prompt, rng, n_boot=N_BOOT):
    """values_by_prompt: array [n_prompts]. Mean and percentile interval of the mean over prompts."""
    v = np.asarray(values_by_prompt, float)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def select(metrics, method, N):
    return [r for r in metrics.values() if r["method"] == method and r["N"] == N]


def rows_for(metrics, method, N):
    """base is the N = 1 method: at every budget its row is its N = 1 sample set."""
    return select(metrics, "base", 1) if method == "base" else select(metrics, method, N)


def curve_table(metrics, rng):
    out = []
    for m in METHODS:
        for N in NS:
            if m == "base" and N != 1:
                continue
            rows = rows_for(metrics, m, N)
            if not rows:
                continue
            pp = per_prompt_means(rows, REWARD["key"])
            mean, lo, hi = bootstrap_mean(list(pp.values()), rng)
            nfe = sorted({r["nfe"] for r in rows})
            out.append({"method": m, "N": N if m != "base" else 1, "nfe_per_sequence": nfe, "nfe_expected": 400 * N if m != "base" else 400,
                        "n_sequences": len(rows), "n_prompts": len(pp), "ba_full_mean": mean, "ci_lo": lo, "ci_hi": hi,
                        "ba_additive_mean": float(np.mean([r[REWARD["add"]] for r in rows])),
                        "abs_gap_add_full_mean": float(np.mean([abs(r[REWARD["add"]] - r[REWARD["key"]]) for r in rows])),
                        "method_final_reward_mean": float(np.nanmean([r["final_reward"] if r["final_reward"] is not None else np.nan for r in rows])),
                        "weighted_mean_reward_mean": float(np.nanmean([r["weighted_mean_reward"] if r["weighted_mean_reward"] is not None else np.nan for r in rows])),
                        "reward_evals_mean": float(np.mean([r["reward_evals"] for r in rows])),
                        "twist_evals_mean": float(np.mean([r["twist_evals"] for r in rows])),
                        "wall_mean_s": float(np.mean([r["wall"] for r in rows]))})
    return out


def pairwise_table(metrics, rng, N):
    out = []
    pp = {m: per_prompt_means(rows_for(metrics, m, N), REWARD["key"]) for m in METHODS}
    for a, b in itertools.combinations(METHODS, 2):
        common = sorted(set(pp[a]) & set(pp[b]))
        if not common:
            continue
        d = np.array([pp[a][p] - pp[b][p] for p in common])
        mean, lo, hi = bootstrap_mean(d, rng)
        out.append({"N": N, "A": a, "B": b, "n_prompts": len(common), "diff_mean": mean, "ci_lo": lo, "ci_hi": hi,
                    "separates": bool(lo > 0 or hi < 0), "primary": "lattice_smc" in (a, b)})
    return out


def heldout_table(metrics, rng, N):
    out = []
    for m in METHODS:
        rows = rows_for(metrics, m, N)
        if not rows:
            continue
        row = {"method": m, "N": N if m != "base" else 1, "n_sequences": len(rows)}
        for key, name in [("pfc", "pfc"), ("realism_w1", "realism_w1"), ("travel_m", "travel_m"), ("ba_full", "ba_full"), ("rep_full", "rep_full")]:
            if key not in rows[0]:
                continue
            mean, lo, hi = bootstrap_mean(list(per_prompt_means(rows, key).values()), rng)
            row[name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        row["R_PFC"] = -row["pfc"]["mean"]
        row["mean_disp_mm"] = np.mean([r["mean_disp_mm"] for r in rows], 0).tolist()
        row["root_oob_frac"] = float(np.mean([r["root_oob_frac"] for r in rows]))
        out.append(row)
    return out


def diversity_table(samples_root, metrics, device, Ns=(1, 8, 32)):
    """Mean pairwise joint-position distance between the 4 seeds of a prompt (mean over frames
    and joints of the L2 distance), absolute and root-relative, per method and N."""
    smpl = SMPLSkeleton(device)
    scale, mn = load_normalizer()
    out = []
    for m in METHODS:
        for N in Ns:
            if m == "base" and N != 1:
                continue
            rows = rows_for(metrics, m, N)
            by = {}
            for r in rows:
                by.setdefault(r["prompt_id"], []).append(r)
            d_abs, d_rel = [], []
            for pid, rs in by.items():
                if len(rs) < 2:
                    continue
                mot = torch.from_numpy(np.stack([np.load(os.path.join(samples_root, r["method"], str(r["N"]), f"{pid}_{r['seed']}.npz"))["motion"] for r in rs])).to(device)
                J = to_joints(mot, smpl, scale, mn, clip_root=False)
                Jr = J - J[:, :, :1]
                pa, pr = [], []
                for i, j in itertools.combinations(range(len(rs)), 2):
                    pa.append((J[i] - J[j]).norm(dim=-1).mean().item())
                    pr.append((Jr[i] - Jr[j]).norm(dim=-1).mean().item())
                d_abs.append(np.mean(pa))
                d_rel.append(np.mean(pr))
            out.append({"method": m, "N": N, "n_prompts": len(d_abs), "pairwise_joint_dist_abs_m": float(np.mean(d_abs)),
                        "pairwise_joint_dist_root_relative_m": float(np.mean(d_rel))})
    return out


def ess_tables(metrics):
    content, noise = [], []
    for m in ["lattice_smc", "lattice_smc_notwist", "greedy_chunk"]:
        for N in NS:
            rows = select(metrics, m, N)
            if not rows:
                continue
            for k in range(1, 4):
                ev = [e for r in rows for e in r["ess_content"] if e["chunk"] == k and not e.get("final_draw")]
                if ev:
                    content.append({"method": m, "N": N, "chunk": k, "mean_ess": float(np.mean([e["ess"] for e in ev])),
                                    "mean_ess_over_N": float(np.mean([e["ess"] for e in ev]) / N),
                                    "frac_resampled": float(np.mean([e["resampled"] for e in ev])), "n_events": len(ev)})
            fin = [e for r in rows for e in r["ess_content"] if e.get("final_draw")]
            if fin:
                content.append({"method": m, "N": N, "chunk": 4, "mean_ess": float(np.mean([e["ess"] for e in fin])),
                                "mean_ess_over_N": float(np.mean([e["ess"] for e in fin]) / N), "frac_resampled": 0.0,
                                "n_events": len(fin), "note": "before the final draw"})
    for N in NS:
        rows = select(metrics, "fk_noise", N)
        if not rows:
            continue
        for k in range(1, 5):
            for step in (10, 20, 30, 40, 50):
                ev = [e for r in rows for e in r["ess_noise"] if e["chunk"] == k and e["step"] == step]
                if ev:
                    noise.append({"method": "fk_noise", "N": N, "chunk": k, "step": step, "mean_ess": float(np.mean([e["ess"] for e in ev])),
                                  "mean_ess_over_N": float(np.mean([e["ess"] for e in ev]) / N),
                                  "frac_resampled": float(np.mean([e["resampled"] for e in ev])), "n_events": len(ev)})
    return content, noise


def gap_table(metrics):
    out = []
    for m in METHODS:
        rows = [r for r in metrics.values() if r["method"] == m]
        if rows:
            out.append({"method": m, "n": len(rows), "abs_gap_mean": float(np.mean([abs(r[REWARD["add"]] - r[REWARD["key"]]) for r in rows])),
                        "abs_gap_max": float(np.max([abs(r[REWARD["add"]] - r[REWARD["key"]]) for r in rows])),
                        "signed_full_minus_additive_mean": float(np.mean([r[REWARD["key"]] - r[REWARD["add"]] for r in rows]))})
    return out


def spread_table(samples_root):
    """Particle reward range (max - min of the final cumulative reward over the N particles) and
    the implied maximum weight ratio exp(range / alpha), from the method logs."""
    out = []
    for m in ["bon_is", "bon_argmax", "fk_noise", "greedy_chunk", "lattice_smc", "lattice_smc_notwist"]:
        for N in NS:
            files = sorted(glob.glob(os.path.join(samples_root, m, str(N), "*.npz")))
            if not files or N == 1:
                continue
            rng_, sd_, alpha = [], [], None
            for f in files:
                log = json.loads(str(np.load(f)["log"]))
                R = np.array(log["rewards"], float)
                rng_.append(R.max() - R.min())
                sd_.append(R.std())
                alpha = log.get("alpha")
            out.append({"method": m, "N": N, "n": len(files), "mean_range": float(np.mean(rng_)), "mean_sd": float(np.mean(sd_)),
                        "alpha": alpha, "max_weight_ratio_exp_range_over_alpha": float(np.exp(np.mean(rng_) / alpha)) if alpha else None})
    return out


def nfe_audit(metrics):
    bad = [k for k, r in metrics.items() if r["nfe"] != 400 * r["N"]]
    return {"n_sequences": len(metrics), "n_with_wrong_nfe": len(bad), "examples": bad[:5],
            "counts": {m: {str(N): len(select(metrics, m, N)) for N in NS} for m in METHODS}}


def conditions(pairs_all, calib):
    sep = [p for p in pairs_all if p["separates"]]
    prim = [p for p in sep if p["primary"]]
    cond1 = {"met": len(sep) > 0, "statement": "at least one pair of methods separates beyond the bootstrap intervals at some budget",
             "n_separating_pairs": len(sep), "n_pairs_tested": len(pairs_all),
             "separating": [{"N": p["N"], "A": p["A"], "B": p["B"], "diff": p["diff_mean"], "ci": [p["ci_lo"], p["ci_hi"]]} for p in sep],
             "primary_separating": [{"N": p["N"], "A": p["A"], "B": p["B"], "diff": p["diff_mean"], "ci": [p["ci_lo"], p["ci_hi"]]} for p in prim]}
    cond2 = {"met": None, "statement": "the learned content-axis twist is better calibrated than the plug-in noise-axis twist on held-out prefixes"}
    if calib:
        learned = calib["learned_twist_vs_future_target"]
        plug_future = {s: v["vs_future_target"] for s, v in calib["plugin"].items()}
        plug_next = {s: v["vs_next_chunk_reward_same_continuation"] for s, v in calib["plugin"].items()}
        cond2.update({"learned_r2_vs_future": learned["r2"], "learned_rmse_vs_future": learned["rmse"],
                      "plugin_r2_vs_future_by_step": {s: v["r2"] for s, v in plug_future.items()},
                      "plugin_rmse_vs_future_by_step": {s: v["rmse"] for s, v in plug_future.items()},
                      "plugin_r2_vs_next_chunk_by_step": {s: v["r2"] for s, v in plug_next.items()},
                      "met": bool(all(learned["r2"] > v["r2"] and learned["rmse"] < v["rmse"] for v in plug_future.values())),
                      "criterion": "learned R^2 higher and RMSE lower than the plug-in at every step, both against the realized full-future log-mean-exp target on the same 500 fresh prefixes"})
    return cond1, cond2


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = fmt.get(c, "{:.4f}").format(v)
            elif isinstance(v, list):
                v = ", ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in v)
            cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def make_figure(curve, out_path):
    """Reward vs NFE, one line per method with bootstrap bands (log-x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = {"base": "#6b6b6b", "bon_argmax": "#4e79a7", "bon_is": "#76b7b2", "fk_noise": "#e15759",
               "greedy_chunk": "#f28e2b", "lattice_smc": "#59a14f", "lattice_smc_notwist": "#b6992d"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in METHODS:
        rows = [r for r in curve if r["method"] == m]
        if not rows:
            continue
        x = [r["nfe_expected"] for r in rows]
        y = [r["ba_full_mean"] for r in rows]
        lo = [r["ci_lo"] for r in rows]
        hi = [r["ci_hi"] for r in rows]
        if m == "base":
            ax.axhline(y[0], color=palette[m], lw=1.2, ls="--", label="base (N = 1)")
            ax.axhspan(lo[0], hi[0], color=palette[m], alpha=0.10, lw=0)
        else:
            ax.plot(x, y, marker="o", ms=4, lw=1.6, color=palette[m], label=m)
            ax.fill_between(x, lo, hi, color=palette[m], alpha=0.15, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([400 * n for n in NS])
    ax.set_xticklabels([f"{400 * n}\n(N={n})" for n in NS], fontsize=8)
    ax.set_xlabel("NFE per sequence (denoiser forwards, guidance counted twice)")
    ax.set_ylabel("mean R_BA over 40 prompts x 4 seeds")
    ax.set_title("Reward vs NFE, 1000-resample bootstrap intervals over prompts")
    ax.grid(True, which="major", axis="y", color="#dddddd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(ROOT, "samples_p2"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--metrics_only", action="store_true")
    ap.add_argument("--tag", default="phase2", help="output prefix under results/ (phase2, phase2b_a002, ...)")
    ap.add_argument("--calibration", default="twist_calibration", help="results/<name>.json for condition (ii)")
    ap.add_argument("--reward", default="ba", choices=["ba", "rep"], help="steering reward of the grid (curve / pairs / gap); the other is held out")
    args = ap.parse_args()
    if args.reward == "rep":
        REWARD["key"], REWARD["add"] = "rep_full", "rep_additive"
    dev = torch.device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag
    metrics, gt_joints, prompts = compute_metrics(args.samples, dev, tag)
    if args.metrics_only:
        return
    t0 = time.time()
    rng = np.random.default_rng(BOOT_SEED)
    curve = curve_table(metrics, rng)
    pairs = pairwise_table(metrics, rng, 8) + pairwise_table(metrics, rng, 32)
    pairs_all_N = [p for N in NS for p in pairwise_table(metrics, rng, N)]
    held = {str(N): heldout_table(metrics, rng, N) for N in (8, 32)}
    div = diversity_table(args.samples, metrics, dev)
    ess_c, ess_n = ess_tables(metrics)
    gap = gap_table(metrics)
    audit = nfe_audit(metrics)
    calib_path = os.path.join(RESULTS_DIR, args.calibration + ".json")
    calib = json.load(open(calib_path)) if os.path.exists(calib_path) else None
    cond1, cond2 = conditions(pairs_all_N, calib)
    gt = gt_reference(gt_joints, prompts)
    out = {"nfe_audit": audit, "gt_reference": gt, "curve": curve, "pairwise_N8_N32": pairs, "pairwise_all_N": pairs_all_N,
           "heldout": held, "diversity": div, "ess_content": ess_c, "ess_noise": ess_n, "gap": gap,
           "condition_1_separation": cond1, "condition_2_calibration": cond2, "seconds": round(time.time() - t0, 1)}
    out["particle_reward_spread"] = spread_table(args.samples)
    json.dump(out, open(os.path.join(RESULTS_DIR, f"{tag}_analysis.json"), "w"), indent=1)
    try:
        make_figure(curve, os.path.join(RESULTS_DIR, f"fig_reward_vs_nfe_{tag}.png" if tag != "phase2" else "fig_reward_vs_nfe.png"))
    except Exception as e:
        out["figure_error"] = str(e)
    md = f"# {tag} tables (generated by analyze_phase2.py)\n\n"
    md += f"NFE audit: {audit['n_sequences']} sequences, {audit['n_with_wrong_nfe']} with NFE != 400 N.\n\n"
    md += f"Ground truth (40 prompts): BA {gt['ba_full']:.4f}, PFC {gt['pfc']:.3f}, realism W1 (leave-in) {gt['realism_w1_leave_one_out']:.5f}, travel {gt['travel_m']:.2f} m, mean disp (mm/frame) {', '.join(f'{v:.2f}' for v in gt['mean_disp_mm'])}.\n\n"
    md += "## Reward vs NFE\n\n" + md_table(curve, ["method", "N", "nfe_expected", "n_sequences", "ba_full_mean", "ci_lo", "ci_hi", "ba_additive_mean", "abs_gap_add_full_mean", "reward_evals_mean", "twist_evals_mean", "wall_mean_s"], {"wall_mean_s": "{:.2f}", "reward_evals_mean": "{:.1f}", "twist_evals_mean": "{:.1f}"})
    for N in (8, 32):
        md += f"\n## Pairwise differences at N = {N} (A - B, mean R_BA, bootstrap over prompts)\n\n" + md_table([p for p in pairs if p["N"] == N], ["A", "B", "n_prompts", "diff_mean", "ci_lo", "ci_hi", "separates", "primary"])
        md += f"\n## Held-out rewards at N = {N}\n\n"
        rows = [{"method": r["method"], "N": r["N"], "ba_full": r["ba_full"]["mean"], "rep_full": r.get("rep_full", {}).get("mean", float("nan")), "pfc": r["pfc"]["mean"], "pfc_ci": [r["pfc"]["ci_lo"], r["pfc"]["ci_hi"]],
                 "realism_w1": r["realism_w1"]["mean"], "realism_ci": [r["realism_w1"]["ci_lo"], r["realism_w1"]["ci_hi"]], "travel_m": r["travel_m"]["mean"],
                 "mean_disp_mm": r["mean_disp_mm"], "root_oob_frac": r["root_oob_frac"]} for r in held[str(N)]]
        md += md_table(rows, ["method", "N", "ba_full", "rep_full", "pfc", "pfc_ci", "realism_w1", "realism_ci", "travel_m", "mean_disp_mm", "root_oob_frac"], {"realism_w1": "{:.5f}", "pfc": "{:.3f}", "travel_m": "{:.2f}"})
    md += "\n## Diversity across the 4 seeds\n\n" + md_table(div, ["method", "N", "n_prompts", "pairwise_joint_dist_abs_m", "pairwise_joint_dist_root_relative_m"])
    md += "\n## ESS, content axis (before each resampling event / the final draw)\n\n" + md_table(ess_c, ["method", "N", "chunk", "mean_ess", "mean_ess_over_N", "frac_resampled", "n_events"], {"mean_ess": "{:.2f}", "mean_ess_over_N": "{:.3f}", "frac_resampled": "{:.2f}"})
    md += "\n## ESS, noise axis (fk_noise, before each potential)\n\n" + md_table(ess_n, ["N", "chunk", "step", "mean_ess", "mean_ess_over_N", "frac_resampled", "n_events"], {"mean_ess": "{:.2f}", "mean_ess_over_N": "{:.3f}", "frac_resampled": "{:.2f}"})
    md += "\n## Additive vs full R_BA on steered samples (all N)\n\n" + md_table(gap, ["method", "n", "abs_gap_mean", "abs_gap_max", "signed_full_minus_additive_mean"])
    md += "\n## Particle reward range and implied maximum weight ratio (from the method logs)\n\n" + md_table(out["particle_reward_spread"], ["method", "N", "n", "mean_range", "mean_sd", "max_weight_ratio_exp_range_over_alpha", "alpha"], {"max_weight_ratio_exp_range_over_alpha": "{:.3g}", "alpha": "{:g}"})
    md += "\n## Pre-stated conditions (SPEC Section 0)\n\n```\n" + json.dumps({"condition_1": {k: v for k, v in cond1.items() if k != "separating"}, "condition_2": cond2}, indent=1) + "\n```\n"
    open(os.path.join(RESULTS_DIR, f"{tag}_tables.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
