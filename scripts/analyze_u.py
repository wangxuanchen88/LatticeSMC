"""Amendment U reanalysis: argmax-return rows for every proper method in every grid (from the logged final particle
rewards), the pre-registered comparisons, reward-vs-cost curves on three axes (NFE, total model calls, wall) and the
reward-vs-diversity frontier at N = 32. Reads stored logs, the existing metrics caches, the regenerated argmax roots
(dance: samples_*_argmax, metrics tag u_argmax_*; music: results/m3/samples_argmax with its scoring files). Writes
results/u/u_{analysis.json,tables.md} and figures. Nothing under the earlier outputs is modified.

  .venv/bin/python scripts/analyze_u.py
"""
import glob
import sys
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", "."))
import itertools
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lattice_smc.analyze_phase2 import NS, RESULTS_DIR, md_table  # noqa: E402
from lattice_smc.data.clips import load_normalizer  # noqa: E402
from lattice_smc.edge_import import ROOT  # noqa: E402
from lattice_smc.rewards.kinematics import to_joints  # noqa: E402
from vis import SMPLSkeleton  # noqa: E402  (edge_import puts EDGE on sys.path, as analyze_phase2 does)

OUT = os.path.join(RESULTS_DIR, "u")
os.makedirs(OUT, exist_ok=True)
N_BOOT, SEED = 1000, 0
PROPER = ["bon_is", "fk_noise", "lattice_smc", "lattice_smc_notwist"]
ALL = ["base", "bon_argmax", "bon_is", "fk_noise", "greedy_chunk", "lattice_smc", "lattice_smc_notwist"]
# (label, samples root, metrics tag, reward key in the metrics cache, alpha)
GRIDS = [("R_BA alpha 0.2", "samples_p2", "phase2_rerun", "ba", 0.2), ("R_BA alpha 0.05", "samples_p2_a005", "phase2b_a005", "ba", 0.05),
         ("R_BA alpha 0.02", "samples_p2_a002", "phase2b_a002", "ba", 0.02), ("R_rep alpha 0.02", "samples_p3_rep_a002", "phase3_rep_a002", "rep", 0.02),
         ("R_rep alpha 0.05", "samples_p3_rep_a005", "phase3_rep_a005", "rep", 0.05)]
# lattice_smc variants on R_rep alpha 0.02 (root, method, label, Ns)
VARIANTS = [("samples_p3_rep_tw050", "lattice_smc", "lattice_smc tempered 0.5", NS), ("samples_p3_rep_tw025", "lattice_smc", "lattice_smc tempered 0.25", NS),
            ("samples_p3_rep_tw010", "lattice_smc", "lattice_smc tempered 0.1", NS), ("samples_p3_rep_ens_mean", "lattice_smc", "lattice_smc ensemble mean", NS),
            ("samples_p3_rep_ens_shrunk", "lattice_smc", "lattice_smc ensemble shrunk", NS), ("samples_p3_rep_wstwist", "lattice_smc", "lattice_smc within-set twist", NS),
            ("samples_p3_rep_oracle", "lattice_smc", "lattice_smc oracle M=16 (10 prompts)", [8, 32]),
            ("samples_p3_rep_roll_M1", "lattice_smc", "rollout M=1", [4, 8]), ("samples_p3_rep_roll_M2", "lattice_smc", "rollout M=2", [4, 8]),
            ("samples_p3_rep_roll_M4", "lattice_smc", "rollout M=4", [4, 8]), ("samples_p3_rep_match", "lattice_smc_notwist", "notwist at matched N'", [9, 14, 24, 18, 28, 48])]
ARGMAX_ROOTS = {"samples_p2": "u_argmax_p2", "samples_p2_a005": "u_argmax_p2_a005", "samples_p2_a002": "u_argmax_p2_a002",
                "samples_p3_rep_a002": "u_argmax_p3_rep_a002", "samples_p3_rep_a005": "u_argmax_p3_rep_a005"}
M3 = os.path.join(RESULTS_DIR, "m3")
MUSIC_METHODS = ["base", "bon_argmax", "bon_is", "greedy_chunk", "fk_noise", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"]
MUSIC_PROPER = ["bon_is", "fk_noise", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"]
COL = {"base": "#7f7f7f", "bon_argmax": "#1f77b4", "bon_is": "#17becf", "fk_noise": "#2ca02c", "greedy_chunk": "#d62728", "lattice_smc": "#9467bd",
       "lattice_smc_notwist": "#8c564b", "lattice_smc_prefix_b1": "#9467bd", "lattice_smc_prefix_b0.5": "#c5b0d5", "lattice_smc_prefix_b0.25": "#e377c2"}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7.5, "axes.grid": True, "grid.alpha": 0.25, "legend.fontsize": 6, "lines.markersize": 3})


# ------------------------------------------------------------------------------------------ loading
def dance_logs(root, method, N):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, root, method, str(N), "*.npz"))):
        z = np.load(f)
        log = json.loads(str(z["log"]))
        R = np.asarray(log["rewards"], float)
        out[(log["prompt_id"], int(log["seed"]))] = {"draw": float(log["final_reward"]), "argmax": float(R.max()), "chosen": int(log["chosen"]),
                                                    "argmax_idx": int(np.argmax(R)), "nfe": int(z["nfe"]), "reward_evals": int(z["reward_evals"]),
                                                    "twist_evals": int(log.get("twist_evals", 0)), "wall": float(z["wall"]), "file": f}
    return out


def metrics_for(tag):
    p = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def met_rows(metrics, method, N):
    return {(r["prompt_id"], r["seed"]): r for r in metrics.values() if r["method"] == method and r["N"] == N}


def boot(v, rng):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_prompt(d):
    by = {}
    for (p, s), v in d.items():
        by.setdefault(p, []).append(v)
    return {p: float(np.mean(v)) for p, v in by.items()}


def paired(a, b, rng):
    common = [k for k in a if k in b]
    by = {}
    for k in common:
        by.setdefault(k[0], []).append(a[k] - b[k])
    d = [float(np.mean(v)) for v in by.values()]
    if not d:
        return None
    m, lo, hi = boot(d, rng)
    return {"n_prompts": len(d), "n_sequences": len(common), "diff": m, "ci_lo": lo, "ci_hi": hi, "separates": bool(lo > 0 or hi < 0)}


def root_rel_diversity(root, method, N, device):
    """Mean over prompts of the mean pairwise root-relative joint distance between the seeds of a prompt (as analyze_phase2)."""
    smpl = SMPLSkeleton(device)
    scale, mn = load_normalizer()
    files = sorted(glob.glob(os.path.join(ROOT, root, method, str(N), "*.npz")))
    by = {}
    for f in files:
        pid = os.path.basename(f).rsplit("_", 1)[0]
        by.setdefault(pid, []).append(f)
    vals = []
    for pid, fs in by.items():
        if len(fs) < 2:
            continue
        mot = torch.from_numpy(np.stack([np.load(f)["motion"] for f in fs])).to(device)
        J = to_joints(mot, smpl, scale, mn, clip_root=False)
        Jr = J - J[:, :, :1]
        vals.append(np.mean([(Jr[i] - Jr[j]).norm(dim=-1).mean().item() for i, j in itertools.combinations(range(len(fs)), 2)]))
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------------------------------ dance
def dance(rng, device):
    rows, pairs, curves, frontier = [], [], {}, []
    for label, root, tag, rkey, alpha in GRIDS:
        met = metrics_for(tag)
        full_key, add_key = f"{rkey}_full", f"{rkey}_additive"
        am_met = metrics_for(ARGMAX_ROOTS[root])
        curves[label] = []
        for m in ALL:
            for N in NS:
                if m == "base" and N != 1:
                    continue
                mr = met_rows(met, m, N)
                if not mr:
                    continue
                full = {k: r[full_key] for k, r in mr.items()}
                add = {k: r[add_key] for k, r in mr.items()}
                mean_f, lo_f, hi_f = boot(list(per_prompt(full).values()), rng)
                mean_a, lo_a, hi_a = boot(list(per_prompt(add).values()), rng)
                row = {"grid": label, "method": m, "N": N, "rule": "draw" if m in PROPER else ("argmax" if m in ("bon_argmax", "greedy_chunk") else "-"),
                       "n": len(mr), "reward_full": mean_f, "full_lo": lo_f, "full_hi": hi_f, "reward_additive": mean_a, "add_lo": lo_a, "add_hi": hi_a,
                       "nfe": float(np.mean([r["nfe"] for r in mr.values()])), "reward_evals": float(np.mean([r["reward_evals"] for r in mr.values()])),
                       "twist_evals": float(np.mean([r["twist_evals"] for r in mr.values()])), "wall_s": float(np.mean([r["wall"] for r in mr.values()]))}
                row["total_calls"] = row["nfe"] + row["reward_evals"] + row["twist_evals"]
                rows.append(row)
                curves[label].append(row)
                if m in PROPER:
                    lg = dance_logs(root, m, N)
                    am = {k: v["argmax"] for k, v in lg.items()}
                    mean_am, lo_am, hi_am = boot(list(per_prompt(am).values()), rng)
                    arow = {**row, "rule": "argmax", "reward_full": float("nan"), "full_lo": float("nan"), "full_hi": float("nan"),
                            "reward_additive": mean_am, "add_lo": lo_am, "add_hi": hi_am,
                            "frac_argmax_equals_draw": float(np.mean([v["chosen"] == v["argmax_idx"] for v in lg.values()]))}
                    if m == "bon_is":
                        # bon logs the FULL reward per particle and shares its particle set with bon_argmax, so the bon_is-argmax
                        # row is bon_argmax's returned sample exactly: take its full and additive rewards from the metrics cache
                        ba = met_rows(met, "bon_argmax", N)
                        arow["equals_bon_argmax_max_abs_diff"] = float(max(abs(am[k] - ba[k][full_key]) for k in am if k in ba)) if rkey == "ba" else float(max(abs(am[k] - ba[k][full_key]) for k in am if k in ba))
                        arow["reward_full"], arow["full_lo"], arow["full_hi"] = boot(list(per_prompt({k: r[full_key] for k, r in ba.items()}).values()), rng)
                        arow["reward_additive"], arow["add_lo"], arow["add_hi"] = boot(list(per_prompt({k: r[add_key] for k, r in ba.items()}).values()), rng)
                        arow["note"] = "= bon_argmax (same particle set; bon logs full rewards)"
                    # regenerated argmax cells: full reward from the argmax root's metrics
                    amr = met_rows(am_met, m, N)
                    if amr:
                        fa = {k: r[full_key] for k, r in amr.items()}
                        # consistency: the regenerated argmax additive reward must equal the log-derived argmax reward
                        adds = {k: r[add_key] for k, r in amr.items()}
                        arow["regen_max_abs_diff_additive"] = float(max(abs(adds[k] - am[k]) for k in adds if k in am)) if rkey == "rep" else float("nan")
                        arow["reward_full"], arow["full_lo"], arow["full_hi"] = boot(list(per_prompt(fa).values()), rng)
                        arow["regenerated"] = True
                        arow["pfc"] = float(np.mean([r["pfc"] for r in amr.values()]))
                        arow["realism_w1"] = float(np.mean([r["realism_w1"] for r in amr.values()]))
                        arow["diversity_root_rel"] = root_rel_diversity(root + "_argmax", m, N, device)
                    rows.append(arow)
                    curves[label].append(arow)
        # pre-registered: lattice_smc_notwist-argmax vs greedy_chunk at alpha 0.02
        if alpha == 0.02:
            for N in (8, 32):
                g = met_rows(met, "greedy_chunk", N)
                lg = dance_logs(root, "lattice_smc_notwist", N)
                pa = paired({k: v["argmax"] for k, v in lg.items()}, {k: r[add_key] for k, r in g.items()}, rng)
                pairs.append({"grid": label, "A": "lattice_smc_notwist-argmax", "B": "greedy_chunk", "N": N, "scale": "additive (logs)", **pa})
                amr = met_rows(am_met, "lattice_smc_notwist", N)
                if amr:
                    pf = paired({k: r[full_key] for k, r in amr.items()}, {k: r[full_key] for k, r in g.items()}, rng)
                    pairs.append({"grid": label, "A": "lattice_smc_notwist-argmax (regenerated)", "B": "greedy_chunk", "N": N, "scale": "full (regenerated)", **pf})
                # the argmax rule against its own draw rule
                pd = paired({k: v["argmax"] for k, v in lg.items()}, {k: v["draw"] for k, v in lg.items()}, rng)
                pairs.append({"grid": label, "A": "lattice_smc_notwist-argmax", "B": "lattice_smc_notwist (draw)", "N": N, "scale": "additive (logs)", **pd})
        # frontier at N = 32: existing methods (full reward, root-relative diversity from the grid's analysis json) + regenerated argmax
        an = json.load(open(os.path.join(RESULTS_DIR, f"{tag}_analysis.json")))
        div = {(d["method"], d["N"]): d["pairwise_joint_dist_root_relative_m"] for d in an["diversity"]}
        for m in ALL:
            N = 1 if m == "base" else 32
            r = next((x for x in rows if x["grid"] == label and x["method"] == m and x["N"] == N and x["rule"] != "argmax"), None)
            if r and (m, N) in div:
                frontier.append({"grid": label, "alpha": alpha, "reward_name": rkey, "method": m, "rule": r["rule"], "reward_full": r["reward_full"], "diversity": div[(m, N)]})
        r = next((x for x in rows if x["grid"] == label and x["method"] == "lattice_smc_notwist" and x["N"] == 32 and x["rule"] == "argmax" and x.get("regenerated")), None)
        if r:
            frontier.append({"grid": label, "alpha": alpha, "reward_name": rkey, "method": "lattice_smc_notwist", "rule": "argmax", "reward_full": r["reward_full"], "diversity": r["diversity_root_rel"]})
    # variants (R_rep alpha 0.02): argmax rows from logs (additive == full for R_rep)
    vrows = []
    for root, m, label, Ns in VARIANTS:
        for N in Ns:
            lg = dance_logs(root, m, N)
            if not lg:
                continue
            md_, lo_d, hi_d = boot(list(per_prompt({k: v["draw"] for k, v in lg.items()}).values()), rng)
            ma, lo_a, hi_a = boot(list(per_prompt({k: v["argmax"] for k, v in lg.items()}).values()), rng)
            pd = paired({k: v["argmax"] for k, v in lg.items()}, {k: v["draw"] for k, v in lg.items()}, rng)
            vrows.append({"variant": label, "N": N, "n": len(lg), "nfe": float(np.mean([v["nfe"] for v in lg.values()])), "draw": md_, "draw_lo": lo_d, "draw_hi": hi_d,
                          "argmax": ma, "argmax_lo": lo_a, "argmax_hi": hi_a, "argmax_minus_draw": pd["diff"], "amd_lo": pd["ci_lo"], "amd_hi": pd["ci_hi"], "separates": pd["separates"],
                          "frac_argmax_equals_draw": float(np.mean([v["chosen"] == v["argmax_idx"] for v in lg.values()]))})
    return rows, pairs, curves, frontier, vrows


# ------------------------------------------------------------------------------------------ music
def music_logs(root, method, N):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, method, str(N), "*", "log.json"))):
        log = json.load(open(f))
        R = np.asarray(log["rewards"], float)
        d = os.path.dirname(f)
        emb = np.load(os.path.join(d, "clap_embed.npy")) if os.path.exists(os.path.join(d, "clap_embed.npy")) else None
        out[(log["prompt_id"], int(log["base_seed"]))] = {"draw": float(log["final_reward"]), "argmax": float(R.max()), "chosen": int(log["chosen"]), "argmax_idx": int(np.argmax(R)),
                                                          "nfe": int(log["nfe"]), "decode_calls": int(log["decode_calls"]), "clap_calls": int(log["clap_calls"]), "wall": float(log["wall"]),
                                                          "emb": emb, "key": f"{method}/{N}/{os.path.basename(d)}", "replay": log.get("replay")}
    return out


def seed_div(lg):
    by = {}
    for (p, s), v in lg.items():
        if v["emb"] is not None:
            by.setdefault(p, []).append(v["emb"] / np.linalg.norm(v["emb"]))
    vals = [float(np.mean([1 - float(E[i] @ E[j]) for i, j in itertools.combinations(range(len(E)), 2)])) for E in by.values() if len(E) >= 2]
    return float(np.mean(vals)) if vals else float("nan")


def music(rng):
    root = os.path.join(M3, "samples")
    aroot = os.path.join(M3, "samples_argmax")
    scores = {r["key"]: r for r in json.load(open(os.path.join(M3, "argmax_scores.json")))["rows"]} if os.path.exists(os.path.join(M3, "argmax_scores.json")) else {}
    tempo = {r["key"]: r for r in json.load(open(os.path.join(M3, "argmax_tempo.json")))["rows"]} if os.path.exists(os.path.join(M3, "argmax_tempo.json")) else {}
    aes = {r["key"]: r for r in json.load(open(os.path.join(M3, "argmax_aesthetics.json")))["rows"]} if os.path.exists(os.path.join(M3, "argmax_aesthetics.json")) else {}
    base_scores = {r["key"]: r for r in json.load(open(os.path.join(M3, "scores.json")))["rows"]}
    base_tempo = {r["key"]: r for r in json.load(open(os.path.join(M3, "tempo.json")))["rows"]}
    base_aes = {r["key"]: r for r in json.load(open(os.path.join(M3, "aesthetics.json")))["rows"]}
    rows, pairs, frontier = [], [], []

    def heldout(lg, sc, tp, ae):
        keys = [v["key"] for v in lg.values()]
        h = {}
        for name, src, field in (("frac_energy_above_8k", sc, "frac_energy_above_8k"), ("seam_ratio", sc, "seam_mean_ratio"), ("tempo_reward", tp, "tempo_reward_additive"),
                                 ("aes_CE", ae, "CE"), ("aes_CU", ae, "CU"), ("aes_PC", ae, "PC"), ("aes_PQ", ae, "PQ")):
            vals = [src[k][field] for k in keys if k in src and src[k].get(field) is not None]
            h[name] = float(np.mean(vals)) if vals else float("nan")
            h[name + "_n"] = len(vals)
        return h

    for m in MUSIC_METHODS:
        for N in NS:
            if m == "base" and N != 1:
                continue
            lg = music_logs(root, m, N)
            if not lg:
                continue
            mean, lo, hi = boot(list(per_prompt({k: v["draw"] for k, v in lg.items()}).values()), rng)
            row = {"method": m, "N": N, "rule": "draw" if m in MUSIC_PROPER else ("argmax" if m in ("bon_argmax", "greedy_chunk") else "-"), "n": len(lg),
                   "reward": mean, "lo": lo, "hi": hi, "nfe": float(np.mean([v["nfe"] for v in lg.values()])), "decode_calls": float(np.mean([v["decode_calls"] for v in lg.values()])),
                   "clap_calls": float(np.mean([v["clap_calls"] for v in lg.values()])), "wall_s": float(np.mean([v["wall"] for v in lg.values()])), "diversity": seed_div(lg)}
            row["total_calls"] = row["nfe"] + row["decode_calls"] + row["clap_calls"]
            if N in (1, 8, 32):
                row.update(heldout(lg, base_scores, base_tempo, base_aes))
            rows.append(row)
            if m in MUSIC_PROPER:
                ma, lo_a, hi_a = boot(list(per_prompt({k: v["argmax"] for k, v in lg.items()}).values()), rng)
                arow = {**row, "rule": "argmax", "reward": ma, "lo": lo_a, "hi": hi_a, "diversity": float("nan"),
                        "frac_argmax_equals_draw": float(np.mean([v["chosen"] == v["argmax_idx"] for v in lg.values()]))}
                for k in list(arow):
                    if k.startswith(("frac_energy", "seam", "tempo", "aes")):
                        arow[k] = float("nan") if not k.endswith("_n") else 0
                alg = music_logs(aroot, m, N)
                if alg:
                    arow["regenerated"] = True
                    arow["n_regenerated"] = len(alg)
                    arow["replay_bitwise_all"] = bool(all(v["replay"]["chosen_latent_bitwise"] for v in alg.values()))
                    arow["replay_max_abs_diff_reward"] = float(max(v["replay"]["abs_diff"] for v in alg.values()))
                    arow["diversity"] = seed_div(alg)
                    arow.update(heldout(alg, scores, tempo, aes))
                    arow["reward_from_regen"] = float(np.mean([v["draw"] for v in alg.values()]))  # the regenerated root's final_reward is the argmax reward
                rows.append(arow)
                if m == "bon_is":
                    ba = music_logs(root, "bon_argmax", N)
                    arow["argmax_equals_bon_argmax_max_abs_diff"] = float(max(abs(lg[k]["argmax"] - ba[k]["draw"]) for k in lg if k in ba))
    for N in (8, 32):
        lg = music_logs(root, "lattice_smc_prefix_b1", N)
        g = music_logs(root, "greedy_chunk", N)
        pairs.append({"A": "lattice_smc_prefix_b1-argmax", "B": "greedy_chunk", "N": N, **paired({k: v["argmax"] for k, v in lg.items()}, {k: v["draw"] for k, v in g.items()}, rng)})
        pairs.append({"A": "lattice_smc_prefix_b1-argmax", "B": "lattice_smc_prefix_b1 (draw)", "N": N, **paired({k: v["argmax"] for k, v in lg.items()}, {k: v["draw"] for k, v in lg.items()}, rng)})
        fk = music_logs(root, "fk_noise", N)
        pairs.append({"A": "fk_noise-argmax", "B": "greedy_chunk", "N": N, **paired({k: v["argmax"] for k, v in fk.items()}, {k: v["draw"] for k, v in g.items()}, rng)})
    for r in rows:
        if (r["N"] == 32 or r["method"] == "base") and np.isfinite(r["diversity"]):
            frontier.append({"grid": "music terminal CLAP", "alpha": 0.006762, "reward_name": "clap", "method": r["method"], "rule": r["rule"], "reward_full": r["reward"], "diversity": r["diversity"]})
    return rows, pairs, frontier


# ------------------------------------------------------------------------------------------ figures
def cost_figure(name, series, xlabels, ylabel, title):
    """series: list of (label, color, linestyle, [(nfe, calls, wall, reward, lo, hi)])"""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, xi, xl in zip(axes, (0, 1, 2), xlabels):
        for label, col, ls, pts in series:
            pts = sorted(pts)
            ax.errorbar([p[xi] for p in pts], [p[3] for p in pts], yerr=[[p[3] - p[4] for p in pts], [p[5] - p[3] for p in pts]], color=col, ls=ls, marker="o" if ls == "-" else "s",
                        mfc="white" if ls != "-" else None, capsize=1.2, elinewidth=0.6, lw=1.1, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
    axes[0].set_ylabel(ylabel)
    axes[1].set_title(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06 * (-(-len(l) // 4)) + 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=130)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    drows, dpairs, curves, dfront, vrows = dance(rng, device)
    mrows, mpairs, mfront = music(rng)
    # cost-axis figures, one per dance grid (additive scale from logs / metrics for every row) and one for music
    for label, root, tag, rkey, alpha in GRIDS:
        series = []
        for m in ALL:
            for rule, ls in (("draw" if m in PROPER else ("argmax" if m in ("bon_argmax", "greedy_chunk") else "-"), "-"), ("argmax", "--")):
                if rule == "argmax" and m not in PROPER:
                    continue
                pts = [(r["nfe"], r["total_calls"], r["wall_s"], r["reward_additive"], r["add_lo"], r["add_hi"]) for r in curves[label] if r["method"] == m and r["rule"] == rule]
                if pts:
                    series.append((m + (" (argmax)" if rule == "argmax" and m in PROPER else ""), COL[m], ls, pts))
        cost_figure("fig_u_cost_" + tag, series, ("NFE per sequence", "total model calls (NFE + reward + twist evals)", "wall per sequence (s)"),
                    ("R_BA (additive, steering scale)" if rkey == "ba" else "R_rep"), label)
    series = []
    for m in MUSIC_METHODS:
        for rule, ls in (("draw" if m in MUSIC_PROPER else ("argmax" if m in ("bon_argmax", "greedy_chunk") else "-"), "-"), ("argmax", "--")):
            if rule == "argmax" and m not in MUSIC_PROPER:
                continue
            pts = [(r["nfe"], r["total_calls"], r["wall_s"], r["reward"], r["lo"], r["hi"]) for r in mrows if r["method"] == m and r["rule"] == rule]
            if pts:
                series.append((m + (" (argmax)" if rule == "argmax" and m in MUSIC_PROPER else ""), COL[m], ls, pts))
    cost_figure("fig_u_cost_music", series, ("NFE per sequence (transformer rows)", "total model calls (NFE + decoder + CLAP)", "wall per sequence (s, 3 processes per GPU)"), "terminal CLAP", "music, alpha 0.006762")
    # frontier
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, sel, xl, yl in ((axes[0], [f for f in dfront if f["reward_name"] == "ba"], "root-relative seed diversity (m)", "R_BA (full)"),
                            (axes[1], [f for f in dfront if f["reward_name"] == "rep"], "root-relative seed diversity (m)", "R_rep"),
                            (axes[2], mfront, "seed diversity (CLAP cosine distance)", "terminal CLAP")):
        alphas = sorted({f["alpha"] for f in sel})
        mk = dict(zip(alphas, ["o", "^", "s", "D"]))
        for f in sel:
            big = f["method"] in ("greedy_chunk", "bon_argmax") or f["rule"] == "argmax"
            ax.scatter(f["diversity"], f["reward_full"], s=34 if big else 14, marker=mk[f["alpha"]], color=COL[f["method"]], edgecolors="k" if big else "none",
                       linewidths=0.9 if f["rule"] == "argmax" else 0.5, facecolors="none" if f["rule"] == "argmax" else COL[f["method"]])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title("N = 32; markers by alpha: " + ", ".join(f"{mk[a]} {a}" for a in alphas), fontsize=6.5)
    handles = [plt.Line2D([], [], color=COL[m], marker="o", ls="", label=m) for m in ALL] + [plt.Line2D([], [], color="k", marker="o", mfc="none", ls="", label="argmax-return row (hollow, black edge)"),
                                                                                              plt.Line2D([], [], color="k", marker="o", ls="", label="greedy_chunk / bon_argmax (black edge)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_u_frontier.pdf"))
    fig.savefig(os.path.join(OUT, "fig_u_frontier.png"), dpi=130)
    # tables
    fmt = {"nfe": "{:.0f}", "reward_evals": "{:.0f}", "twist_evals": "{:.0f}", "total_calls": "{:.0f}", "wall_s": "{:.2f}", "decode_calls": "{:.0f}", "clap_calls": "{:.0f}",
           "frac_argmax_equals_draw": "{:.2f}", "diversity": "{:.4f}", "diversity_root_rel": "{:.4f}", "frac_energy_above_8k": "{:.5f}", "seam_ratio": "{:.3f}", "tempo_reward": "{:.3f}",
           "aes_CE": "{:.2f}", "aes_CU": "{:.2f}", "aes_PC": "{:.2f}", "aes_PQ": "{:.2f}", "pfc": "{:.3f}", "realism_w1": "{:.5f}", "replay_max_abs_diff_reward": "{:.1e}", "regen_max_abs_diff_additive": "{:.1e}"}
    md = "# Amendment U tables\n\n## Dance grids: weighted-draw rows (full and additive reward) and argmax rows (additive from logs; full where regenerated)\n\n"
    md += md_table(drows, ["grid", "method", "N", "rule", "n", "reward_full", "full_lo", "full_hi", "reward_additive", "add_lo", "add_hi", "frac_argmax_equals_draw", "regenerated", "regen_max_abs_diff_additive", "equals_bon_argmax_max_abs_diff", "pfc", "realism_w1", "diversity_root_rel", "nfe", "total_calls", "wall_s"], fmt)
    md += "\n## R_rep alpha 0.02 lattice_smc variants, oracle and rollout rows: draw vs argmax (exact from logs; R_rep additive == full)\n\n"
    md += md_table(vrows, ["variant", "N", "n", "nfe", "draw", "draw_lo", "draw_hi", "argmax", "argmax_lo", "argmax_hi", "argmax_minus_draw", "amd_lo", "amd_hi", "separates", "frac_argmax_equals_draw"], fmt)
    md += "\n## Dance pre-registered comparisons and argmax-vs-draw (bootstrap over prompts)\n\n" + md_table(dpairs, ["grid", "A", "B", "N", "scale", "n_prompts", "n_sequences", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Music: rows with argmax (from logs; regenerated at N = 8, 32 for lattice_smc_prefix_b1 and fk_noise: held-out and diversity)\n\n"
    md += md_table(mrows, ["method", "N", "rule", "n", "reward", "lo", "hi", "frac_argmax_equals_draw", "regenerated", "replay_bitwise_all", "replay_max_abs_diff_reward", "argmax_equals_bon_argmax_max_abs_diff", "tempo_reward", "tempo_reward_n", "frac_energy_above_8k", "aes_CE", "aes_CU", "aes_PC", "aes_PQ", "seam_ratio", "diversity", "nfe", "total_calls", "wall_s"], fmt)
    md += "\n## Music pre-registered comparisons (bootstrap over prompts)\n\n" + md_table(mpairs, ["A", "B", "N", "n_prompts", "n_sequences", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Frontier points (N = 32)\n\n" + md_table(dfront + mfront, ["grid", "alpha", "method", "rule", "reward_full", "diversity"], {"diversity": "{:.4f}"})
    open(os.path.join(OUT, "u_tables.md"), "w").write(md)
    json.dump({"dance_rows": drows, "dance_pairs": dpairs, "variant_rows": vrows, "music_rows": mrows, "music_pairs": mpairs, "frontier": dfront + mfront},
              open(os.path.join(OUT, "u_analysis.json"), "w"), indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()
