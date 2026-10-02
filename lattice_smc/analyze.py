"""Analysis. Phase 1 part (SPEC 2.5): determinism across sample roots, boundary continuity, BA and
PFC against ground truth, additive-vs-full R_BA gap, NFE and wall-time summary.

  python -m lattice_smc.analyze --phase1 --samples samples --compare samples_rerun,samples_gpu1

Held-out rewards are read here only (project rule 6). Writes results/phase1_checks.json and
results/phase1_checks.md. Phase 2 curves, pairwise differences, hacking check, ESS traces and
calibration tables are added in Phase 2.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from lattice_smc.data.clips import load_normalizer
from lattice_smc.data.pairs import CHUNK_LEN, PromptSet
from lattice_smc.edge_import import EDGE_DIR, ROOT  # noqa: F401
from lattice_smc.rewards.ba import ba_additive, ba_full
from lattice_smc.rewards.kinematics import frame_jumps, to_joints
from lattice_smc.rewards.pfc import pfc_scaled
from vis import SMPLSkeleton  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
BOUNDARY_RATIO_LIMIT = 3.0


def load_samples(root, method, N):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, method, str(N), "*.npz"))):
        z = np.load(p)
        name = os.path.splitext(os.path.basename(p))[0]
        out[name] = {"motion": z["motion"], "nfe": int(z["nfe"]), "wall": float(z["wall"]), "log": json.loads(str(z["log"]))}
    return out


def boundary_stats(joints, K):
    """(boundary jumps [K-1], within-chunk jumps) for one [K*150, 24, 3] sequence."""
    j = frame_jumps(joints)  # j[t] = jump between t and t+1
    seams = np.array([j[k * CHUNK_LEN - 1] for k in range(1, K)])
    within_mask = np.ones(len(j), dtype=bool)
    within_mask[[k * CHUNK_LEN - 1 for k in range(1, K)]] = False
    return seams, j[within_mask]


def seam_components(joints_all, K):
    """Root-translation and root-relative-pose components of the seam statistic, pooled over
    sequences: (root seam mean, root within median, pose seam mean, pose within median)."""
    root = joints_all[:, :, 0]
    rel = joints_all - root[:, :, None]
    rj = np.linalg.norm(root[:, 1:] - root[:, :-1], axis=-1)
    pj = np.linalg.norm(rel[:, 1:] - rel[:, :-1], axis=-1).mean(-1)
    seam_idx = [k * CHUNK_LEN - 1 for k in range(1, K)]
    within_mask = np.ones(rj.shape[1], dtype=bool)
    within_mask[seam_idx] = False
    return (float(rj[:, seam_idx].mean()), float(np.median(rj[:, within_mask])),
            float(pj[:, seam_idx].mean()), float(np.median(pj[:, within_mask])))


def jump_profile(joints_all, K, lo=-3, hi=3):
    """Mean frame-to-frame jump at seam offsets lo..hi: key 't->t+1' for t = 149 + offset."""
    j = np.stack([frame_jumps(x) for x in joints_all])
    out = {}
    for off in range(lo, hi + 1):
        idx = [k * CHUNK_LEN - 1 + off for k in range(1, K)]
        out[f"{CHUNK_LEN - 1 + off}->{CHUNK_LEN + off}"] = float(j[:, idx].mean())
    return out


def determinism(samples, others):
    rows = []
    for name, root in others.items():
        n_eq = n_cmp = 0
        max_abs = 0.0
        for key, s in samples.items():
            if key not in root:
                continue
            n_cmp += 1
            d = np.abs(s["motion"] - root[key]["motion"]).max()
            max_abs = max(max_abs, float(d))
            n_eq += int(np.array_equal(s["motion"], root[key]["motion"]))
        rows.append({"root": name, "n_compared": n_cmp, "n_bitwise_equal": n_eq, "max_abs_diff": max_abs,
                     "physical_gpus_this_root": sorted({r["log"].get("cuda_visible_devices", "?") for r in root.values()}),
                     "physical_gpus_reference_root": sorted({r["log"].get("cuda_visible_devices", "?") for r in samples.values()}),
                     "n_pairs_on_different_gpus": sum(1 for k, s in samples.items() if k in root and
                                                      s["log"].get("cuda_visible_devices") != root[k]["log"].get("cuda_visible_devices"))})
    return rows


def phase1(args):
    t0 = time.time()
    dev = torch.device(args.device)
    prompts = PromptSet()
    K = prompts.K
    smpl = SMPLSkeleton(dev)
    scale, mn = load_normalizer()
    samples = load_samples(args.samples, "base", 1)
    assert samples, f"no samples under {args.samples}/base/1"
    others = {c: load_samples(c, "base", 1) for c in args.compare.split(",") if c}

    gt_joints = to_joints(prompts.gt_motion.to(dev), smpl, scale, mn).cpu().numpy()
    names = sorted(samples)
    motions = torch.from_numpy(np.stack([samples[n]["motion"] for n in names])).to(dev)
    joints = to_joints(motions, smpl, scale, mn).cpu().numpy()
    # frames whose root leaves the normalizer's [-1, 1] range are clipped by unnormalize before FK
    # (matters for integrated displacements, Amendment E); ground truth is in range by construction
    root_oob = (motions[..., 4:7].abs() > 1).any(-1)
    root_out_of_range = {"n_frames": int(root_oob.sum()), "frac_frames": float(root_oob.float().mean()),
                         "n_sequences_with_any": int(root_oob.any(-1).sum()),
                         "max_abs_root": float(motions[..., 4:7].abs().max())}

    per_sample = []
    seams_all, within_all, seams_gt, within_gt = [], [], [], []
    for n, jn in zip(names, joints):
        pid, seed = n.rsplit("_", 1)
        pi = prompts.index(pid)
        beats = prompts.beat_frames[pi]
        seams, within = boundary_stats(jn, K)
        seams_all.append(seams)
        within_all.append(within)
        full = ba_full(jn, beats)
        add, r_k = ba_additive(jn, beats, K)
        per_sample.append({"sample": n, "prompt_id": pid, "seed": int(seed), "nfe": samples[n]["nfe"],
                           "wall": samples[n]["wall"], "ba_full": full, "ba_additive": add, "r_k": r_k,
                           "pfc": pfc_scaled(jn), "seam_jumps": seams.tolist(), "within_median": float(np.median(within)),
                           "ratio": float(seams.mean() / np.median(within))})
    gt_rows = []
    for pi, p in enumerate(prompts.prompts):
        seams, within = boundary_stats(gt_joints[pi], K)
        seams_gt.append(seams)
        within_gt.append(within)
        beats = prompts.beat_frames[pi]
        add, _ = ba_additive(gt_joints[pi], beats, K)
        gt_rows.append({"prompt_id": p["prompt_id"], "ba_full": ba_full(gt_joints[pi], beats), "ba_additive": add,
                        "pfc": pfc_scaled(gt_joints[pi]), "ratio": float(seams.mean() / np.median(within))})

    seams_all, within_all = np.concatenate(seams_all), np.concatenate(within_all)
    seams_gt, within_gt = np.concatenate(seams_gt), np.concatenate(within_gt)
    r_seam, r_within, p_seam, p_within = seam_components(joints, K)
    gr_seam, gr_within, gp_seam, gp_within = seam_components(gt_joints, K)
    ba_f = np.array([r["ba_full"] for r in per_sample], dtype=float)
    ba_a = np.array([r["ba_additive"] for r in per_sample], dtype=float)
    pf = np.array([r["pfc"] for r in per_sample])
    nfe = np.array([r["nfe"] for r in per_sample])
    wall = np.array([r["wall"] for r in per_sample])
    by_prompt = {}
    for r in per_sample:
        by_prompt.setdefault(r["prompt_id"], []).append(r["ba_full"])
    prompt_means = np.array([np.mean(v) for v in by_prompt.values()])

    out = {
        "K": K, "n_prompts": len(prompts), "n_samples": len(per_sample), "samples_root": args.samples,
        "seeds": sorted({r["seed"] for r in per_sample}),
        "ckpt": sorted({r["log"].get("ckpt", "?") for r in samples.values()}),
        "ckpt_step": sorted({int(r["log"].get("ckpt_step", -1)) for r in samples.values()}),
        "loss_cfg": [dict(t) for t in {tuple(sorted(r["log"].get("loss_cfg", {}).items())) for r in samples.values()}],
        "root_out_of_range": root_out_of_range,
        "nfe": {"expected_per_sequence": K * 50 * 2, "min": int(nfe.min()), "max": int(nfe.max()),
                "all_equal_expected": bool(np.all(nfe == K * 50 * 2)),
                "reward_evals_total": int(sum(s["log"]["reward_evals"] for s in samples.values()))},
        "wall": {"mean_per_sequence_s": float(wall.mean()), "mean_per_chunk_s": float(wall.mean() / K),
                 "total_s": float(wall.sum())},
        "determinism": determinism(samples, others),
        "boundary": {
            "ratio_pooled_mean_seam_over_median_within": float(seams_all.mean() / np.median(within_all)),
            "ratio_pooled_median_seam_over_median_within": float(np.median(seams_all) / np.median(within_all)),
            "ratio_per_sample_mean": float(np.mean([r["ratio"] for r in per_sample])),
            "ratio_per_sample_max": float(np.max([r["ratio"] for r in per_sample])),
            "seam_jump_mean": float(seams_all.mean()), "within_jump_median": float(np.median(within_all)),
            "within_jump_mean": float(within_all.mean()),
            "gt_ratio_pooled_mean_seam_over_median_within": float(seams_gt.mean() / np.median(within_gt)),
            "gt_seam_jump_mean": float(seams_gt.mean()), "gt_within_jump_median": float(np.median(within_gt)),
            "root_translation": {"seam_mean": r_seam, "within_median": r_within, "ratio": r_seam / r_within,
                                 "gt_seam_mean": gr_seam, "gt_within_median": gr_within, "gt_ratio": gr_seam / gr_within},
            "root_relative_pose": {"seam_mean": p_seam, "within_median": p_within, "ratio": p_seam / p_within,
                                   "gt_seam_mean": gp_seam, "gt_within_median": gp_within, "gt_ratio": gp_seam / gp_within},
            "jump_profile_146_152": jump_profile(joints, K), "gt_jump_profile_146_152": jump_profile(gt_joints, K),
            "per_boundary_seam_mean": [float(np.mean([r["seam_jumps"][k] for r in per_sample])) for k in range(K - 1)],
            "limit": BOUNDARY_RATIO_LIMIT,
            "pass": bool(seams_all.mean() / np.median(within_all) <= BOUNDARY_RATIO_LIMIT),
        },
        "ba": {"gen_mean": float(np.nanmean(ba_f)), "gen_std_over_samples": float(np.nanstd(ba_f)),
               "gen_mean_of_prompt_means": float(prompt_means.mean()),
               "gen_std_over_prompt_means": float(prompt_means.std()),
               "gt_mean": float(np.nanmean([r["ba_full"] for r in gt_rows])),
               "additive_gen_mean": float(np.nanmean(ba_a)),
               "additive_gt_mean": float(np.nanmean([r["ba_additive"] for r in gt_rows])),
               "abs_gap_additive_vs_full_gen_mean": float(np.nanmean(np.abs(ba_f - ba_a))),
               "abs_gap_additive_vs_full_gen_max": float(np.nanmax(np.abs(ba_f - ba_a))),
               "abs_gap_additive_vs_full_gt_mean": float(np.nanmean([abs(r["ba_full"] - r["ba_additive"]) for r in gt_rows])),
               "signed_gap_full_minus_additive_gen_mean": float(np.nanmean(ba_f - ba_a)),
               "n_prompts_without_beats": int(sum(len(b) == 0 for b in prompts.beat_frames)),
               "sigma": 3.0},
        "pfc": {"gen_mean": float(pf.mean()), "gen_std_over_samples": float(pf.std()),
                "gt_mean": float(np.mean([r["pfc"] for r in gt_rows]))},
        "seconds": round(time.time() - t0, 1),
        "per_sample": per_sample, "gt": gt_rows,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = args.tag or "phase1_checks"
    json.dump(out, open(os.path.join(RESULTS_DIR, name + ".json"), "w"), indent=1)
    summary = {k: v for k, v in out.items() if k not in ("per_sample", "gt")}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(RESULTS_DIR, name + ".md"), "w") as f:
        f.write("```\n" + json.dumps(summary, indent=2) + "\n```\n")


def model_table(paths):
    """Markdown table comparing the Phase 1 checks of several models (one checks json each)."""
    cols = [(os.path.splitext(os.path.basename(p))[0], json.load(open(p))) for p in paths]
    rows = []

    def add(label, fn, fmt="{:.4f}"):
        rows.append([label] + [fmt.format(fn(c)) if fn(c) is not None else "-" for _, c in cols])

    add("checkpoint", lambda c: ", ".join(c["ckpt"]), "{}")
    add("checkpoint step", lambda c: ", ".join(str(v) for v in c.get("ckpt_step", [])) or "-", "{}")
    add("loss / representation", lambda c: "; ".join(f'{d.get("repr")}, root_weight {d.get("root_weight")}, seam_lambda {d.get("seam_lambda")}' for d in c.get("loss_cfg", [])) or "relative (Phase 1)", "{}")
    add("frames with root outside [-1, 1] (clipped before FK): n / frac / max |root|", lambda c: (lambda r: f'{r["n_frames"]} / {r["frac_frames"]:.5f} / {r["max_abs_root"]:.3f}')(c["root_out_of_range"]) if "root_out_of_range" in c else None, "{}")
    add("sequences (prompts x seeds)", lambda c: c["n_samples"], "{}")
    add("NFE per sequence = K x 50 x 2 for all", lambda c: f'{c["nfe"]["min"]}..{c["nfe"]["max"]} ({"pass" if c["nfe"]["all_equal_expected"] else "FAIL"})', "{}")
    add("determinism: bitwise-equal / compared, per compare root", lambda c: "; ".join(f'{r["n_bitwise_equal"]}/{r["n_compared"]} (max diff {r["max_abs_diff"]:g}, {r["n_pairs_on_different_gpus"]} cross-GPU)' for r in c["determinism"]) or "no compare root", "{}")
    b = lambda c: c["boundary"]
    add("boundary ratio (mean seam / median within), limit 3", lambda c: f'{b(c)["ratio_pooled_mean_seam_over_median_within"]:.3f} ({"PASS" if b(c)["pass"] else "FAIL"})', "{}")
    add("  median seam / median within", lambda c: b(c)["ratio_pooled_median_seam_over_median_within"], "{:.3f}")
    add("  per-sample ratio mean / max", lambda c: f'{b(c)["ratio_per_sample_mean"]:.2f} / {b(c)["ratio_per_sample_max"]:.2f}', "{}")
    add("  seam jump mean (m) / within median (m)", lambda c: f'{b(c)["seam_jump_mean"]:.4f} / {b(c)["within_jump_median"]:.4f}', "{}")
    add("  per boundary k=1,2,3 seam mean (m)", lambda c: ", ".join(f"{v:.4f}" for v in b(c).get("per_boundary_seam_mean", [])) or "-", "{}")
    add("  root-translation: seam / within / ratio", lambda c: (lambda r: f'{r["seam_mean"]:.4f} / {r["within_median"]:.4f} / {r["ratio"]:.2f}')(b(c)["root_translation"]) if "root_translation" in b(c) else None, "{}")
    add("  root-relative pose: seam / within / ratio", lambda c: (lambda r: f'{r["seam_mean"]:.4f} / {r["within_median"]:.4f} / {r["ratio"]:.2f}')(b(c)["root_relative_pose"]) if "root_relative_pose" in b(c) else None, "{}")
    add("  jump profile 146->147 .. 152->153 (m)", lambda c: ", ".join(f"{v:.4f}" for v in b(c)["jump_profile_146_152"].values()) if "jump_profile_146_152" in b(c) else None, "{}")
    add("BA gen mean (std over prompt means)", lambda c: f'{c["ba"]["gen_mean"]:.4f} ({c["ba"]["gen_std_over_prompt_means"]:.4f})', "{}")
    add("BA gt mean", lambda c: c["ba"]["gt_mean"])
    add("PFC gen mean (x1e4)", lambda c: c["pfc"]["gen_mean"], "{:.3f}")
    add("PFC gt mean (x1e4)", lambda c: c["pfc"]["gt_mean"], "{:.3f}")
    add("R_BA additive gen mean", lambda c: c["ba"]["additive_gen_mean"])
    add("|additive - full| gen mean / max", lambda c: f'{c["ba"]["abs_gap_additive_vs_full_gen_mean"]:.4f} / {c["ba"]["abs_gap_additive_vs_full_gen_max"]:.4f}', "{}")
    add("|additive - full| gt mean", lambda c: c["ba"]["abs_gap_additive_vs_full_gt_mean"])
    add("wall per sequence (s)", lambda c: c["wall"]["mean_per_sequence_s"], "{:.3f}")
    gt = cols[0][1]["boundary"]
    head = "| check | " + " | ".join(n for n, _ in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in rows)
    note = (f'\nGround truth on the same 40 prompts: boundary ratio {gt["gt_ratio_pooled_mean_seam_over_median_within"]:.3f}'
            + (f', root-translation ratio {gt["root_translation"]["gt_ratio"]:.3f}, pose ratio {gt["root_relative_pose"]["gt_ratio"]:.3f}, jump profile '
               + ", ".join(f"{v:.4f}" for v in gt["gt_jump_profile_146_152"].values()) if "root_translation" in gt else "") + ".\n")
    return head + body + note


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--samples", default=os.path.join(ROOT, "samples"))
    ap.add_argument("--compare", default="", help="comma-separated other sample roots for the determinism check")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="", help="output name under results/ (default phase1_checks)")
    ap.add_argument("--model_table", default="", help="comma-separated checks jsons -> results/phase1b_table.md")
    a = ap.parse_args()
    if a.phase1:
        phase1(a)
    if a.model_table:
        md = model_table(a.model_table.split(","))
        open(os.path.join(RESULTS_DIR, "phase1b_table.md"), "w").write(md)
        print(md)
