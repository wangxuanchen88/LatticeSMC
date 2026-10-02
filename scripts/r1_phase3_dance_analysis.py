"""R1 Phase 3 (dance) analysis: K = 6 (40 conditions) and K = 8 (16 conditions) on the 4M model, beat alignment and
repetition, N in {8, 32}; methods best-of-N (draw / argmax), chunk pruning, boundary schedule (draw / argmax), dense
schedule (draw / argmax). Full reward of the returned sequence (metrics caches; argmax rows regenerated), held-out PFC, W1,
root-relative diversity, ESS before every boundary and the fraction of boundaries at which resampling fired, paired
differences at N = 32 (1000 resamples over conditions, seed 0). For K = 8 the K = 4 grids restricted to the same 16 prompts
are reported alongside. Writes results/r1/dance_k{6,8}.json and results/r1/dance_k{6,8}.tex.

  LATTICE_PROMPTS is not needed: metrics caches are read, diversity is computed from the stored motions on the GPU.
  .venv/bin/python scripts/r1_phase3_dance_analysis.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", "."))
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from analyze_u import boot, dance_logs, met_rows, metrics_for, paired, per_prompt, root_rel_diversity  # noqa: E402

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
R1 = os.path.join(ROOT, "results", "r1")
rng = np.random.default_rng(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
METHODS = [("best-of-N (draw)", "bon_is", "draw", ""), ("best-of-N (argmax)", "bon_argmax", "argmax", ""), ("chunk pruning", "greedy_chunk", "argmax", ""),
           ("boundary (draw)", "lattice_smc_notwist", "draw", ""), ("boundary (argmax)", "lattice_smc_notwist", "argmax", "_argmax"),
           ("dense (draw)", "fk_noise", "draw", ""), ("dense (argmax)", "fk_noise", "argmax", "_argmax")]


def ess_stats(root, method, N):
    lg = dance_logs(root, method, N)
    ess_by_k, fired_by_k = {}, {}
    for f in [v["file"] for v in lg.values()]:
        log = json.loads(str(np.load(f)["log"]))
        for e in log.get("ess_content", []):
            if e.get("final_draw"):
                continue
            ess_by_k.setdefault(e["chunk"], []).append(e["ess"] / N)
            fired_by_k.setdefault(e["chunk"], []).append(float(e["resampled"]))
    return {str(k): float(np.mean(v)) for k, v in sorted(ess_by_k.items())}, {str(k): float(np.mean(v)) for k, v in sorted(fired_by_k.items())}, (float(np.mean([x for v in fired_by_k.values() for x in v])) if fired_by_k else None)


METHODS_A001 = [("boundary (draw), alpha 0.01", "lattice_smc_notwist", "draw", ""), ("boundary (argmax), alpha 0.01", "lattice_smc_notwist", "argmax", "_argmax"),
                ("dense (draw), alpha 0.01", "fk_noise", "draw", ""), ("dense (argmax), alpha 0.01", "fk_noise", "argmax", "_argmax")]


def base_ci(tag, rkey, restrict=None):
    """Base (N = 1) held-out PFC and W1 with bootstrap intervals over the same conditions."""
    mr = met_rows(metrics_for(tag), "base", 1)
    if restrict is not None:
        mr = {k: v for k, v in mr.items() if k[0] in restrict}
    out = {}
    for key in ("pfc", "realism_w1"):
        m, lo, hi = boot(list(per_prompt({k: r[key] for k, r in mr.items()}).values()), rng)
        out[key] = {"mean": m, "ci": [lo, hi]}
    return out


def table(K, rkey, restrict=None):
    """Rows for one (K, reward). restrict: set of prompt ids for the K = 4 restriction (else all)."""
    tag = f"r1_k{K}_{rkey}"
    full_key = f"{rkey}_full"
    rows, seqs = [], {}
    for N in (1, 8, 32):
        for name, m, rule, suffix in ([("base", "base", "-", "")] if N == 1 else METHODS):
            mr = met_rows(metrics_for(tag + suffix), m, N)
            if restrict is not None:
                mr = {k: v for k, v in mr.items() if k[0] in restrict}
            if not mr:
                continue
            full = {k: r[full_key] for k, r in mr.items()}
            mean, lo, hi = boot(list(per_prompt(full).values()), rng)
            root = os.path.join(ROOT, f"samples_r1_k{K}_{rkey}{suffix}")
            ess, fired, fired_all = ess_stats(root, m, N) if m in ("lattice_smc_notwist", "fk_noise", "greedy_chunk") else ({}, {}, None)
            rows.append({"K": K, "reward": rkey, "method": name, "N": N, "n_sequences": len(mr), "n_prompts": len(per_prompt(full)), "reward_full": mean, "ci": [lo, hi],
                         "pfc": float(np.mean([r["pfc"] for r in mr.values()])), "realism_w1": float(np.mean([r["realism_w1"] for r in mr.values()])),
                         "nfe": sorted({r["nfe"] for r in mr.values()}), "diversity_root_rel": root_rel_diversity(f"samples_r1_k{K}_{rkey}{suffix}", m, N, device),
                         "ess_over_N_before_boundary": ess, "frac_resampled_by_boundary": fired, "frac_boundaries_resampled": fired_all,
                         "source": f"results/{tag}{suffix}_metrics.json"})
            seqs[(name, N)] = full
    if rkey == "ba":  # Phase 3b: alpha = 0.01 rows at N = 32
        for name, m, rule, suffix in METHODS_A001:
            t = f"r1_k{K}_ba_a001{suffix}"
            mr = met_rows(metrics_for(t), m, 32) if os.path.exists(os.path.join(ROOT, "results", f"{t}_metrics.json")) else {}
            if not mr:
                continue
            full = {k: r[full_key] for k, r in mr.items()}
            mean, lo, hi = boot(list(per_prompt(full).values()), rng)
            ess, fired, fired_all = ess_stats(os.path.join(ROOT, f"samples_r1_k{K}_ba_a001{suffix}"), m, 32)
            rows.append({"K": K, "reward": rkey, "method": name, "N": 32, "alpha": 0.01, "n_sequences": len(mr), "n_prompts": len(per_prompt(full)), "reward_full": mean, "ci": [lo, hi],
                         "pfc": float(np.mean([r["pfc"] for r in mr.values()])), "realism_w1": float(np.mean([r["realism_w1"] for r in mr.values()])), "nfe": sorted({r["nfe"] for r in mr.values()}),
                         "diversity_root_rel": root_rel_diversity(f"samples_r1_k{K}_ba_a001{suffix}", m, 32, device), "ess_over_N_before_boundary": ess, "frac_resampled_by_boundary": fired,
                         "frac_boundaries_resampled": fired_all, "source": f"results/{t}_metrics.json"})
            seqs[(name, 32)] = full
    pairs = []
    PAIRS = [("boundary (argmax)", "chunk pruning"), ("boundary (argmax)", "best-of-N (argmax)"), ("boundary (draw)", "best-of-N (draw)"), ("boundary (draw)", "chunk pruning"),
             ("dense (argmax)", "chunk pruning"), ("dense (argmax)", "best-of-N (argmax)"), ("dense (draw)", "best-of-N (draw)"), ("dense (draw)", "chunk pruning"),
             ("dense (argmax)", "boundary (argmax)"), ("dense (draw)", "boundary (draw)"), ("boundary (argmax)", "boundary (draw)"), ("dense (argmax)", "dense (draw)"),
             ("boundary (argmax), alpha 0.01", "chunk pruning"), ("boundary (draw), alpha 0.01", "chunk pruning"), ("dense (argmax), alpha 0.01", "chunk pruning"), ("dense (draw), alpha 0.01", "chunk pruning"),
             ("boundary (argmax), alpha 0.01", "boundary (argmax)"), ("dense (argmax), alpha 0.01", "dense (argmax)"), ("boundary (argmax), alpha 0.01", "best-of-N (argmax)"), ("dense (argmax), alpha 0.01", "best-of-N (argmax)")]
    for A, B in PAIRS:
        if (A, 32) in seqs and (B, 32) in seqs:
            pairs.append({"K": K, "reward": rkey, "A": A, "B": B, "N": 32, **paired(seqs[(A, 32)], seqs[(B, 32)], rng)})
    return rows, pairs


def k4_restricted(rkey, prompts16):
    """K = 4 rows from the existing grids restricted to the 16 K = 8 prompts (paired over 16 conditions)."""
    tag, atag = ("phase2b_a002", "u_argmax_p2_a002") if rkey == "ba" else ("phase3_rep_a002", "u_argmax_p3_rep_a002")
    fk_argmax_tag = "verify_fkargmax_p2_a002" if rkey == "ba" else None
    full_key = f"{rkey}_full"
    rows, seqs = [], {}
    for N in (1, 8, 32):
        for name, m, rule, suffix in ([("base", "base", "-", "")] if N == 1 else METHODS):
            src = tag if suffix == "" else (atag if m == "lattice_smc_notwist" else fk_argmax_tag)
            if src is None:  # dense argmax at K = 4: full reward exists at N = 32 only (2026-09-18 regeneration)
                continue
            mr = met_rows(metrics_for(src), m, N)
            mr = {k: v for k, v in mr.items() if k[0] in prompts16}
            if not mr:
                continue
            full = {k: r[full_key] for k, r in mr.items()}
            mean, lo, hi = boot(list(per_prompt(full).values()), rng)
            rows.append({"K": 4, "reward": rkey, "method": name, "N": N, "n_sequences": len(mr), "n_prompts": len(per_prompt(full)), "reward_full": mean, "ci": [lo, hi],
                         "pfc": float(np.mean([r["pfc"] for r in mr.values()])), "realism_w1": float(np.mean([r["realism_w1"] for r in mr.values()])), "source": f"results/{src}_metrics.json (16 prompts)"})
            seqs[(name, N)] = full
    pairs = []
    for A, B in (("boundary (argmax)", "chunk pruning"), ("boundary (argmax)", "best-of-N (argmax)"), ("boundary (draw)", "best-of-N (draw)"), ("boundary (draw)", "chunk pruning"), ("dense (argmax)", "chunk pruning"), ("dense (argmax)", "boundary (argmax)"),
                 ("dense (argmax)", "best-of-N (argmax)"), ("dense (draw)", "best-of-N (draw)"), ("dense (draw)", "chunk pruning"), ("dense (draw)", "boundary (draw)")):
        if (A, 32) in seqs and (B, 32) in seqs:
            pairs.append({"K": 4, "reward": rkey, "A": A, "B": B, "N": 32, **paired(seqs[(A, 32)], seqs[(B, 32)], rng)})
    return rows, pairs


def tex_rows(rows, pairs, title):
    out = [f"% {title}", "\\begin{tabular}{llcccc}", "\\toprule", "Method & $N$ & Reward [95\\% CI] & PFC & $W_1$ & diversity \\\\", "\\midrule"]
    for r in rows:
        out.append(f"{r['method']} & {r['N']} & ${r['reward_full']:.3f}$ $[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]$ & {r['pfc']:.2f} & {r['realism_w1']:.4f} & {r.get('diversity_root_rel', float('nan')):.3f} \\\\")
    out += ["\\midrule"]
    for p in pairs:
        out.append(f"{p['A']} $-$ {p['B']} & {p['N']} & ${p['diff']:+.3f}$ $[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]${'$^*$' if p['separates'] else ''} & & & \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


if __name__ == "__main__":
    p8 = json.load(open(os.path.join(ROOT, "data", "prompts_K8.json")))
    prompts16 = {p["paper_prompt_id"] for p in p8["prompts"]}
    for K in (6, 8):
        out = {"K": K, "n_boot": 1000, "seed": 0, "alpha": 0.02, "rewards": {}}
        tex = []
        for rkey in ("ba", "rep"):
            rows, pairs = table(K, rkey)
            out["rewards"][rkey] = {"rows": rows, "pairs_N32": pairs, "base_heldout_ci": base_ci(f"r1_k{K}_{rkey}", rkey)}
            tex.append(tex_rows(rows, pairs, f"K = {K}, {rkey}, 4M model, alpha 0.02, {rows[0]['n_prompts'] if rows else 0} conditions"))
            if K == 8:
                r4, p4 = k4_restricted(rkey, prompts16)
                out["rewards"][rkey]["k4_restricted_rows"] = r4
                out["rewards"][rkey]["k4_restricted_base_heldout_ci"] = base_ci("phase2b_a002" if rkey == "ba" else "phase3_rep_a002", rkey, prompts16)
                out["rewards"][rkey]["k4_restricted_pairs_N32"] = p4
                tex.append(tex_rows(r4, p4, f"K = 4 restricted to the same 16 conditions, {rkey}"))
            for r in rows:
                print(K, rkey, r["method"], r["N"], round(r["reward_full"], 4), [round(x, 4) for x in r["ci"]], "pfc", round(r["pfc"], 2), "fired", r["frac_boundaries_resampled"])
            for p in pairs:
                print(K, rkey, p["A"], "-", p["B"], f"{p['diff']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]", p["separates"])
        json.dump(out, open(os.path.join(R1, f"dance_k{K}.json"), "w"), indent=1)
        open(os.path.join(R1, f"dance_k{K}.tex"), "w").write("\n\n".join(tex) + "\n")
