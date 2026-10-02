"""Phase M3 analysis (Amendment T), as in Phase 2b, from the method logs under results/m3/samples plus the
scoring files (scores.json: recomputed terminal CLAP, 8 kHz fraction, seam; tempo.json; aesthetics.json).

  .venv/bin/python m3/analyze_m3.py [--samples results/m3/samples] [--out results/m3]

Writes m3_analysis.json, m3_tables.md, fig_m3_reward_vs_nfe.png. Primary reward = terminal CLAP of the returned
clip (root of trust: the value recomputed from the saved audio in scores.json when present, otherwise the log's
final_reward; the max abs difference between the two is reported). Bootstrap: 1000 resamples over prompts of the
per-prompt mean over seeds, seed 0, percentile intervals; paired differences likewise on per-prompt mean differences.
"""
import argparse
import glob
import json
import os

import numpy as np

METHODS = ["base", "bon_argmax", "bon_is", "greedy_chunk", "fk_noise", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"]
NS = [1, 2, 4, 8, 16, 32]
N_BOOT, SEED = 1000, 0
PRIMARY_PAIRS = [("lattice_smc_prefix_b1", "bon_is"), ("lattice_smc_prefix_b1", "greedy_chunk"), ("lattice_smc_prefix_b1", "fk_noise"),
                 ("lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5"), ("lattice_smc_prefix_b1", "lattice_smc_prefix_b0.25"),
                 ("lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25")]
OTHER_PAIRS = [("bon_is", "bon_argmax"), ("greedy_chunk", "bon_argmax"), ("fk_noise", "bon_is"), ("fk_noise", "greedy_chunk"),
               ("lattice_smc_prefix_b1", "bon_argmax"), ("lattice_smc_prefix_b1", "base")]


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


def load_logs(root):
    data = {}
    for m in METHODS:
        for N in NS:
            for f in sorted(glob.glob(os.path.join(root, m, str(N), "*", "log.json"))):
                log = json.load(open(f))
                d = os.path.dirname(f)
                emb = np.load(os.path.join(d, "clap_embed.npy")) if os.path.exists(os.path.join(d, "clap_embed.npy")) else None
                data[(m, N, log["prompt_id"], log["base_seed"])] = {"log": log, "emb": emb, "key": f"{m}/{N}/{os.path.basename(d)}"}
    return data


def boot(v, rng):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_prompt(vals):
    by = {}
    for (p, s), r in vals.items():
        by.setdefault(p, []).append(r)
    return {p: float(np.mean(v)) for p, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3/samples")
    ap.add_argument("--out", default=os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3")
    ap.add_argument("--tag", default="m3")
    ap.add_argument("--reward_label", default="terminal CLAP")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data = load_logs(a.samples)
    scores = {r["key"]: r for r in json.load(open(os.path.join(a.out, "scores.json")))["rows"]} if os.path.exists(os.path.join(a.out, "scores.json")) else {}
    tempo = {r["key"]: r for r in json.load(open(os.path.join(a.out, "tempo.json")))["rows"]} if os.path.exists(os.path.join(a.out, "tempo.json")) else {}
    aes = {r["key"]: r for r in json.load(open(os.path.join(a.out, "aesthetics.json")))["rows"]} if os.path.exists(os.path.join(a.out, "aesthetics.json")) else {}

    def reward(rec):
        s = scores.get(rec["key"])
        return s["clap_terminal_recomputed"] if s else rec["log"]["final_reward"]

    # root of trust
    diffs = [s["clap_abs_diff"] for s in scores.values()]
    trust = {"n_recomputed": len(diffs), "n_sequences": len(data), "max_abs_diff_recomputed_vs_logged": float(max(diffs)) if diffs else None,
             "mean_abs_diff": float(np.mean(diffs)) if diffs else None}
    # curves and NFE audit
    rows, nfe_audit = [], {}
    curve = {}
    for m in METHODS:
        for N in NS:
            sel = {(p, s): v for (mm, n, p, s), v in data.items() if mm == m and n == N}
            if not sel:
                continue
            R = {k: reward(v) for k, v in sel.items()}
            pp = per_prompt(R)
            mean, lo, hi = boot(list(pp.values()), rng)
            nfes = sorted({v["log"]["nfe"] for v in sel.values()})
            nfe_audit[f"{m}/{N}"] = {"nfe_values": nfes, "expected": 32 * N, "all_equal_expected": nfes == [32 * N]}
            row = {"method": m, "N": N, "n_sequences": len(sel), "n_prompts": len(pp), "nfe": nfes[0] if len(nfes) == 1 else nfes,
                   "R_mean": mean, "ci_lo": lo, "ci_hi": hi,
                   "weighted_mean_reward": float(np.mean([v["log"]["weighted_mean_reward"] for v in sel.values()])),
                   "particle_mean": float(np.mean([np.mean(v["log"]["rewards"]) for v in sel.values()])),
                   "particle_max": float(np.mean([np.max(v["log"]["rewards"]) for v in sel.values()])),
                   "decode_calls": float(np.mean([v["log"]["decode_calls"] for v in sel.values()])),
                   "clap_calls": float(np.mean([v["log"]["clap_calls"] for v in sel.values()])),
                   "wall_s": float(np.mean([v["log"]["wall"] for v in sel.values()]))}
            # resampling fractions and ESS / N per chunk (content axis) or per chunk (noise axis, any step)
            for k in (1, 2, 3):
                ev = [e for v in sel.values() for e in v["log"]["ess_content"] if e["chunk"] == k and not e.get("final_draw")]
                row[f"resampled_k{k}"] = float(np.mean([e["resampled"] for e in ev])) if ev else float("nan")
                row[f"ess_over_N_k{k}"] = float(np.mean([e["ess"] for e in ev]) / N) if ev else float("nan")
            for k in (1, 2, 3, 4):
                ev = [e for v in sel.values() for e in v["log"]["ess_noise"] if e["chunk"] == k and not e.get("final_draw")]
                if ev:
                    row[f"noise_resampled_k{k}"] = float(np.mean([e["resampled"] for e in ev]))
                    row[f"noise_ess_over_N_k{k}"] = float(np.mean([e["ess"] for e in ev]) / N)
            tel = [v["log"].get("telescoping_max_abs_err") for v in sel.values() if v["log"].get("telescoping_max_abs_err") is not None]
            if tel:
                row["telescoping_max_abs_err"] = float(np.max(tel))
            # diversity: mean pairwise cosine distance between the seeds' terminal CLAP embeddings of a prompt
            div = []
            for p in pp:
                E = [v["emb"] for (pp_, s), v in sel.items() if pp_ == p and v["emb"] is not None]
                if len(E) >= 2:
                    E = np.stack(E)
                    E = E / np.linalg.norm(E, axis=1, keepdims=True)
                    C = E @ E.T
                    iu = np.triu_indices(len(E), 1)
                    div.append(float(np.mean(1 - C[iu])))
            row["seed_diversity_cos_dist"] = float(np.mean(div)) if div else float("nan")
            # held-out
            for name, src, field in (("frac_energy_above_8k", scores, "frac_energy_above_8k"), ("seam_ratio", scores, "seam_mean_ratio"),
                                     ("tempo_reward", tempo, "tempo_reward_additive"), ("tempo_reward_partial", tempo, "tempo_reward_partial"),
                                     ("tempo_whole_within_5pct", tempo, "whole_within_5pct"),
                                     ("aes_CE", aes, "CE"), ("aes_CU", aes, "CU"), ("aes_PC", aes, "PC"), ("aes_PQ", aes, "PQ")):
                vals = [src[v["key"]][field] for v in sel.values() if v["key"] in src and src[v["key"]].get(field) is not None]
                if vals:
                    row[name] = float(np.mean(vals))
                    row[name + "_n"] = len(vals)
            rows.append(row)
            curve[(m, N)] = (mean, lo, hi, row["nfe"])
    # pairwise differences at N = 8 and 32
    pairs = []
    for A, B in PRIMARY_PAIRS + OTHER_PAIRS:
        for N in (8, 32):
            NB = 1 if B == "base" else N
            da = {(p, s): reward(v) for (m, n, p, s), v in data.items() if m == A and n == N}
            db = {(p, s): reward(v) for (m, n, p, s), v in data.items() if m == B and n == NB}
            common = [k for k in da if k in db]
            if not common:
                continue
            by = {}
            for k in common:
                by.setdefault(k[0], []).append(da[k] - db[k])
            d = [float(np.mean(v)) for v in by.values()]
            mean, lo, hi = boot(d, rng)
            pairs.append({"A": A, "B": B, "N": N, "n_prompts": len(d), "n_sequences": len(common), "diff": mean, "ci_lo": lo, "ci_hi": hi,
                          "separates": bool(lo > 0 or hi < 0), "primary": (A, B) in PRIMARY_PAIRS})
    q = {}
    for A, B in PRIMARY_PAIRS:
        for N in (8, 32):
            pr = next((p for p in pairs if p["A"] == A and p["B"] == B and p["N"] == N), None)
            if pr:
                q[f"{A} - {B} @ N={N}"] = {"diff": pr["diff"], "ci": [pr["ci_lo"], pr["ci_hi"]], "separates": pr["separates"],
                                          "verdict": ("A higher" if pr["diff"] > 0 else "B higher") if pr["separates"] else "no separation"}
    n_sep = sum(p["separates"] for p in pairs)
    out = {"rows": rows, "pairs": pairs, "pre_registered": q, "nfe_audit": nfe_audit, "root_of_trust": trust,
           "n_separating_pairs": n_sep, "n_pairs": len(pairs)}
    json.dump(out, open(os.path.join(a.out, f"{a.tag}_analysis.json"), "w"), indent=1)
    # tables
    fmt = {"resampled_k1": "{:.2f}", "resampled_k2": "{:.2f}", "resampled_k3": "{:.2f}", "ess_over_N_k1": "{:.3f}", "ess_over_N_k2": "{:.3f}",
           "ess_over_N_k3": "{:.3f}", "decode_calls": "{:.0f}", "clap_calls": "{:.0f}", "wall_s": "{:.1f}", "telescoping_max_abs_err": "{:.2e}",
           "noise_resampled_k1": "{:.2f}", "noise_resampled_k2": "{:.2f}", "noise_resampled_k3": "{:.2f}", "noise_resampled_k4": "{:.2f}",
           "noise_ess_over_N_k1": "{:.3f}", "noise_ess_over_N_k2": "{:.3f}", "noise_ess_over_N_k3": "{:.3f}", "noise_ess_over_N_k4": "{:.3f}",
           "frac_energy_above_8k": "{:.5f}", "tempo_reward": "{:.3f}", "tempo_reward_partial": "{:.3f}", "tempo_whole_within_5pct": "{:.3f}",
           "aes_CE": "{:.2f}", "aes_CU": "{:.2f}", "aes_PC": "{:.2f}", "aes_PQ": "{:.2f}", "seed_diversity_cos_dist": "{:.4f}", "seam_ratio": "{:.3f}"}
    md = f"# Phase M3 tables ({a.reward_label}, alpha from results/m2/alpha.json)\n\n## Reward vs NFE (mean over sequences, bootstrap over prompts)\n\n"
    md += md_table(rows, ["method", "N", "n_sequences", "n_prompts", "nfe", "R_mean", "ci_lo", "ci_hi", "weighted_mean_reward", "particle_mean", "particle_max", "decode_calls", "clap_calls", "wall_s"], fmt)
    md += "\n## Resampling activity (content axis: fraction of events resampled and ESS / N before events per chunk; noise axis: per chunk over the 4 events)\n\n"
    md += md_table([r for r in rows if r["N"] > 1], ["method", "N", "resampled_k1", "resampled_k2", "resampled_k3", "ess_over_N_k1", "ess_over_N_k2", "ess_over_N_k3",
                                                     "noise_resampled_k1", "noise_resampled_k2", "noise_resampled_k3", "noise_resampled_k4", "noise_ess_over_N_k1", "noise_ess_over_N_k4", "telescoping_max_abs_err"], fmt)
    md += "\n## Pre-registered comparisons (A - B, bootstrap over prompts of the per-prompt mean difference)\n\n"
    md += md_table([p for p in pairs if p["primary"]], ["A", "B", "N", "n_prompts", "n_sequences", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Other pairs\n\n" + md_table([p for p in pairs if not p["primary"]], ["A", "B", "N", "n_prompts", "n_sequences", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Held out (never used for steering) and diversity, N in {1, 8, 32}\n\n"
    md += md_table([r for r in rows if r["N"] in (1, 8, 32)], ["method", "N", "R_mean", "tempo_reward", "tempo_reward_n", "tempo_whole_within_5pct", "frac_energy_above_8k", "aes_CE", "aes_CU", "aes_PC", "aes_PQ", "seam_ratio", "seed_diversity_cos_dist"], fmt)
    md += "\n## Root of trust and NFE audit\n\n```\n" + json.dumps({"root_of_trust": trust, "nfe_all_equal_expected": all(v["all_equal_expected"] for v in nfe_audit.values()),
                                                                    "nfe_mismatches": {k: v for k, v in nfe_audit.items() if not v["all_equal_expected"]}}, indent=1) + "\n```\n"
    md += "\n## Pre-registered comparisons, verdicts\n\n```\n" + json.dumps(q, indent=1) + "\n```\n"
    open(os.path.join(a.out, f"{a.tag}_tables.md"), "w").write(md)
    print(md)
    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in METHODS:
            pts = [(N, *curve[(m, N)]) for N in NS if (m, N) in curve]
            if not pts:
                continue
            x = [p[4] for p in pts]
            y = [p[1] for p in pts]
            ax.errorbar(x, y, yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], marker="o", ms=4, lw=1.5, capsize=2, label=m)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("NFE per sequence (transformer rows)")
        ax.set_ylabel(a.reward_label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"fig_{a.tag}_reward_vs_nfe.png"), dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
