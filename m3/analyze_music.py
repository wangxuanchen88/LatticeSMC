"""Music grid analysis for Phase M3 (reward clap) and Phase M4 (reward motif, Amendment V), with the Amendment U argmax rows.

  .venv/bin/python m3/analyze_music.py --samples results/m4/samples --out results/m4 --reward motif --tag m4

Primary reward per sequence = the value recomputed from the saved audio (scores.json: clap_terminal_recomputed for
"clap", motif_recomputed for "motif"; the max abs difference to the logged final_reward is the root-of-trust check).
Argmax rows (rule "argmax") come from the logged per-particle rewards (exact: the logged rewards are the terminal scores
of every particle); their held-out values are only available where the argmax particle was regenerated. Held out for
"motif": terminal CLAP (clap_terminal_recomputed), tempo, 8 kHz, aesthetics. Bootstrap 1000 over prompts, seed 0.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

METHODS = ["base", "bon_argmax", "bon_is", "greedy_chunk", "fk_noise", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"]
PROPER = ["bon_is", "fk_noise", "lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"]
NS = [1, 2, 4, 8, 16, 32]
N_BOOT, SEED = 1000, 0
PAIRS = {"clap": [("lattice_smc_prefix_b1", "bon_is"), ("lattice_smc_prefix_b1", "greedy_chunk"), ("lattice_smc_prefix_b1", "fk_noise"), ("lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5"),
                  ("lattice_smc_prefix_b1", "lattice_smc_prefix_b0.25"), ("lattice_smc_prefix_b0.5", "lattice_smc_prefix_b0.25"), ("lattice_smc_prefix_b1-argmax", "greedy_chunk")],
         "motif": [("lattice_smc_prefix_b1", "greedy_chunk"), ("lattice_smc_prefix_b1", "bon_is"), ("lattice_smc_prefix_b1-argmax", "greedy_chunk"), ("lattice_smc_prefix_b1-argmax", "bon_is"),
                   ("greedy_chunk", "bon_is"), ("lattice_smc_prefix_b1", "fk_noise"), ("lattice_smc_prefix_b1", "lattice_smc_prefix_b0.5"), ("lattice_smc_prefix_b1", "lattice_smc_prefix_b0.25"),
                   ("lattice_smc_prefix_b1-argmax", "lattice_smc_prefix_b1")]}
OTHER = [("bon_is", "bon_argmax"), ("greedy_chunk", "bon_argmax"), ("fk_noise", "bon_is"), ("fk_noise", "greedy_chunk"), ("lattice_smc_prefix_b1", "bon_argmax"), ("lattice_smc_prefix_b1", "base"),
         ("fk_noise-argmax", "greedy_chunk"), ("bon_is-argmax", "bon_argmax")]


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
    return float(v.mean()), float(np.percentile(v[idx].mean(1), 2.5)), float(np.percentile(v[idx].mean(1), 97.5))


def per_prompt(vals):
    by = {}
    for (p, s), r in vals.items():
        by.setdefault(p, []).append(r)
    return {p: float(np.mean(v)) for p, v in by.items()}


def seed_div(embs):
    by = {}
    for (p, s), e in embs.items():
        if e is not None:
            by.setdefault(p, []).append(e / np.linalg.norm(e))
    vals = [float(np.mean([1 - float(E[i] @ E[j]) for i, j in itertools.combinations(range(len(E)), 2)])) for E in by.values() if len(E) >= 2]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reward", default="clap", choices=["clap", "motif"])
    ap.add_argument("--tag", default="m3")
    ap.add_argument("--argmax_samples", default=None, help="root of regenerated argmax clips (held-out and diversity for argmax rows)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data = load_logs(a.samples)
    S = lambda name: {r["key"]: r for r in json.load(open(os.path.join(a.out, name)))["rows"]} if os.path.exists(os.path.join(a.out, name)) else {}  # noqa: E731
    scores, tempo, aes = S("scores.json"), S("tempo.json"), S("aesthetics.json")
    adata = load_logs(a.argmax_samples) if a.argmax_samples else {}
    ascores, atempo, aaes = S("argmax_scores.json"), S("argmax_tempo.json"), S("argmax_aesthetics.json")
    rkey = "clap_terminal_recomputed" if a.reward == "clap" else "motif_recomputed"

    def reward(rec):
        s = scores.get(rec["key"])
        return s[rkey] if s else rec["log"]["final_reward"]

    diffs = [s[("clap_abs_diff" if a.reward == "clap" else "motif_abs_diff")] for s in scores.values() if s.get("clap_abs_diff" if a.reward == "clap" else "motif_abs_diff") is not None]
    trust = {"n_recomputed": len(diffs), "n_sequences": len(data), "max_abs_diff_recomputed_vs_logged": float(max(diffs)) if diffs else None}
    heldout_fields = [("frac_energy_above_8k", scores, ascores, "frac_energy_above_8k"), ("seam_ratio", scores, ascores, "seam_mean_ratio"), ("tempo_reward", tempo, atempo, "tempo_reward_additive"),
                      ("aes_CE", aes, aaes, "CE"), ("aes_CU", aes, aaes, "CU"), ("aes_PC", aes, aaes, "PC"), ("aes_PQ", aes, aaes, "PQ")]
    if a.reward == "motif":
        heldout_fields = [("clap_terminal", scores, ascores, "clap_terminal_recomputed"), ("s3_chunk3_to_chunk1", scores, ascores, "s3_recomputed")] + heldout_fields
    rows, curve, byrow = [], {}, {}
    for m in METHODS:
        for N in NS:
            sel = {(p, s): v for (mm, n, p, s), v in data.items() if mm == m and n == N}
            if not sel:
                continue
            R = {k: reward(v) for k, v in sel.items()}
            mean, lo, hi = boot(list(per_prompt(R).values()), rng)
            row = {"method": m, "N": N, "rule": "draw" if m in PROPER else ("argmax" if m in ("bon_argmax", "greedy_chunk") else "-"), "n_sequences": len(sel), "n_prompts": len(per_prompt(R)),
                   "nfe": sorted({v["log"]["nfe"] for v in sel.values()}), "R_mean": mean, "ci_lo": lo, "ci_hi": hi,
                   "particle_mean": float(np.mean([np.mean(v["log"]["rewards"]) for v in sel.values()])), "particle_max": float(np.mean([np.max(v["log"]["rewards"]) for v in sel.values()])),
                   "decode_calls": float(np.mean([v["log"]["decode_calls"] for v in sel.values()])), "clap_calls": float(np.mean([v["log"]["clap_calls"] for v in sel.values()])),
                   "wall_s": float(np.mean([v["log"]["wall"] for v in sel.values()])), "seed_diversity_cos_dist": seed_div({k: v["emb"] for k, v in sel.items()})}
            for k in (1, 2, 3):
                ev = [e for v in sel.values() for e in v["log"]["ess_content"] if e["chunk"] == k and not e.get("final_draw")]
                row[f"resampled_k{k}"] = float(np.mean([e["resampled"] for e in ev])) if ev else float("nan")
                row[f"ess_over_N_k{k}"] = float(np.mean([e["ess"] for e in ev]) / N) if ev else float("nan")
            for k in (1, 2, 3, 4):
                ev = [e for v in sel.values() for e in v["log"]["ess_noise"] if e["chunk"] == k and not e.get("final_draw")]
                if ev:
                    row[f"noise_resampled_k{k}"] = float(np.mean([e["resampled"] for e in ev]))
            tel = [v["log"].get("telescoping_max_abs_err") for v in sel.values() if v["log"].get("telescoping_max_abs_err") is not None]
            if tel:
                row["telescoping_max_abs_err"] = float(np.max(tel))
            for name, src, _, field in heldout_fields:
                vals = [src[v["key"]][field] for v in sel.values() if v["key"] in src and src[v["key"]].get(field) is not None]
                if vals:
                    row[name], row[name + "_n"] = float(np.mean(vals)), len(vals)
            rows.append(row)
            byrow[(m, N)] = R
            if m in PROPER:  # Amendment U: argmax-return row from the logged per-particle rewards
                RA = {k: float(np.max(v["log"]["rewards"])) for k, v in sel.items()}
                ma, la, ha = boot(list(per_prompt(RA).values()), rng)
                arow = {"method": m + "-argmax", "N": N, "rule": "argmax", "n_sequences": len(sel), "n_prompts": len(per_prompt(RA)), "nfe": row["nfe"], "R_mean": ma, "ci_lo": la, "ci_hi": ha,
                        "particle_mean": row["particle_mean"], "particle_max": row["particle_max"], "decode_calls": row["decode_calls"], "clap_calls": row["clap_calls"], "wall_s": row["wall_s"],
                        "frac_argmax_equals_draw": float(np.mean([int(np.argmax(v["log"]["rewards"])) == v["log"]["chosen"] for v in sel.values()])), "seed_diversity_cos_dist": float("nan")}
                asel = {(p, s): v for (mm, n, p, s), v in adata.items() if mm == m and n == N}
                if asel:
                    arow["regenerated"] = len(asel)
                    arow["replay_bitwise_all"] = bool(all(v["log"]["replay"]["chosen_latent_bitwise"] for v in asel.values()))
                    arow["seed_diversity_cos_dist"] = seed_div({k: v["emb"] for k, v in asel.items()})
                    for name, _, asrc, field in heldout_fields:
                        vals = [asrc[v["key"]][field] for v in asel.values() if v["key"] in asrc and asrc[v["key"]].get(field) is not None]
                        if vals:
                            arow[name], arow[name + "_n"] = float(np.mean(vals)), len(vals)
                rows.append(arow)
                byrow[(m + "-argmax", N)] = RA
    pairs = []
    for A, B in PAIRS[a.reward] + OTHER:
        for N in (8, 32):
            da, db = byrow.get((A, N)), byrow.get((B, 1 if B == "base" else N))
            if not da or not db:
                continue
            by = {}
            for k in da:
                if k in db:
                    by.setdefault(k[0], []).append(da[k] - db[k])
            d = [float(np.mean(v)) for v in by.values()]
            m, lo, hi = boot(d, rng)
            pairs.append({"A": A, "B": B, "N": N, "n_prompts": len(d), "diff": m, "ci_lo": lo, "ci_hi": hi, "separates": bool(lo > 0 or hi < 0), "primary": (A, B) in PAIRS[a.reward]})
    q = {f"{p['A']} - {p['B']} @ N={p['N']}": {"diff": p["diff"], "ci": [p["ci_lo"], p["ci_hi"]], "verdict": (("A higher" if p["diff"] > 0 else "B higher") if p["separates"] else "no separation")}
         for p in pairs if p["primary"]}
    out = {"reward": a.reward, "rows": rows, "pairs": pairs, "pre_registered": q, "root_of_trust": trust,
           "nfe_all_32N": all(r["nfe"] == [32 * r["N"]] for r in rows)}
    json.dump(out, open(os.path.join(a.out, f"{a.tag}_analysis.json"), "w"), indent=1)
    fmt = {"resampled_k1": "{:.2f}", "resampled_k2": "{:.2f}", "resampled_k3": "{:.2f}", "ess_over_N_k1": "{:.3f}", "ess_over_N_k2": "{:.3f}", "ess_over_N_k3": "{:.3f}", "decode_calls": "{:.0f}",
           "clap_calls": "{:.0f}", "wall_s": "{:.1f}", "telescoping_max_abs_err": "{:.1e}", "noise_resampled_k1": "{:.2f}", "noise_resampled_k2": "{:.2f}", "noise_resampled_k3": "{:.2f}",
           "noise_resampled_k4": "{:.2f}", "frac_energy_above_8k": "{:.5f}", "tempo_reward": "{:.3f}", "aes_CE": "{:.2f}", "aes_CU": "{:.2f}", "aes_PC": "{:.2f}", "aes_PQ": "{:.2f}",
           "seed_diversity_cos_dist": "{:.4f}", "seam_ratio": "{:.3f}", "frac_argmax_equals_draw": "{:.2f}", "clap_terminal": "{:.4f}", "s3_chunk3_to_chunk1": "{:.4f}"}
    label = "terminal CLAP" if a.reward == "clap" else "R_motif"
    md = f"# {a.tag} tables ({label})\n\n## Reward vs NFE (draw rows and argmax rows; bootstrap over prompts)\n\n"
    md += md_table(rows, ["method", "N", "rule", "n_sequences", "nfe", "R_mean", "ci_lo", "ci_hi", "particle_mean", "particle_max", "frac_argmax_equals_draw", "decode_calls", "clap_calls", "wall_s"], fmt)
    md += "\n## Resampling activity\n\n" + md_table([r for r in rows if r["N"] > 1 and r["rule"] != "argmax"], ["method", "N", "resampled_k1", "resampled_k2", "resampled_k3", "ess_over_N_k1", "ess_over_N_k2", "ess_over_N_k3",
                                                                                                             "noise_resampled_k1", "noise_resampled_k2", "noise_resampled_k3", "noise_resampled_k4", "telescoping_max_abs_err"], fmt)
    md += "\n## Pre-registered comparisons\n\n" + md_table([p for p in pairs if p["primary"]], ["A", "B", "N", "n_prompts", "diff", "ci_lo", "ci_hi", "separates"])
    md += "\n## Other pairs\n\n" + md_table([p for p in pairs if not p["primary"]], ["A", "B", "N", "n_prompts", "diff", "ci_lo", "ci_hi", "separates"])
    hcols = ["method", "N", "rule", "R_mean"] + [f[0] for f in heldout_fields] + ["tempo_reward_n", "seed_diversity_cos_dist", "regenerated"]
    md += "\n## Held out (never used for steering) and diversity, N in {1, 8, 32}\n\n" + md_table([r for r in rows if r["N"] in (1, 8, 32)], hcols, fmt)
    md += "\n## Root of trust, NFE audit, verdicts\n\n```\n" + json.dumps({"root_of_trust": trust, "nfe_all_32N": out["nfe_all_32N"], "pre_registered": q}, indent=1) + "\n```\n"
    open(os.path.join(a.out, f"{a.tag}_tables.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
