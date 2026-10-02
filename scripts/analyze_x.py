"""Amendment X analysis: the alpha = 0.02 R_BA and R_rep grids on the scaled base model chunk_dispE_L against the same
grids on the 4M model (E), both return rules. Rows per method at every N: draw rule = the returned sequence's full reward
from the metrics cache; argmax rule = the max of the logged final particle rewards (additive scale, exact for R_rep), with
the regenerated full reward, held-out values and diversity for lattice_smc_notwist at N in {8, 32}; bon_is-argmax is
bon_argmax's returned sample. Pre-registered question: do the N = 32 orderings under both rules match the 4M model's on
both rewards (reported as the two orderings side by side, Spearman rank correlation, and the set of pairwise sign
agreements). Writes results/x/x_{analysis.json,tables.md} and fig_crossover_x_{ba,rep}_{E,L}.{pdf,png} (the figures are written by analyze_w.crossover_figure into results/w and moved to results/x).

  .venv/bin/python scripts/analyze_x.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", "."))
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/scripts")
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_u import boot, dance_logs, met_rows, metrics_for, paired, per_prompt, root_rel_diversity  # noqa: E402
from analyze_w import ci, crossover_figure, f, md  # noqa: E402
from lattice_smc.analyze_phase2 import RESULTS_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "x")
os.makedirs(OUT, exist_ok=True)
NS = [1, 2, 4, 8, 16, 32]
PROPER = ["bon_is", "fk_noise", "lattice_smc", "lattice_smc_notwist"]
ALL = ["base", "bon_argmax", "bon_is", "greedy_chunk", "fk_noise", "lattice_smc", "lattice_smc_notwist"]
# (reward key, label, {model: (samples root, metrics tag, argmax root, argmax metrics tag, analysis tag)})
GRIDS = [("ba", "R_BA alpha 0.02", {"E (4.1 M)": ("samples_p2_a002", "phase2b_a002", "samples_p2_a002_argmax", "u_argmax_p2_a002", "phase2b_a002"),
                                    "L (37.6 M)": ("samples_L_ba_a002", "L_ba_a002", "samples_L_ba_a002_argmax", "L_argmax_ba_a002", "L_ba_a002")}),
         ("rep", "R_rep alpha 0.02", {"E (4.1 M)": ("samples_p3_rep_a002", "phase3_rep_a002", "samples_p3_rep_a002_argmax", "u_argmax_p3_rep_a002", "phase3_rep_a002"),
                                      "L (37.6 M)": ("samples_L_rep_a002", "L_rep_a002", "samples_L_rep_a002_argmax", "L_argmax_rep_a002", "L_rep_a002")})]


def grid_rows(rkey, root, tag, aroot, atag, ana_tag, rng, device):
    """Rows (method, N, rule) with reward (full where available, additive from the logs), held-out, diversity; per-sequence
    dicts for the pairwise cells."""
    met, amet = metrics_for(tag), metrics_for(atag)
    ana = json.load(open(os.path.join(RESULTS_DIR, f"{ana_tag}_analysis.json")))
    div = {(d["method"], d["N"]): d["pairwise_joint_dist_root_relative_m"] for d in ana["diversity"]}
    full_key, add_key = f"{rkey}_full", f"{rkey}_additive"
    rows, seq = [], {}
    for m in ALL:
        for N in NS:
            if m == "base" and N != 1:
                continue
            mr = met_rows(met, m, N)
            if not mr:
                continue
            full = {k: r[full_key] for k, r in mr.items()}
            add = {k: r[add_key] for k, r in mr.items()}
            mf = boot(list(per_prompt(full).values()), rng)
            ma = boot(list(per_prompt(add).values()), rng)
            rule = "draw" if m in PROPER else ("argmax" if m in ("bon_argmax", "greedy_chunk") else "-")
            rows.append({"method": m, "N": N, "rule": rule, "n": len(mr), "reward_full": mf[0], "full_lo": mf[1], "full_hi": mf[2], "reward_additive": ma[0], "add_lo": ma[1], "add_hi": ma[2],
                         "pfc": float(np.mean([r["pfc"] for r in mr.values()])), "realism_w1": float(np.mean([r["realism_w1"] for r in mr.values()])),
                         "travel_m": float(np.mean([r["travel_m"] for r in mr.values()])), "diversity": div.get((m, N), float("nan")),
                         "nfe": float(np.mean([r["nfe"] for r in mr.values()])), "wall_s": float(np.mean([r["wall"] for r in mr.values()])), "regenerated": False})
            seq[(m, N, "draw" if m in PROPER else "argmax")] = {"full": full, "additive": add}
            if m in PROPER:
                lg = dance_logs(root, m, N)
                am = {k: v["argmax"] for k, v in lg.items()}
                mam = boot(list(per_prompt(am).values()), rng)
                arow = {"method": m, "N": N, "rule": "argmax", "n": len(lg), "reward_full": float("nan"), "full_lo": float("nan"), "full_hi": float("nan"),
                        "reward_additive": mam[0], "add_lo": mam[1], "add_hi": mam[2], "pfc": float("nan"), "realism_w1": float("nan"), "travel_m": float("nan"), "diversity": float("nan"),
                        "nfe": rows[-1]["nfe"], "wall_s": rows[-1]["wall_s"], "regenerated": False,
                        "frac_argmax_equals_draw": float(np.mean([v["chosen"] == v["argmax_idx"] for v in lg.values()]))}
                seq[(m, N, "argmax")] = {"additive": am}
                if m == "bon_is":
                    ba = met_rows(met, "bon_argmax", N)
                    arow["equals_bon_argmax_max_abs_diff"] = float(max(abs(am[k] - ba[k][full_key]) for k in am if k in ba))
                    fb = {k: r[full_key] for k, r in ba.items()}
                    arow["reward_full"], arow["full_lo"], arow["full_hi"] = boot(list(per_prompt(fb).values()), rng)
                    arow.update({"pfc": float(np.mean([r["pfc"] for r in ba.values()])), "realism_w1": float(np.mean([r["realism_w1"] for r in ba.values()])),
                                 "travel_m": float(np.mean([r["travel_m"] for r in ba.values()])), "diversity": div.get(("bon_argmax", N), float("nan")), "regenerated": "= bon_argmax"})
                    seq[(m, N, "argmax")]["full"] = fb
                amr = met_rows(amet, m, N)
                if amr:
                    fa = {k: r[full_key] for k, r in amr.items()}
                    adds = {k: r[add_key] for k, r in amr.items()}
                    arow["regen_max_abs_diff_additive"] = float(max(abs(adds[k] - am[k]) for k in adds if k in am))
                    arow["reward_full"], arow["full_lo"], arow["full_hi"] = boot(list(per_prompt(fa).values()), rng)
                    arow.update({"pfc": float(np.mean([r["pfc"] for r in amr.values()])), "realism_w1": float(np.mean([r["realism_w1"] for r in amr.values()])),
                                 "travel_m": float(np.mean([r["travel_m"] for r in amr.values()])), "diversity": root_rel_diversity(aroot, m, N, device), "regenerated": True})
                    seq[(m, N, "argmax")]["full"] = fa
                rows.append(arow)
    return rows, seq, ana


def ordering(rows, N, rule_pick, scale):
    """Methods ordered by reward at N under a rule (rule_pick(method) -> row rule) on one scale: 'full' (the returned
    sequence's full reward; argmax rows without a regenerated sequence are skipped) or 'additive' (the logs' steering
    scale, available for every row; identical to full for R_rep)."""
    out = []
    for m in ALL:
        if m == "base":
            continue
        r = next((x for x in rows if x["method"] == m and x["N"] == N and x["rule"] == rule_pick(m)), None)
        if r is None:
            continue
        val = r["reward_full"] if scale == "full" else r["reward_additive"]
        if np.isnan(val):
            continue
        out.append((m, val, scale))
    out.sort(key=lambda t: -t[1])
    return out


def spearman(o1, o2):
    m1 = {m: i for i, (m, _, _) in enumerate(o1)}
    m2 = {m: i for i, (m, _, _) in enumerate(o2)}
    common = [m for m in m1 if m in m2]
    a = np.array([m1[m] for m in common], float)
    b = np.array([m2[m] for m in common], float)
    return float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 else float("nan")


def sign_agreement(o1, o2):
    m1 = {m: v for m, v, _ in o1}
    m2 = {m: v for m, v, _ in o2}
    common = [m for m in m1 if m in m2]
    agree, total, disagreements = 0, 0, []
    for a, b in itertools.combinations(common, 2):
        total += 1
        if np.sign(m1[a] - m1[b]) == np.sign(m2[a] - m2[b]):
            agree += 1
        else:
            disagreements.append(f"{a} vs {b}")
    return agree, total, disagreements


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    analysis, out = {}, ["# Amendment X: scaled base model (chunk_dispE_L, 37.6 M) vs the 4M model (E) on the alpha = 0.02 grids", ""]
    for rkey, label, models in GRIDS:
        analysis[label] = {}
        out += [f"## {label}", ""]
        data = {}
        for model, (root, tag, aroot, atag, ana_tag) in models.items():
            rows, seq, ana = grid_rows(rkey, root, tag, aroot, atag, ana_tag, rng, device)
            data[model] = (rows, seq, ana)
            gt = ana.get("gt_reference", {})
            cols = ["method", "N", "rule", "reward (full)", "reward (additive, logs)", "PFC", "realism W1", "travel m", "diversity (root-rel.)", "argmax = draw frac."]
            trows = []
            for r in rows:
                if r["N"] not in (1, 8, 32):
                    continue
                trows.append({"method": r["method"], "N": r["N"], "rule": r["rule"] + (" (regenerated)" if r["regenerated"] is True else (" (= bon_argmax)" if r["regenerated"] == "= bon_argmax" else (" (logs only)" if r["rule"] == "argmax" and r["method"] in PROPER else ""))),
                              "reward (full)": ci(r["reward_full"], r["full_lo"], r["full_hi"]), "reward (additive, logs)": ci(r["reward_additive"], r["add_lo"], r["add_hi"]),
                              "PFC": f(r["pfc"], 2), "realism W1": f(r["realism_w1"], 5), "travel m": f(r["travel_m"], 2), "diversity (root-rel.)": f(r["diversity"]),
                              "argmax = draw frac.": f(r.get("frac_argmax_equals_draw"), 2)})
            out += [f"### {model}: `{root}` (ground truth {rkey}_full {f(gt.get(rkey + '_full'))}, PFC {f(gt.get('pfc'), 2)}, W1 {f(gt.get('realism_w1_leave_one_out'), 5)}; NFE audit: {ana['nfe_audit']['n_with_wrong_nfe']} of {ana['nfe_audit']['n_sequences']} sequences off)", "",
                    md(trows, cols), ""]
            # curve by N (full reward, draw rows; argmax additive)
            analysis[label][model] = {"rows": rows}
            # pairwise cells
            pairs = []
            for N in (8, 32):
                g_full = seq[("greedy_chunk", N, "argmax")]["full"]
                g_add = seq[("greedy_chunk", N, "argmax")]["additive"]
                for m in ("lattice_smc_notwist", "lattice_smc", "fk_noise", "bon_is"):
                    if (m, N, "draw") not in seq:
                        continue
                    pd = paired(seq[(m, N, "draw")]["full"], g_full, rng)
                    pairs.append({"pair": f"{m} (draw) - greedy_chunk", "scale": "full", "N": N, **pd})
                    pa = paired(seq[(m, N, "argmax")]["additive"], g_add, rng)
                    pairs.append({"pair": f"{m}-argmax - greedy_chunk", "scale": "additive (logs)", "N": N, **pa})
                    if "full" in seq[(m, N, "argmax")]:
                        pf = paired(seq[(m, N, "argmax")]["full"], g_full, rng)
                        pairs.append({"pair": f"{m}-argmax - greedy_chunk", "scale": "full (regenerated)" if m != "bon_is" else "full (= bon_argmax)", "N": N, **pf})
                    pad = paired(seq[(m, N, "argmax")]["additive"], seq[(m, N, "draw")]["additive"], rng)
                    pairs.append({"pair": f"{m}-argmax - {m} (draw)", "scale": "additive (logs)", "N": N, **pad})
                if ("lattice_smc", N, "draw") in seq:
                    pl = paired(seq[("lattice_smc", N, "draw")]["full"], seq[("lattice_smc_notwist", N, "draw")]["full"], rng)
                    pairs.append({"pair": "lattice_smc (draw) - lattice_smc_notwist (draw)", "scale": "full", "N": N, **pl})
                    pl = paired(seq[("lattice_smc", N, "argmax")]["additive"], seq[("lattice_smc_notwist", N, "argmax")]["additive"], rng)
                    pairs.append({"pair": "lattice_smc-argmax - lattice_smc_notwist-argmax", "scale": "additive (logs)", "N": N, **pl})
            analysis[label][model]["pairs"] = pairs
            out += [f"Pairwise cells ({model}; A - B, per-prompt paired bootstrap):", "",
                    md([{"pair": p["pair"], "scale": p["scale"], "N": p["N"], "diff [95 % CI]": ci(p["diff"], p["ci_lo"], p["ci_hi"]), "separates": p["separates"]} for p in pairs],
                       ["pair", "scale", "N", "diff [95 % CI]", "separates"]), ""]
            # crossover figure per model
            cs = {}
            for N in NS[1:]:
                g = {k: v["draw"] for k, v in dance_logs(root, "greedy_chunk", N).items()}
                for m in ("lattice_smc_notwist", "lattice_smc"):
                    lg = dance_logs(root, m, N)
                    if not lg:
                        continue
                    for rule in ("draw", "argmax"):
                        cs[(m, rule, N)] = paired({k: v[rule] for k, v in lg.items()}, g, rng)
            series = {(m, rule): [(N, cs[(m, rule, N)]["diff"], cs[(m, rule, N)]["ci_lo"], cs[(m, rule, N)]["ci_hi"]) for N in NS[1:] if (m, rule, N) in cs]
                      for m in ("lattice_smc_notwist", "lattice_smc") for rule in ("draw", "argmax")}
            series = {k: v for k, v in series.items() if v}
            overlay = {("lattice_smc_notwist", "argmax"): [(p["N"], p["diff"], p["ci_lo"], p["ci_hi"]) for p in pairs if p["pair"] == "lattice_smc_notwist-argmax - greedy_chunk" and p["scale"] == "full (regenerated)"]}
            analysis[label][model]["crossover"] = {f"{m}|{rule}|{N}": v for (m, rule, N), v in cs.items()}
            tag_fig = f"x_{rkey}_{'L' if model.startswith('L') else 'E'}"
            crossover_figure(tag_fig, f"{label}, model {model}: lattice minus greedy_chunk", series, f"{'R_BA (additive)' if rkey == 'ba' else 'R_rep'} difference", overlay)
        # orderings at N = 32 (and 8) under both rules, both models
        for N in (32, 8):
            out += [f"### Orderings at N = {N} (reward of the returned sequence: full where a sequence exists, else the logs' additive scale)", ""]
            analysis[label][f"ordering_N{N}"] = {}
            for rule_name, pick, scale in (("draw", lambda m: "argmax" if m in ("bon_argmax", "greedy_chunk") else "draw", "full"),
                                           ("draw", lambda m: "argmax" if m in ("bon_argmax", "greedy_chunk") else "draw", "additive"),
                                           ("argmax", lambda m: "argmax", "additive")):
                ords = {model: ordering(data[model][0], N, pick, scale) for model in models}
                oE, oL = ords["E (4.1 M)"], ords["L (37.6 M)"]
                rho = spearman(oE, oL)
                agree, total, dis = sign_agreement(oE, oL)
                same = [m for m, _, _ in oE] == [m for m, _, _ in oL]
                analysis[label][f"ordering_N{N}"][f"{rule_name}|{scale}"] = {"E": oE, "L": oL, "spearman": rho, "pairwise_sign_agreement": [agree, total], "disagreements": dis, "identical": same}
                note = "; the additive scale is the logs' steering quantity, the only one on which every argmax row exists" if scale == "additive" else ""
                out += [f"**{rule_name} rule, {scale} scale** (bon_argmax and greedy_chunk are argmax by definition{note}):", "",
                        f"- E (4.1 M): " + " > ".join(f"{m} {v:.4f}" for m, v, s in oE),
                        f"- L (37.6 M): " + " > ".join(f"{m} {v:.4f}" for m, v, s in oL),
                        f"- identical order: **{'yes' if same else 'no'}**; Spearman rho {rho:.2f}; pairwise sign agreement {agree} / {total}" + (f"; disagreements: {', '.join(dis)}" if dis else ""), ""]
    json.dump(analysis, open(os.path.join(OUT, "x_analysis.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "x_tables.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
