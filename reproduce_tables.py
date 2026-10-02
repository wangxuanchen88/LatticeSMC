"""Reproduce the paper's Tables 1, 3, 7, 8, 9 and 12 from the stored analysis JSONs under analysis_json/, print each table as a
LaTeX body and as plain text, and compare every printed value with the manuscript. No sample, checkpoint or GPU is read.

  python reproduce_tables.py                       # all six tables, checked against paper_tables.tex (verbatim manuscript excerpts)
  python reproduce_tables.py --tex path/to/main.tex  # check against the full manuscript instead
  python reproduce_tables.py --tables 1,3,7,8,9,12   # a subset

Table numbers follow the manuscript source (NOTES.md numbering): 1 = tab:main, 3 = tab:horizon, 7 = tab:prunesweep,
8 = tab:k6, 9 = tab:k8, 12 = tab:intervals (numbered 11 in the compiled PDF).

Check rule: a manuscript value printed with d decimals matches if |stored - printed| <= 0.5 * 10^-d, or if it equals the
half-up rounding of the stored value first rounded to d + 1 or d + 2 decimals (a printed intermediate rounded again; reported
as a note). Anything else is a mismatch.
Exit status: 0 every compared value matches; bit 1 (value 1) set if any value differs by more than rounding; bit 2 (value 2)
set if a required input file is missing (the table is then skipped and reported).
"""
import argparse
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "analysis_json")
GB, GR = "R_BA alpha 0.02", "R_rep alpha 0.02"


class MissingInput(Exception):
    pass


def J(p):
    fp = os.path.join(A, p)
    if not os.path.exists(fp):
        raise MissingInput(fp)
    return json.load(open(fp))


# ----------------------------------------------------------------------------------------------------- stored-value access
def drow(U, grid, method, rule, N):
    return next(x for x in U["dance_rows"] if x["grid"] == grid and x["method"] == method and x["rule"] == rule and x["N"] == N)


def dfull(U, grid, method, rule, N):
    r = drow(U, grid, method, rule, N)
    v = r["reward_full"]
    return v if v == v else r["reward_additive"]  # NaN full reward (argmax rows never regenerated) -> logged additive value


def dadd(U, grid, method, rule, N):
    return drow(U, grid, method, rule, N)["reward_additive"]


def mrow(U, method, rule, N):
    return next(x for x in U["music_rows"] if x["method"] == method and x["rule"] == rule and x["N"] == N)


def m4row(M4, method, rule, N):
    name = method + ("-argmax" if rule == "argmax" and method not in ("bon_argmax", "greedy_chunk") else "")
    return next(x for x in M4["rows"] if x["method"] == name and x["N"] == N)


def ppair(ana, A_, B_, N):
    """Phase-analysis pairwise_N8_N32 entries hold A - B; the sign is flipped when the stored order is B, A."""
    for p in ana["pairwise_N8_N32"]:
        if p["A"] == A_ and p["B"] == B_ and p["N"] == N:
            return p["diff_mean"], p["ci_lo"], p["ci_hi"]
        if p["A"] == B_ and p["B"] == A_ and p["N"] == N:
            return -p["diff_mean"], -p["ci_hi"], -p["ci_lo"]
    raise KeyError((A_, B_, N))


def lpair(rows, A_, B_, N, **keys):
    p = next(p for p in rows if p["A"] == A_ and p["B"] == B_ and p["N"] == N and all(p.get(k) == v for k, v in keys.items()))
    return p["diff"], p["ci_lo"], p["ci_hi"]


# --------------------------------------------------------------------------------------------------------- row rendering
def F(v, fmt="3"):
    return (float(v), fmt)


def render_num(v, fmt):
    if fmt == "0":
        return str(int(round(v)))
    if fmt.startswith("+"):
        return f"{v:+.{int(fmt[1:])}f}"
    return f"{v:.{int(fmt)}f}"


class Row:
    """A table row as literal LaTeX pieces and (value, format) numbers, so the exact stored value is kept for the check."""

    def __init__(self, *parts):
        self.parts = parts

    def tex(self):
        return "".join(p if isinstance(p, str) else render_num(*p) for p in self.parts)

    def values(self):
        out = []
        for p in self.parts:
            if isinstance(p, str):
                out += [(float(t), False) for t in number_tokens(p)]
            else:
                out.append((p[0], True))
        return out

    def plain(self):
        s = self.tex()
        s = re.sub(r"\\multicolumn\{\d+\}\{[a-z]\}\{([^}]*)\}", r"\1", s)
        s = s.replace("\\\\", "").replace("$", "").replace("\\ ", " ").replace("--", "-").replace("\\method{}", "LatticeSMC")
        s = re.sub(r"\^\{\\star\}", "*", s)
        s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)
        s = re.sub(r"\\(emph|textbf|underline|mathrm)\{([^}]*)\}", r"\2", s)
        s = re.sub(r"\\[A-Za-z]+", "", s)
        s = re.sub(r"[{}]", "", s)
        return " | ".join(c.strip() for c in s.split("&"))


def number_tokens(s):
    s = s.replace("--", " to ").replace("{,}", "")
    s = re.sub(r"\\(textbf|underline|mathbf|emph|method|multicolumn|mathrm|ref|label)\b", " ", s)
    s = s.replace("$", " ")
    s = re.sub(r"\^\{[^}]*\}", " ", s)
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    s = re.sub(r"[{}]", " ", s)
    return re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s)


def ci(m, lo, hi, d="3"):
    return ["$", F(m, d), "\\ [", F(lo, d), ", ", F(hi, d), "]$"]


def pm(v, lo, hi, star, d="+3"):
    return ["$", F(v, d), "\\ [", F(lo, d), ", ", F(hi, d), "]$" + ("$^{*}$" if star else "")]


def star(lo, hi):
    return lo > 0 or hi < 0


# ---------------------------------------------------------------------------------------------------------------- tables
def table_main():
    U, M3, M4 = J("u/u_analysis.json"), J("m3/m3_analysis.json"), J("m4/m4_analysis.json")
    P2A, P3A, FK = J("phase2b_a002_analysis.json"), J("phase3_rep_a002_analysis.json"), J("verify_fkargmax_summary.json")
    cb = next(c for c in P2A["curve"] if c["method"] == "base")
    rb = next(h for h in P3A["heldout"]["32"] if h["method"] == "base")["rep_full"]
    m3b, m4b = M3["rows"][0], M4["rows"][0]
    assert m3b["method"] == "base" and m4b["method"] == "base"
    base = [(dfull(U, GB, "base", "-", 1), cb["ci_lo"], cb["ci_hi"]), (dfull(U, GR, "base", "-", 1), rb["ci_lo"], rb["ci_hi"]),
            (m3b["R_mean"], m3b["ci_lo"], m3b["ci_hi"]), (m4b["R_mean"], m4b["ci_lo"], m4b["ci_hi"])]
    r = ["Base ($N = 1$)"]
    for m, lo, hi in base:
        r += [" & \\multicolumn{2}{c}{", F(m), " [", F(lo), ", ", F(hi), "]}"]
    rows = [Row(*r, " \\\\")]

    def cells(vals):
        out = []
        for v in vals:
            out += [" & ", "--" if v is None else (F(v) if not isinstance(v, tuple) else v)]
        return out
    rows.append(Row("Best-of-N", *cells([dfull(U, GB, "bon_is", "draw", 32), dfull(U, GB, "bon_argmax", "argmax", 32), dfull(U, GR, "bon_is", "draw", 32), dfull(U, GR, "bon_argmax", "argmax", 32),
                                          mrow(U, "bon_is", "draw", 32)["reward"], mrow(U, "bon_argmax", "argmax", 32)["reward"], m4row(M4, "bon_is", "draw", 32)["R_mean"], m4row(M4, "bon_argmax", "argmax", 32)["R_mean"]]), " \\\\"))
    rows.append(Row("Chunk pruning (top-$M$)", *cells([None, dfull(U, GB, "greedy_chunk", "argmax", 32), None, dfull(U, GR, "greedy_chunk", "argmax", 32), None, mrow(U, "greedy_chunk", "argmax", 32)["reward"], None, m4row(M4, "greedy_chunk", "argmax", 32)["R_mean"]]), " \\\\"))
    rows.append(Row("Twisted SMC, learned twist", " & ", F(dfull(U, GB, "lattice_smc", "draw", 32)), " & ", F(dadd(U, GB, "lattice_smc", "argmax", 32)), "$^{a}$ & ", F(dfull(U, GR, "lattice_smc", "draw", 32)), " & ", F(dadd(U, GR, "lattice_smc", "argmax", 32)), " & -- & -- & -- & -- \\\\"))
    rows.append(Row("\\method{}, dense (FK steering placement)", " & ", F(dfull(U, GB, "fk_noise", "draw", 32)), "$^{\\star}$ & ", F(FK["4M"]["full_mean"]), "$^{\\star}$ & ", F(dfull(U, GR, "fk_noise", "draw", 32)), "$^{\\star}$ & ", F(dadd(U, GR, "fk_noise", "argmax", 32)), "$^{\\star}$",
                    *cells([mrow(U, "fk_noise", "draw", 32)["reward"], mrow(U, "fk_noise", "argmax", 32)["reward"], m4row(M4, "fk_noise", "draw", 32)["R_mean"], m4row(M4, "fk_noise", "argmax", 32)["R_mean"]]), " \\\\"))
    rows.append(Row("\\method{}, boundary", *cells([dfull(U, GB, "lattice_smc_notwist", "draw", 32), dfull(U, GB, "lattice_smc_notwist", "argmax", 32), dfull(U, GR, "lattice_smc_notwist", "draw", 32), dfull(U, GR, "lattice_smc_notwist", "argmax", 32)]),
                    " & ", F(mrow(U, "lattice_smc_prefix_b1", "draw", 32)["reward"]), "$^{\\star}$ & ", F(mrow(U, "lattice_smc_prefix_b1", "argmax", 32)["reward"]), "$^{\\star}$ & ", F(m4row(M4, "lattice_smc_prefix_b1", "draw", 32)["R_mean"]), "$^{\\star}$ & ", F(m4row(M4, "lattice_smc_prefix_b1", "argmax", 32)["R_mean"]), "$^{\\star}$ \\\\"))
    head = "& \\multicolumn{2}{c}{Beat alignment} & \\multicolumn{2}{c}{Repetition} & \\multicolumn{2}{c}{Prompt adherence} & \\multicolumn{2}{c}{Motif recurrence} \\\\\n& draw & argmax & draw & argmax & draw & argmax & draw & argmax \\\\"
    note = ("sources: analysis_json/u/u_analysis.json (dance_rows, music_rows; full reward of the returned sequence, argmax rows regenerated), m3/m3_analysis.json and m4/m4_analysis.json (music base rows and motif rows), "
            "phase2b_a002_analysis.json and phase3_rep_a002_analysis.json (dance base intervals), verify_fkargmax_summary.json (dense-schedule argmax on beat alignment, full reward of the regenerated sequence). "
            "Bold / underline ranking marks of the manuscript are not reproduced.")
    return head, rows, None, note


def table_horizon():
    U, FK, K6, K8 = J("u/u_analysis.json"), J("verify_fkargmax_summary.json"), J("r1/dance_k6.json"), J("r1/dance_k8.json")

    def kv(K, rkey, method):
        return next(r for r in K["rewards"][rkey]["rows"] if r["method"] == method and r["N"] == 32)["reward_full"]
    k4 = {"ba": {"bon": dfull(U, GB, "bon_argmax", "argmax", 32), "prune": dfull(U, GB, "greedy_chunk", "argmax", 32), "boundary": dfull(U, GB, "lattice_smc_notwist", "argmax", 32), "dense": FK["4M"]["full_mean"]},
          "rep": {"bon": dfull(U, GR, "bon_argmax", "argmax", 32), "prune": dfull(U, GR, "greedy_chunk", "argmax", 32), "boundary": dfull(U, GR, "lattice_smc_notwist", "argmax", 32), "dense": dadd(U, GR, "fk_noise", "argmax", 32)}}
    names = {"bon": "best-of-N (argmax)", "prune": "chunk pruning", "boundary": "boundary (argmax)", "dense": "dense (argmax)"}
    rows = []
    for label, key in (("Best-of-N", "bon"), ("Chunk pruning ($M = N/4$)", "prune"), ("\\method{}, boundary", "boundary"), ("\\method{}, dense", "dense")):
        parts = [label]
        for rkey in ("ba", "rep"):
            parts += [" & ", F(k4[rkey][key]), " & ", F(kv(K6, rkey, names[key])), " & ", F(kv(K8, rkey, names[key]))]
        rows.append(Row(*parts, " \\\\"))
    rows.append(Row("\\method{}, boundary / dense, $\\alpha = 0.01$ & -- & ", F(kv(K6, "ba", "boundary (argmax), alpha 0.01")), " / ", F(kv(K6, "ba", "dense (argmax), alpha 0.01")), " & ",
                    F(kv(K8, "ba", "boundary (argmax), alpha 0.01")), " / ", F(kv(K8, "ba", "dense (argmax), alpha 0.01")), " & -- & -- & -- \\\\"))
    head = "& \\multicolumn{3}{c}{Beat alignment} & \\multicolumn{3}{c}{Repetition} \\\\\n$K =$ & 4 & 6 & 8 & 4 & 6 & 8 \\\\"
    return head, rows, None, "sources: K = 4 as Table 1; K = 6 and 8 from analysis_json/r1/dance_k6.json and dance_k8.json (rows at N = 32, argmax return)."


def table_prunesweep():
    PS = J("r1/pruning_sweep.json")
    rows = []
    M = {1: "chunk pruning M=1", 8: "chunk pruning M=8 (N/4, stored)", 16: "chunk pruning M=16 (N/2)"}
    B = "boundary schedule argmax (stored)"
    for i, reward in enumerate(("beat alignment", "repetition", "prompt adherence", "motif recurrence")):
        if i:
            rows.append("\\midrule")
        for m in (1, 8, 16):
            r = next(x for x in PS["rows"] if x["reward"] == reward and x["method"] == M[m])
            mean, (lo, hi) = r["reward_full"]["mean"], r["reward_full"]["ci"]
            if reward in ("beat alignment", "repetition"):
                held = ["PFC ", F(r["pfc"], "2"), ", $W_1$ ", F(r["realism_w1"], "4")]
            else:
                held = ["tempo ", F(r["tempo_reward"], "3"), ", 8 kHz ", F(r["frac_energy_above_8k"], "4"), ", div.\\ ", F(r["seed_diversity"], "3")]
            parts = [reward.capitalize() if m == 1 else "", " & ", str(m), " & ", *ci(mean, lo, hi), " & ", *held, " & "]
            if m != 8:
                d, plo, phi = lpair(PS["pairs"], M[m], M[8], 32, reward=reward)
                parts += ["$", F(d, "+3"), "$" + ("$^{*}$" if star(plo, phi) else "")]
            d, plo, phi = lpair(PS["pairs"], M[m], B, 32, reward=reward)
            parts += [" & $", F(d, "+3"), "$" + ("$^{*}$" if star(plo, phi) else ""), " \\\\"]
            rows.append(Row(*parts))
    head = "Reward & $M$ & Reward [95\\% CI] & Held out & $-$ ($M = N/4$) & $-$ boundary argmax \\\\"
    return head, rows, None, "source: analysis_json/r1/pruning_sweep.json (rows and paired differences; PFC is x 10^4 as stored)."


def table_k(K):
    D = J(f"r1/dance_k{K}.json")

    def row(rkey, method, N):
        return next(r for r in D["rewards"][rkey]["rows"] if r["method"] == method and r["N"] == N and r.get("alpha", 0.02) == (0.01 if "0.01" in method else 0.02))

    def held(r):
        return [F(r["pfc"], "2"), " & ", F(r["realism_w1"], "4"), " & ", F(r["diversity_root_rel"], "3")]

    def held2(a, b):
        return [F(a["pfc"], "2"), " / ", F(b["pfc"], "2"), " & ", F(a["realism_w1"], "4"), " / ", F(b["realism_w1"], "4"), " & ", F(a["diversity_root_rel"], "3"), " / ", F(b["diversity_root_rel"], "3")]

    def ess(r):
        e = r["ess_over_N_before_boundary"]
        lo, hi = min(e.values()), max(e.values())
        return [F(lo, "2")] if abs(hi - lo) < 5e-3 and r["method"] == "chunk pruning" else [F(lo, "2"), "--", F(hi, "2")]
    rows = []
    for rkey, label in (("ba", "Beat alignment"), ("rep", "Repetition")):
        if rkey == "rep":
            rows.append("\\midrule")
        rows.append(Row("\\multicolumn{8}{l}{\\emph{" + label + "}, $\\alpha = 0.02$} \\\\"))
        b = row(rkey, "base", 1)
        rows.append(Row("Base & 1 & ", *ci(b["reward_full"], *b["ci"]), " & & & ", *held(b), " \\\\"))
        for N in (8, 32):
            d, a = row(rkey, "best-of-N (draw)", N), row(rkey, "best-of-N (argmax)", N)
            rows.append(Row(f"Best-of-N (draw / argmax) & {N} & ", *ci(d["reward_full"], *d["ci"]), " / ", *ci(a["reward_full"], *a["ci"]), " & & & ", *held2(d, a), " \\\\"))
            p = row(rkey, "chunk pruning", N)
            rows.append(Row(f"Chunk pruning & {N} & ", *ci(p["reward_full"], *p["ci"]), " & ", F(p["frac_boundaries_resampled"], "2"), " & ", *ess(p), " & ", *held(p), " \\\\"))
            d, a = row(rkey, "boundary (draw)", N), row(rkey, "boundary (argmax)", N)
            rows.append(Row(f"\\method{{}}, boundary (draw / argmax) & {N} & ", *ci(d["reward_full"], *d["ci"]), " / ", *ci(a["reward_full"], *a["ci"]), " & ", F(d["frac_boundaries_resampled"], "2"), " & ", *ess(d), " & ", *held2(d, a), " \\\\"))
            d, a = row(rkey, "dense (draw)", N), row(rkey, "dense (argmax)", N)
            rows.append(Row(f"\\method{{}}, dense (draw / argmax) & {N} & ", *ci(d["reward_full"], *d["ci"]), " / ", *ci(a["reward_full"], *a["ci"]), " & & & ", *held2(d, a), " \\\\"))
        if rkey == "ba":
            rows.append(Row("\\multicolumn{8}{l}{\\emph{Beat alignment}, $\\alpha = 0.01$} \\\\"))
            d, a = row("ba", "boundary (draw), alpha 0.01", 32), row("ba", "boundary (argmax), alpha 0.01", 32)
            rows.append(Row("\\method{}, boundary (draw / argmax) & 32 & ", *ci(d["reward_full"], *d["ci"]), " / ", *ci(a["reward_full"], *a["ci"]), " & ", F(d["frac_boundaries_resampled"], "2"), " & ", *ess(d), " & ", *held2(d, a), " \\\\"))
            d, a = row("ba", "dense (draw), alpha 0.01", 32), row("ba", "dense (argmax), alpha 0.01", 32)
            rows.append(Row("\\method{}, dense (draw / argmax) & 32 & ", *ci(d["reward_full"], *d["ci"]), " / ", *ci(a["reward_full"], *a["ci"]), " & & & ", *held2(d, a), " \\\\"))
    # caption numbers: base held-out intervals (beat-alignment conditions) and the paired differences at N = 32
    hb = D["rewards"]["ba"]["base_heldout_ci"]
    cap = ["Base intervals on these conditions: PFC $[", F(hb["pfc"]["ci"][0], "2"), ", ", F(hb["pfc"]["ci"][1], "2"), "]$, $W_1$ $[", F(hb["realism_w1"]["ci"][0], "4"), ", ", F(hb["realism_w1"]["ci"][1], "4"), "]$. Paired differences at $N = 32$, beat alignment: "]
    P = D["rewards"]["ba"]["pairs_N32"]
    for lab, a_, b_ in (("boundary argmax $-$ pruning", "boundary (argmax)", "chunk pruning"), ("dense argmax $-$ pruning", "dense (argmax)", "chunk pruning"),
                        ("boundary argmax, $\\alpha = 0.01$, $-$ pruning", "boundary (argmax), alpha 0.01", "chunk pruning"), ("dense argmax, $\\alpha = 0.01$, $-$ pruning", "dense (argmax), alpha 0.01", "chunk pruning"),
                        ("boundary argmax, $\\alpha = 0.01$, $-$ best-of-N argmax", "boundary (argmax), alpha 0.01", "best-of-N (argmax)")):
        d, lo, hi = lpair(P, a_, b_, 32)
        cap += [lab + " $", F(d, "+3"), "\\ [", F(lo, "+3"), ", ", F(hi, "+3"), "]$; "]
    cap[-1] = cap[-1][:-2] + ". Repetition: "
    P = D["rewards"]["rep"]["pairs_N32"]
    for lab, a_, b_ in (("boundary argmax $-$ pruning", "boundary (argmax)", "chunk pruning"), ("boundary argmax $-$ best-of-N argmax", "boundary (argmax)", "best-of-N (argmax)"), ("dense argmax $-$ boundary argmax", "dense (argmax)", "boundary (argmax)")):
        d, lo, hi = lpair(P, a_, b_, 32)
        cap += [lab + " $", F(d, "+3"), "\\ [", F(lo, "+3"), ", ", F(hi, "+3"), "]$; "]
    cap[-1] = cap[-1][:-2] + "."
    head = "Method & $N$ & Reward [95\\% CI] & fired & $\\mathrm{ESS}/N$ & PFC & $W_1$ & diversity \\\\"
    return head, rows, Row(*cap), f"source: analysis_json/r1/dance_k{K}.json (rows, pairs_N32, base_heldout_ci)."


def table_intervals():
    U, M3, M4, SP = J("u/u_analysis.json"), J("m3/m3_analysis.json"), J("m4/m4_analysis.json"), J("r1/schedule_pairs.json")
    P2A, P3A = J("phase2b_a002_analysis.json"), J("phase3_rep_a002_analysis.json")
    up = lambda grid, a, b: lpair([p for p in U["dance_pairs"] if p["grid"] == grid], a, b, 32)  # noqa: E731
    mp = lambda a, b: lpair(U["music_pairs"], a, b, 32)  # noqa: E731
    m3p = lambda a, b: lpair(M3["pairs"], a, b, 32)  # noqa: E731
    m4p = lambda a, b: lpair(M4["pairs"], a, b, 32)  # noqa: E731
    sp = lambda reward, model, a, b: lpair(SP["rows"], a, b, 32, reward=reward, model=model)  # noqa: E731
    R = {
        "boundary $-$ best-of-N, draw": [ppair(P2A, "lattice_smc_notwist", "bon_is", 32), ppair(P3A, "lattice_smc_notwist", "bon_is", 32), m3p("lattice_smc_prefix_b1", "bon_is"), m4p("lattice_smc_prefix_b1", "bon_is")],
        "boundary $-$ chunk pruning, argmax": [up(GB, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk"), up(GR, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk"), mp("lattice_smc_prefix_b1-argmax", "greedy_chunk"), m4p("lattice_smc_prefix_b1-argmax", "greedy_chunk")],
        "boundary $-$ chunk pruning, draw": [None, ppair(P3A, "lattice_smc_notwist", "greedy_chunk", 32), None, None],
        "dense $-$ chunk pruning, argmax": [sp("beat alignment", "4M", "dense argmax", "chunk pruning"), sp("repetition", "4M", "dense argmax", "chunk pruning"), None, None],
        "dense $-$ boundary, argmax": [sp("beat alignment", "4M", "dense argmax", "boundary argmax"), sp("repetition", "4M", "dense argmax", "boundary argmax"), sp("prompt adherence", "music", "dense argmax", "boundary argmax"), sp("motif recurrence", "music", "dense argmax", "boundary argmax")],
        "dense $-$ boundary, draw": [None, None, sp("prompt adherence", "music", "dense draw", "boundary draw"), sp("motif recurrence", "music", "dense draw", "boundary draw")],
        "dense $-$ chunk pruning, argmax, 38M": [sp("beat alignment", "38M", "dense argmax", "chunk pruning"), sp("repetition", "38M", "dense argmax", "chunk pruning"), None, None],
        "dense $-$ boundary, argmax, 38M": [sp("beat alignment", "38M", "dense argmax", "boundary argmax"), sp("repetition", "38M", "dense argmax", "boundary argmax"), None, None],
        "argmax $-$ draw, boundary": [up(GB, "lattice_smc_notwist-argmax", "lattice_smc_notwist (draw)"), up(GR, "lattice_smc_notwist-argmax", "lattice_smc_notwist (draw)"), mp("lattice_smc_prefix_b1-argmax", "lattice_smc_prefix_b1 (draw)"), m4p("lattice_smc_prefix_b1-argmax", "lattice_smc_prefix_b1")],
    }
    rows = []
    for lab, cells in R.items():
        parts = [lab]
        for c in cells:
            parts += [" & "] + (["--"] if c is None else ["$", F(c[0], "+3"), "\\ [", F(c[1], "+3"), ", ", F(c[2], "+3"), "]$"])
        rows.append(Row(*parts, " \\\\"))
    head = "Difference & Beat alignment & Repetition & Prompt adherence & Motif recurrence \\\\"
    return head, rows, None, ("sources: phase2b_a002 / phase3_rep_a002 pairwise (draw rows), u_analysis dance_pairs and music_pairs (argmax rows), m3 / m4 pairs (music), "
                              "analysis_json/r1/schedule_pairs.json (dense-schedule rows).")


TABLES = {"1": ("tab:main", table_main), "3": ("tab:horizon", table_horizon), "7": ("tab:prunesweep", table_prunesweep), "8": ("tab:k6", lambda: table_k(6)),
          "9": ("tab:k8", lambda: table_k(8)), "12": ("tab:intervals", table_intervals)}
PDF_NUMBER = {"12": 11}


# ------------------------------------------------------------------------------------------------------ manuscript side
def balanced(s, start):
    """Return the text inside the brace group starting at s[start] == '{'."""
    depth, i = 0, start
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i]
        i += 1
    raise ValueError("unbalanced braces")


def paper_table(tex, label):
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return None
    b = tex.rfind("\\begin{table}", 0, i)
    e = tex.find("\\end{table}", i)
    block = tex[b:e]
    ci_ = block.find("\\caption")
    caption = balanced(block, block.find("{", ci_))
    body = block[block.find("\\midrule"):block.find("\\bottomrule")]
    rows = [ln.strip() for ln in body.splitlines() if "\\\\" in ln and ln.strip() != "\\midrule"]
    return caption, rows


def round_half_up(v, d):
    return float(Decimal(repr(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def classify(pt, v):
    """'match': within half a unit of the last printed digit; 'rounding': within one unit and equal to the half-up rounding
    of the stored value first rounded to one or two more decimals (a printed intermediate rounded again); else 'mismatch'."""
    d = len(pt.split(".")[1]) if "." in pt else 0
    diff = abs(float(pt) - v)
    if diff <= 0.5 * 10 ** (-d) + 1e-9:
        return "match", d
    if diff < 10 ** (-d):
        for dd in (d + 1, d + 2):
            if abs(round_half_up(round_half_up(v, dd), d) - float(pt)) < 1e-9:
                return "rounding", d
    return "mismatch", d


def compare_tokens(printed, mine, where, out, notes):
    """printed: numeric strings from the manuscript; mine: list of (value, exact). Returns the number of mismatches."""
    bad = 0
    if len(printed) != len(mine):
        out.append(f"  {where}: token count differs (manuscript {len(printed)}, stored {len(mine)}); manuscript tokens {printed}; stored {[round(v, 4) for v, _ in mine]}")
        return 1
    for k, (pt, (v, _)) in enumerate(zip(printed, mine)):
        c, d = classify(pt, v)
        if c == "mismatch":
            bad += 1
            out.append(f"  {where}, value {k + 1}: manuscript {pt} vs stored {v:.{d + 2}f} (differs by more than rounding at {d} decimals)")
        elif c == "rounding":
            notes.append(f"  {where}, value {k + 1}: manuscript {pt} vs stored {v:.{d + 2}f} (rounding of a {d + 1}- or {d + 2}-decimal intermediate)")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(HERE, "paper_tables.tex"), help="manuscript source or the verbatim table excerpts to check against")
    ap.add_argument("--tables", default="1,3,7,8,9,12")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    tex = None if a.no_check else open(a.tex).read()
    status, n_checked, n_bad, missing = 0, 0, 0, []
    for num in a.tables.split(","):
        num = num.strip()
        label, fn = TABLES[num]
        title = f"Table {num} ({label}" + (f"; Table {PDF_NUMBER[num]} in the compiled PDF" if num in PDF_NUMBER else "") + ")"
        print("=" * 100 + f"\n{title}\n" + "=" * 100)
        try:
            head, rows, caption, note = fn()
        except MissingInput as e:
            print(f"SKIPPED: required input not found: {e}")
            missing.append(f"{title}: {e}")
            status |= 2
            continue
        print("% " + note)
        print(head)
        print("\\midrule")
        for r in rows:
            print(r if isinstance(r, str) else r.tex())
        if caption is not None:
            print("% caption numbers: " + caption.tex())
        print("\n-- plain text --")
        for r in rows:
            if not isinstance(r, str):
                print(r.plain())
        if caption is not None:
            print("caption: " + caption.plain())
        if tex is None:
            print()
            continue
        pt = paper_table(tex, label)
        if pt is None:
            print(f"\n-- check: {label} not found in {a.tex}; not compared")
            continue
        pcap, prows = pt
        mrows = [r for r in rows if not isinstance(r, str)]
        msgs, notes, bad, checked = [], [], 0, 0
        if len(prows) != len(mrows):
            msgs.append(f"  row count differs: manuscript {len(prows)}, stored {len(mrows)}")
            bad += 1
        for i, (pr, mr) in enumerate(zip(prows, mrows)):
            toks = number_tokens(pr)
            checked += len(toks)
            bad += compare_tokens(toks, mr.values(), f"row {i + 1} ({mr.plain().split('|')[0].strip()[:40]})", msgs, notes)
        if caption is not None:
            k = pcap.find("Base intervals")
            ptoks = number_tokens(pcap[k:]) if k >= 0 else []
            where = "caption"
            checked += len(ptoks)
            bad += compare_tokens(ptoks, caption.values(), where, msgs, notes)
        n_checked += checked
        n_bad += bad
        print(f"\n-- check against the manuscript: {checked} values compared, {bad} mismatches beyond rounding, {len(notes)} rounding-only notes")
        for m in msgs:
            print(m)
        for m in notes:
            print("  (note)" + m)
        if bad:
            status |= 1
        print()
    print("=" * 100)
    if tex is not None:
        print(f"TOTAL: {n_checked} values compared, {n_bad} mismatches beyond rounding; {len(missing)} table(s) skipped for missing inputs")
        for m in missing:
            print("  missing: " + m)
    print(f"exit status {status}")
    sys.exit(status)


if __name__ == "__main__":
    main()
