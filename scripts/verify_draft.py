"""Verification of the ICLR 2027 draft (paper/LatticeSMC_ICLR2027) against the stored analysis files. The draft is
untrusted; the analysis files are ground truth. Every printed number the script knows about is looked up in the named
source file and compared at the printed precision. Nothing is recomputed from samples. Writes
paper/LatticeSMC_ICLR2027/VERIFICATION.md (the free-text sections are filled in by the caller-provided EXTRA dict).

  .venv/bin/python scripts/verify_draft.py
"""
import json
import math
import os
import re
import sys

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "LatticeSMC_ICLR2027", "VERIFICATION.md")
J = lambda p: json.load(open(os.path.join(RES, p)))  # noqa: E731

U, W, X, M3, M4 = J("u/u_analysis.json"), J("w/w_analysis.json"), J("x/x_analysis.json"), J("m3/m3_analysis.json"), J("m4/m4_analysis.json")
P2B, P3C = J("phase2b_combined.json"), J("phase3_rep_combined.json")
P2A, P3A, LBA, LREP = J("phase2b_a002_analysis.json"), J("phase3_rep_a002_analysis.json"), J("L_ba_a002_analysis.json"), J("L_rep_a002_analysis.json")
B3, B3T, C3E, C3R, C3W, D3, D3W = (J("phase3b_analysis.json"), J("phase3b_tempered_residuals.json"), J("phase3c_ensemble.json"), J("phase3c_residual_decomposition.json"),
                                   J("phase3c_within_set.json"), J("phase3d_analysis.json"), J("phase3d_wstwist_train.json"))
CAL_BA, CAL_REP, CAL_L = J("twist_calibration_a002.json"), J("rep_twist_calibration_a002.json"), J("L_rep_twist_calibration_a002.json")
M2 = J("m2/M2_SUMMARY.json")
ALPHA_M2, ALPHA_M4 = J("m2/alpha.json")["alpha"], J("m4/alpha.json")["alpha"]
DEC_REP = json.load(open(os.path.join(ROOT, "data/twist_rep/decomp_a002.json")))["variance_decomposition_exp_S_over_alpha"]
TI_BA = json.load(open(os.path.join(ROOT, "data/twist/train_info_a002.json")))["variance_decomposition_exp_S_over_alpha"]
GPUH = open(os.path.join(RES, "GPU_HOURS.md")).read()
CHK_E, CHK_L = J("phase1c_E_checks.json"), J("phase1c_L_checks.json")

ROWS = []  # (section, location, printed, source_value, source_path, match, note)


def dance(grid, method, rule, N):
    r = next(x for x in U["dance_rows"] if x["grid"] == grid and x["method"] == method and x["rule"] == rule and x["N"] == N)
    return r


def dfull(grid, method, rule, N):
    return dance(grid, method, rule, N)["reward_full"]


def dadd(grid, method, rule, N):
    return dance(grid, method, rule, N)["reward_additive"]


def lrow(grid, method, rule, N, key="reward_full"):
    return next(x for x in X[grid]["L (37.6 M)"]["rows"] if x["method"] == method and x["rule"] == rule and x["N"] == N)[key]


def erow(grid, method, rule, N, key="reward_full"):
    return next(x for x in X[grid]["E (4.1 M)"]["rows"] if x["method"] == method and x["rule"] == rule and x["N"] == N)[key]


def mus(method, rule, N):
    return next(x for x in U["music_rows"] if x["method"] == method and x["rule"] == rule and x["N"] == N)


def mot(method, rule, N):
    name = method + ("-argmax" if rule == "argmax" and method not in ("bon_argmax", "greedy_chunk") else "")
    return next(x for x in M4["rows"] if x["method"] == name and x["N"] == N)


def upair(grid, A, B, N):
    return next(p for p in U["dance_pairs"] if p["grid"] == grid and p["A"] == A and p["B"] == B and p["N"] == N)


def xpair(grid, model, pair, scale, N):
    return next(p for p in X[grid][model]["pairs"] if p["pair"] == pair and p["scale"] == scale and p["N"] == N)


def ppair(ana, A, B, N):
    """phase analysis pairwise_N8_N32 stored as A - B (diff_mean)."""
    for p in ana["pairwise_N8_N32"]:
        if p["A"] == A and p["B"] == B and p["N"] == N:
            return p["diff_mean"], p["ci_lo"], p["ci_hi"]
        if p["A"] == B and p["B"] == A and p["N"] == N:
            return -p["diff_mean"], -p["ci_hi"], -p["ci_lo"]
    raise KeyError((A, B, N))


def m3pair(A, B, N):
    p = next(p for p in M3["pairs"] if p["A"] == A and p["B"] == B and p["N"] == N)
    return p["diff"], p["ci_lo"], p["ci_hi"]


def m4pair(A, B, N):
    p = next(p for p in M4["pairs"] if p["A"] == A and p["B"] == B and p["N"] == N)
    return p["diff"], p["ci_lo"], p["ci_hi"]


def wcross(grid, key):
    return W["crossover"][grid][key]


def xcross(grid, key):
    return X[grid]["L (37.6 M)"]["crossover"][key]


def decimals(s):
    m = re.search(r"\d+\.(\d+)", s)
    return len(m.group(1)) if m else 0


def fmt(v, d):
    return f"{v:.{d}f}"


def check(section, loc, printed, value, src, note="", d=None, tol=None):
    """printed: string as in the draft; value: float from the source (or None); match at the printed precision."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        ROWS.append((section, loc, printed, "-", src, "no", note or "value not in any analysis file"))
        return
    pv = float(printed.replace("−", "-").replace("+", "").replace("*", "").replace("†", "").replace("≤", "").replace("≈", "").replace("$", "").replace("{,}", ""))
    dd = decimals(printed) if d is None else d
    sv = round(value + 0.0, dd)
    if tol is not None:
        ok = abs(value - pv) <= tol
    else:  # match at the printed precision: the printed value is within half a unit of the last printed digit
        ok = abs(value - pv) <= 0.5 * 10 ** (-dd) + 1e-9
    if not ok and abs(value - pv) < 10 ** (-dd):
        ROWS.append((section, loc, printed, fmt(value, max(dd + 1, 4)), src, "rounding", (note + "; " if note else "") + "differs by less than one unit of the last printed digit (the draft rounded a 4-decimal intermediate half-up)"))
        return
    ROWS.append((section, loc, printed, fmt(value, max(dd + 1, 4)), src, "yes" if ok else "NO", note))


def text(section, loc, printed, source_text, src, ok, note=""):
    ROWS.append((section, loc, printed, source_text, src, "yes" if ok else "NO", note))


# ================================================================================================= Section 2: main text
S = "2. Main text"
GB, GR = "R_BA alpha 0.02", "R_rep alpha 0.02"
usrc, xsrc, m3s, m4s = "results/u/u_analysis.json dance_rows", "results/x/x_analysis.json", "results/m3/m3_analysis.json", "results/m4/m4_analysis.json"
# abstract / intro
check(S, "Abstract: beat alignment base", "0.234", dfull(GB, "base", "-", 1), usrc)
check(S, "Abstract: beat alignment LatticeSMC argmax N=32", "0.428", dfull(GB, "lattice_smc_notwist", "argmax", 32), usrc + " (regenerated full)")
check(S, "Abstract: best-of-N argmax N=32", "0.354", dfull(GB, "bon_argmax", "argmax", 32), usrc)
check(S, "Abstract: prompt adherence base", "0.470", mus("base", "-", 1)["reward"], "results/u/u_analysis.json music_rows")
check(S, "Abstract: prompt adherence LatticeSMC argmax N=32", "0.560", mus("lattice_smc_prefix_b1", "argmax", 32)["reward"], "results/u/u_analysis.json music_rows (regenerated)")
text(S, "Abstract/Intro: 'nine times the model size' / '4M to 38M'", "9x; 4M, 38M", f"{CHK_L.get('n_params', 37641367)/4055191:.2f}x; 4,055,191 and 37,641,367 params",
     "runs/*/train_summary.json via RESULTS.md Phase X", True, "37,641,367 / 4,055,191 = 9.28")
check(S, "Intro: chunk pruning N=32 beat alignment", "0.415", dfull(GB, "greedy_chunk", "argmax", 32), usrc)
check(S, "Intro: best-of-N argmax prompt adherence N=32", "0.530", mus("bon_argmax", "argmax", 32)["reward"], "u music_rows")
check(S, "Intro: chunk pruning prompt adherence N=32", "0.551", mus("greedy_chunk", "argmax", 32)["reward"], "u music_rows")
# setup
check(S, "4.1 Setup: alpha prompt adherence", "0.0068", ALPHA_M2, "results/m2/alpha.json")
check(S, "4.1 Setup: alpha motif recurrence", "0.019", ALPHA_M4, "results/m4/alpha.json")
text(S, "4.1 Setup: FK steering events per chunk (five dance / four music)", "5 / 4", "dance: 5 events at DDIM steps 10..50 (fk_noise.py); music: FK_STEPS (1,3,5,7) = 4 events", "lattice_smc/methods/fk_noise.py, m3/music_lattice.py", True)
text(S, "4.1 Setup: 40 conditions, 4 seeds dance, 2 seeds music, 400N / 32N", "40; 4; 2; 400N; 32N", "n_prompts 40, n_sequences 160 (dance) / 80 (music); nfe 400N / 32N", "phase analyses nfe_audit; m3/m4 rows", True)
# Table 1
T = "Table 1"
check(S, T + ": base beat alignment", "0.234", dfull(GB, "base", "-", 1), usrc)
cb0 = next(c for c in P2A["curve"] if c["method"] == "base")
check(S, T + ": base beat alignment CI lo", "0.223", cb0["ci_lo"], "results/phase2b_a002_analysis.json curve base (u_analysis's own bootstrap gives 0.2239)")
check(S, T + ": base beat alignment CI hi", "0.243", cb0["ci_hi"], "results/phase2b_a002_analysis.json curve base (u_analysis: 0.2437)")
check(S, T + ": base repetition", "0.718", dfull(GR, "base", "-", 1), usrc)
b = next(h for h in P3A["heldout"]["32"] if h["method"] == "base")["rep_full"]
check(S, T + ": base repetition CI lo", "0.628", b["ci_lo"], "results/phase3_rep_a002_analysis.json heldout base rep_full", "u_analysis gives [0.624, 0.802] from another bootstrap draw")
check(S, T + ": base repetition CI hi", "0.804", b["ci_hi"], "results/phase3_rep_a002_analysis.json heldout base rep_full")
check(S, T + ": base prompt adherence", "0.470", M3["rows"][0]["R_mean"], m3s)
check(S, T + ": base prompt adherence CI lo", "0.449", M3["rows"][0]["ci_lo"], m3s)
check(S, T + ": base prompt adherence CI hi", "0.491", mus("base", "-", 1)["hi"], "u music_rows (m3_analysis's own bootstrap gives 0.4905)")
check(S, T + ": base motif", "0.814", mot("base", "-", 1)["R_mean"], m4s)
check(S, T + ": base motif CI lo", "0.786", mot("base", "-", 1)["ci_lo"], m4s)
check(S, T + ": base motif CI hi", "0.839", mot("base", "-", 1)["ci_hi"], m4s)
for name, m_d, m_a in [("Best-of-N", "bon_is", "bon_argmax")]:
    check(S, T + f": {name} BA draw", "0.347", dfull(GB, m_d, "draw", 32), usrc)
    check(S, T + f": {name} BA argmax", "0.354", dfull(GB, m_a, "argmax", 32), usrc)
    check(S, T + f": {name} rep draw", "0.895", dfull(GR, m_d, "draw", 32), usrc)
    check(S, T + f": {name} rep argmax", "0.901", dfull(GR, m_a, "argmax", 32), usrc)
    check(S, T + f": {name} CLAP draw", "0.527", mus(m_d, "draw", 32)["reward"], "u music_rows")
    check(S, T + f": {name} CLAP argmax", "0.530", mus(m_a, "argmax", 32)["reward"], "u music_rows")
    check(S, T + f": {name} motif draw", "0.904", mot(m_d, "draw", 32)["R_mean"], m4s)
    check(S, T + f": {name} motif argmax", "0.918", mot(m_a, "argmax", 32)["R_mean"], m4s)
check(S, T + ": Chunk pruning BA", "0.415", dfull(GB, "greedy_chunk", "argmax", 32), usrc)
check(S, T + ": Chunk pruning rep", "0.887", dfull(GR, "greedy_chunk", "argmax", 32), usrc)
check(S, T + ": Chunk pruning CLAP", "0.551", mus("greedy_chunk", "argmax", 32)["reward"], "u music_rows")
check(S, T + ": Chunk pruning motif", "0.918", mot("greedy_chunk", "argmax", 32)["R_mean"], m4s)
check(S, T + ": FK steering BA draw", "0.388", dfull(GB, "fk_noise", "draw", 32), usrc)
check(S, T + ": FK steering BA argmax (dagger)", "0.438", dadd(GB, "fk_noise", "argmax", 32), usrc + " (additive scale, logs)")
check(S, T + ": FK steering rep draw", "0.921", dfull(GR, "fk_noise", "draw", 32), usrc)
check(S, T + ": FK steering rep argmax", "0.962", dadd(GR, "fk_noise", "argmax", 32), usrc + " (logs; additive == full on R_rep)")
check(S, T + ": FK steering CLAP draw", "0.546", mus("fk_noise", "draw", 32)["reward"], "u music_rows")
check(S, T + ": FK steering CLAP argmax", "0.559", mus("fk_noise", "argmax", 32)["reward"], "u music_rows (regenerated)")
check(S, T + ": FK steering motif draw", "0.908", mot("fk_noise", "draw", 32)["R_mean"], m4s)
check(S, T + ": FK steering motif argmax", "0.932", mot("fk_noise", "argmax", 32)["R_mean"], m4s)
check(S, T + ": LatticeSMC BA draw", "0.385", dfull(GB, "lattice_smc_notwist", "draw", 32), usrc + " (lattice_smc_notwist)")
check(S, T + ": LatticeSMC BA argmax", "0.428", dfull(GB, "lattice_smc_notwist", "argmax", 32), usrc + " (regenerated)")
check(S, T + ": LatticeSMC rep draw", "0.924", dfull(GR, "lattice_smc_notwist", "draw", 32), usrc)
check(S, T + ": LatticeSMC rep argmax", "0.947", dfull(GR, "lattice_smc_notwist", "argmax", 32), usrc)
check(S, T + ": LatticeSMC CLAP draw", "0.547", mus("lattice_smc_prefix_b1", "draw", 32)["reward"], "u music_rows")
check(S, T + ": LatticeSMC CLAP argmax", "0.560", mus("lattice_smc_prefix_b1", "argmax", 32)["reward"], "u music_rows")
check(S, T + ": LatticeSMC motif draw", "0.903", mot("lattice_smc_prefix_b1", "draw", 32)["R_mean"], m4s)
check(S, T + ": LatticeSMC motif argmax", "0.929", mot("lattice_smc_prefix_b1", "argmax", 32)["R_mean"], m4s)
d, lo, hi = ppair(P2A, "lattice_smc_notwist", "bon_is", 32)
check(S, T + ": LatticeSMC - best-of-N, BA draw (paired)", "+0.038", d, "results/phase2b_a002_analysis.json pairwise_N8_N32 (notwist - bon_is)", f"CI [{lo:.4f}, {hi:.4f}]")
check(S, T + ": LatticeSMC - best-of-N, BA argmax", "+0.074", dfull(GB, "lattice_smc_notwist", "argmax", 32) - dfull(GB, "bon_argmax", "argmax", 32), usrc + " (difference of the two row means; no paired cell in any file)")
d, lo, hi = ppair(P3A, "lattice_smc_notwist", "bon_is", 32)
check(S, T + ": LatticeSMC - best-of-N, rep draw (paired)", "+0.028", d, "results/phase3_rep_a002_analysis.json pairwise_N8_N32", f"CI [{lo:.4f}, {hi:.4f}]")
check(S, T + ": LatticeSMC - best-of-N, rep argmax", "+0.046", dfull(GR, "lattice_smc_notwist", "argmax", 32) - dfull(GR, "bon_argmax", "argmax", 32), usrc + " (difference of row means 0.9467 - 0.9013)")
d, lo, hi = m3pair("lattice_smc_prefix_b1", "bon_is", 32)
check(S, T + ": LatticeSMC - best-of-N, CLAP draw (paired)", "+0.020", d, m3s + " pairs", f"CI [{lo:.4f}, {hi:.4f}]")
check(S, T + ": LatticeSMC - best-of-N, CLAP argmax", "+0.030", mus("lattice_smc_prefix_b1", "argmax", 32)["reward"] - mus("bon_argmax", "argmax", 32)["reward"], "u music_rows (difference of row means)")
d, lo, hi = m4pair("lattice_smc_prefix_b1", "bon_is", 32)
check(S, T + ": LatticeSMC - best-of-N, motif draw (paired)", "-0.002", d, m4s + " pairs", f"CI [{lo:.4f}, {hi:.4f}]")
check(S, T + ": LatticeSMC - best-of-N, motif argmax", "+0.011", mot("lattice_smc_prefix_b1", "argmax", 32)["R_mean"] - mot("bon_argmax", "argmax", 32)["R_mean"], m4s + " (difference of row means; the paired cell in the file is vs bon_is: +0.0248)")
p = upair(GB, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk", 32)
check(S, T + ": LatticeSMC - chunk pruning, BA argmax (paired, full)", "+0.013", p["diff"], "results/u/u_analysis.json dance_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
d, lo, hi = ppair(P3A, "lattice_smc_notwist", "greedy_chunk", 32)
check(S, T + ": LatticeSMC - chunk pruning, rep draw (paired)", "+0.036", d, "results/phase3_rep_a002_analysis.json pairwise_N8_N32", f"CI [{lo:.4f}, {hi:.4f}]")
p = upair(GR, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk", 32)
check(S, T + ": LatticeSMC - chunk pruning, rep argmax (paired)", "+0.059", p["diff"], "u dance_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
p = next(p for p in U["music_pairs"] if p["A"] == "lattice_smc_prefix_b1-argmax" and p["B"] == "greedy_chunk" and p["N"] == 32)
check(S, T + ": LatticeSMC - chunk pruning, CLAP argmax (paired)", "+0.009", p["diff"], "results/u/u_analysis.json music_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
d, lo, hi = m4pair("lattice_smc_prefix_b1-argmax", "greedy_chunk", 32)
check(S, T + ": LatticeSMC - chunk pruning, motif argmax (paired)", "+0.011", d, m4s + " pairs", f"CI [{lo:.4f}, {hi:.4f}]")
gap = next(g for g in P2A["gap"] if g["method"] == "fk_noise")
text(S, T + " caption: dagger values 'within 0.01 of the full reward'", "within 0.01", f"fk_noise mean |full - additive| = {gap['abs_gap_mean']:.4f}, signed full - additive = {gap['signed_full_minus_additive_mean']:.4f} (draw rows, 4M); 38M: {next(g for g in LBA['gap'] if g['method']=='fk_noise')['abs_gap_mean']:.4f} / {next(g for g in LBA['gap'] if g['method']=='fk_noise')['signed_full_minus_additive_mean']:.4f}",
     "results/phase2b_a002_analysis.json gap; results/L_ba_a002_analysis.json gap", True, "the gap of the argmax particles themselves is not in any file; see Section 4")
# 4.2
d, lo, hi = ppair(P2A, "lattice_smc_notwist", "bon_is", 32)
check(S, "4.2: draw-rule gain over terminal IS, beat alignment", "0.038", d, "phase2b_a002 pairwise", f"interval [{lo:.4f}, {hi:.4f}] excludes zero: {lo > 0}")
d, lo, hi = ppair(P3A, "lattice_smc_notwist", "bon_is", 32)
check(S, "4.2: draw-rule gain, repetition", "0.028", d, "phase3_rep_a002 pairwise", f"interval [{lo:.4f}, {hi:.4f}] excludes zero: {lo > 0}")
d, lo, hi = m3pair("lattice_smc_prefix_b1", "bon_is", 32)
check(S, "4.2: draw-rule gain, prompt adherence", "0.020", d, m3s + " pairs", f"interval [{lo:.4f}, {hi:.4f}] excludes zero: {lo > 0}")
for lab, pr, val, src in [("4.2: argmax BA LatticeSMC", "0.428", dfull(GB, "lattice_smc_notwist", "argmax", 32), usrc), ("4.2: argmax BA best-of-N", "0.354", dfull(GB, "bon_argmax", "argmax", 32), usrc),
                          ("4.2: argmax rep LatticeSMC", "0.947", dfull(GR, "lattice_smc_notwist", "argmax", 32), usrc), ("4.2: argmax rep best-of-N", "0.901", dfull(GR, "bon_argmax", "argmax", 32), usrc),
                          ("4.2: argmax CLAP LatticeSMC", "0.560", mus("lattice_smc_prefix_b1", "argmax", 32)["reward"], "u music_rows"), ("4.2: argmax CLAP best-of-N", "0.530", mus("bon_argmax", "argmax", 32)["reward"], "u music_rows"),
                          ("4.2: argmax motif LatticeSMC", "0.929", mot("lattice_smc_prefix_b1", "argmax", 32)["R_mean"], m4s), ("4.2: argmax motif best-of-N", "0.918", mot("bon_argmax", "argmax", 32)["R_mean"], m4s)]:
    check(S, lab, pr, val, src)
ra = {r["method"]: r["frac_events_resampled"] for r in P2B["resampling_activity"] if r["alpha"] == 0.02 and r["N"] == 32}
text(S, "4.2 / 4.4 / Fig 7 caption: 'ESS falls below N/2 at 91 to 98 percent of chunk boundaries'", "91 to 98 percent",
     f"alpha 0.02, N = 32: lattice_smc_notwist {ra['lattice_smc_notwist']:.4f}, lattice_smc {ra['lattice_smc']:.4f}, fk_noise {ra['fk_noise']:.4f} (fk_noise events are within-chunk steps)",
     "results/phase2b_combined.json resampling_activity", 0.905 <= ra["lattice_smc_notwist"] and ra["lattice_smc"] <= 0.985)
p = upair(GB, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk", 32); check(S, "4.2: argmax lead over pruning, BA", "0.013", p["diff"], "u dance_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
p = upair(GR, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk", 32); check(S, "4.2: argmax lead over pruning, rep", "0.059", p["diff"], "u dance_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
p = next(p for p in U["music_pairs"] if p["A"] == "lattice_smc_prefix_b1-argmax" and p["B"] == "greedy_chunk" and p["N"] == 32); check(S, "4.2: argmax lead over pruning, CLAP", "0.009", p["diff"], "u music_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
d, lo, hi = m4pair("lattice_smc_prefix_b1-argmax", "greedy_chunk", 32); check(S, "4.2: argmax lead over pruning, motif", "0.011", d, m4s, f"CI [{lo:.4f}, {hi:.4f}]")
d, lo, hi = ppair(P3A, "lattice_smc_notwist", "greedy_chunk", 32); check(S, "4.2: draw lead over pruning, rep", "0.036", d, "phase3_rep_a002 pairwise", f"CI [{lo:.4f}, {hi:.4f}]")
# schedules within 0.005
diffs = {}
for ana, name in [(P2A, "BA"), (P3A, "rep")]:
    for p in ana["pairwise_all_N"]:
        if {p["A"], p["B"]} == {"fk_noise", "lattice_smc_notwist"}:
            diffs[f"{name} N={p['N']}"] = abs(p["diff_mean"])
for p in M3["pairs"]:
    if {p["A"], p["B"]} == {"fk_noise", "lattice_smc_prefix_b1"}:
        diffs[f"CLAP N={p['N']}"] = abs(p["diff"])
for p in M4["pairs"]:
    if {p["A"], p["B"]} == {"fk_noise", "lattice_smc_prefix_b1"}:
        diffs[f"motif N={p['N']}"] = abs(p["diff"])
worst = max(diffs, key=diffs.get)
text(S, "4.2: 'boundary and dense schedules end within 0.005 of each other at every budget on all four rewards' (draw rows)", "within 0.005",
     "|FK draw - LatticeSMC draw|: " + ", ".join(f"{k} {v:.4f}" for k, v in diffs.items()), "phase2b_a002 / phase3_rep_a002 pairwise_all_N; m3 / m4 pairs (N = 8, 32 only)",
     all(v <= 0.005 for v in diffs.values()), f"largest {worst} = {diffs[worst]:.4f}; violated on beat alignment at N = 4, 8 and on repetition at N = 4, 8, 16, and on motif at N = 32")
# held-out inside base interval
viol = []
for ana, name in [(P2A, "BA 4M"), (P3A, "rep 4M"), (LBA, "BA 38M"), (LREP, "rep 38M")]:
    base = next(h for h in ana["heldout"]["32"] if h["method"] == "base")
    for N in ("8", "32"):
        for h in ana["heldout"][N]:
            for k in ("pfc", "realism_w1"):
                if not (base[k]["ci_lo"] <= h[k]["mean"] <= base[k]["ci_hi"]):
                    viol.append(f"{name} {h['method']} N={N} {k} {h[k]['mean']:.4f}")
for grid in (GB, GR):
    for N in (8, 32):
        r = dance(grid, "lattice_smc_notwist", "argmax", N)
        ana = P2A if grid == GB else P3A
        base = next(h for h in ana["heldout"]["32"] if h["method"] == "base")
        for k, kk in (("pfc", "pfc"), ("realism_w1", "realism_w1")):
            if not (base[k]["ci_lo"] <= r[kk] <= base[k]["ci_hi"]):
                viol.append(f"{grid} notwist-argmax N={N} {k} {r[kk]:.4f}")
text(S, "4.2: 'every held-out quantity stays inside its base interval for every method at every budget' (dance PFC, W1 at N = 8, 32, both models, incl. regenerated argmax rows)", "inside base interval",
     f"base 4M PFC [{P2A['heldout']['32'][0]['pfc']['ci_lo']:.3f}, {P2A['heldout']['32'][0]['pfc']['ci_hi']:.3f}], W1 [{P2A['heldout']['32'][0]['realism_w1']['ci_lo']:.5f}, {P2A['heldout']['32'][0]['realism_w1']['ci_hi']:.5f}]; violations: {viol or 'none'}",
     "phase2b_a002 / phase3_rep_a002 / L_ba_a002 / L_rep_a002 heldout; u dance_rows", not viol,
     "only N = 8 and 32 are in the analysis files ('every budget' cannot be checked at N = 2, 4, 16); music held-out metrics have no base bootstrap interval in any analysis file (m3/m4 rows carry means only)")
cl = [r for r in M4["rows"] if r["N"] == 32 and r["rule"] == "draw"]
text(S, "4.2: 'steering on motif recurrence leaves prompt adherence at its base value'", "base value 0.470",
     "terminal CLAP of the N = 32 draw rows on the motif grid: " + ", ".join(f"{r['method']} {r['clap_terminal']:.4f}" for r in cl) + f"; base 0.4703 [0.449, 0.491]", m4s, all(0.449 <= r["clap_terminal"] <= 0.491 for r in cl))
# 4.3
p = upair(GB, "lattice_smc_notwist-argmax", "lattice_smc_notwist (draw)", 32); check(S, "4.3: argmax - draw, BA", "0.042", p["diff"], "u dance_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
p = upair(GR, "lattice_smc_notwist-argmax", "lattice_smc_notwist (draw)", 32); check(S, "4.3: argmax - draw, rep", "0.023", p["diff"], "u dance_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
p = next(p for p in U["music_pairs"] if p["A"] == "lattice_smc_prefix_b1-argmax" and p["B"] == "lattice_smc_prefix_b1 (draw)" and p["N"] == 32); check(S, "4.3: argmax - draw, CLAP", "0.012", p["diff"], "u music_pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
d, lo, hi = m4pair("lattice_smc_prefix_b1-argmax", "lattice_smc_prefix_b1", 32); check(S, "4.3: argmax - draw, motif", "0.027", d, m4s, f"CI [{lo:.4f}, {hi:.4f}]")
p = upair(GB, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk", 8); check(S, "4.3: pruning ahead by 0.015 at N = 8 on BA", "0.015", -p["diff"], "u dance_pairs (full, regenerated)", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
cb = {N: wcross(GB, f"lattice_smc_notwist|argmax|{N}") for N in (16, 32)}
text(S, "4.3: BA crossover between N = 16 and 32", "N = 16 -> 32", f"argmax - pruning: N=16 {cb[16]['diff']:+.4f} [{cb[16]['ci_lo']:.4f}, {cb[16]['ci_hi']:.4f}], N=32 {cb[32]['diff']:+.4f} [{cb[32]['ci_lo']:.4f}, {cb[32]['ci_hi']:.4f}]", "results/w/w_analysis.json crossover", cb[16]["diff"] < 0 < cb[32]["diff"])
cc = {N: wcross("music terminal CLAP", f"lattice_smc_prefix_b1|argmax|{N}") for N in (4, 8, 16)}
text(S, "4.3: CLAP crossover at N = 8", "N = 8", ", ".join(f"N={N} {v['diff']:+.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]" for N, v in cc.items()), "w crossover", cc[4]["diff"] < 0 < cc[8]["diff"], "sign change at N = 8; first interval excluding zero at N = 16")
cm = {N: wcross("music R_motif", f"lattice_smc_prefix_b1|argmax|{N}") for N in (2, 4, 8)}
text(S, "4.3: motif crossover at N = 4", "N = 4", ", ".join(f"N={N} {v['diff']:+.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]" for N, v in cm.items()), "w crossover", cm[4]["diff"] > 0, "already +0.0007 at N = 2 (all rules coincide at N = 2); first interval excluding zero at N = 8")
cr = {(rule, N): wcross(GR, f"lattice_smc_notwist|{rule}|{N}") for rule in ("draw", "argmax") for N in (4,)}
text(S, "4.3: repetition ahead from N = 4 under both rules", "N = 4", ", ".join(f"{k[0]} N={k[1]} {v['diff']:+.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]" for k, v in cr.items()), "w crossover", all(v["ci_lo"] > 0 for v in cr.values()))
div_l, div_f = mus("lattice_smc_prefix_b1", "argmax", 32)["diversity"] / mus("lattice_smc_prefix_b1", "draw", 32)["diversity"] - 1, mus("fk_noise", "argmax", 32)["diversity"] / mus("fk_noise", "draw", 32)["diversity"] - 1
text(S, "4.3: argmax reduces music embedding diversity by 8 to 9 percent relative to the draw", "8 to 9 percent", f"LatticeSMC {100*div_l:.1f} %, FK {100*div_f:.1f} % (0.0542 -> 0.0497; 0.0527 -> 0.0484)", "u music_rows", False, "both reductions are 8.2-8.3 %, i.e. 'about 8 percent'; '8 to 9' is the RESULTS.md wording, not an interval in the files")
text(S, "4.3: argmax keeps dance pose diversity at the base value", "base value", f"notwist-argmax N=32 diversity {dance(GB, 'lattice_smc_notwist', 'argmax', 32)['diversity_root_rel']:.4f} vs base {next(d for d in P2A['diversity'] if d['method']=='base')['pairwise_joint_dist_root_relative_m']:.4f}", "u dance_rows; phase2b_a002 diversity", True)
# 4.4
for lab, pr, val in [("4.4: decoder calls dense N=32", "512", mus("fk_noise", "argmax", 32)["decode_calls"]), ("4.4: decoder calls boundary N=32", "128", mus("lattice_smc_prefix_b1", "argmax", 32)["decode_calls"]), ("4.4: decoder calls best-of-N N=32", "32", mus("bon_argmax", "argmax", 32)["decode_calls"])]:
    check(S, lab, pr, val, "u music_rows decode_calls", d=0)
check(S, "4.4 / Fig 6 caption: wall-clock ratio dense / boundary", "3.6", mus("fk_noise", "argmax", 32)["wall_s"] / mus("lattice_smc_prefix_b1", "argmax", 32)["wall_s"], "u music_rows wall_s (2314.2 / 647.5)")
check(S, "4.4: dense draw N=8", "0.538", mus("fk_noise", "draw", 8)["reward"], "u music_rows")
check(S, "4.4: dense draw N=32", "0.546", mus("fk_noise", "draw", 32)["reward"], "u music_rows")
check(S, "4.4: boundary draw N=32", "0.547", mus("lattice_smc_prefix_b1", "draw", 32)["reward"], "u music_rows")
check(S, "4.4: boundary argmax N=32", "0.560", mus("lattice_smc_prefix_b1", "argmax", 32)["reward"], "u music_rows")
ra2 = {r["method"]: r["frac_events_resampled"] for r in P2B["resampling_activity"] if r["alpha"] == 0.2 and r["N"] == 32}
text(S, "4.4 / Fig 7 caption: at alpha = 0.2 no method ever resamples", "0", f"alpha 0.2, N = 32: {ra2}", "phase2b_combined resampling_activity", all(v == 0 for v in ra2.values()))
o = X[GB]["ordering_N32"]
text(S, "4.4: on beat alignment all ten pairwise orderings among the five common methods identical under both rules", "10 / 10, both rules", f"draw|full {o['draw|full']['pairwise_sign_agreement']}, argmax|additive {o['argmax|additive']['pairwise_sign_agreement']}", "results/x/x_analysis.json ordering_N32", o["draw|full"]["pairwise_sign_agreement"] == [10, 10] and o["argmax|additive"]["pairwise_sign_agreement"] == [10, 10])
p = xpair(GR, "L (37.6 M)", "lattice_smc_notwist-argmax - greedy_chunk", "full (regenerated)", 32); check(S, "4.4: 38M repetition argmax - pruning", "+0.028", p["diff"], xsrc + " L pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
cl16 = xcross(GB, "lattice_smc_notwist|argmax|16"); cl32 = xcross(GB, "lattice_smc_notwist|argmax|32")
text(S, "4.4: 38M crossover against pruning between N = 16 and 32 (BA)", "N = 16 -> 32", f"N=16 {cl16['diff']:+.4f} [{cl16['ci_lo']:.4f}, {cl16['ci_hi']:.4f}], N=32 {cl32['diff']:+.4f}", xsrc + " L crossover", cl32["ci_lo"] > 0 and cl16["ci_lo"] < 0, "at N = 16 the 38M difference is already +0.0017 (interval contains zero); the sign change is between 8 and 16")
# 5.1
ratios = {k: v["across_over_within"] for k, v in TI_BA.items()}
text(S, "5.1 / Table 3: beat alignment between-prefix share 'at most 0.15' (<= 0.13, <= 0.13, <= 0.15)", "<= 0.13 / 0.13 / 0.15",
     f"exp(S/alpha) across/within variance ratio (upper bound on the share), alpha 0.02, M = 8: {ratios['1']:.3f} / {ratios['2']:.3f} / {ratios['3']:.3f}; share of total {TI_BA['1']['across_over_total']:.3f} / {TI_BA['2']['across_over_total']:.3f} / {TI_BA['3']['across_over_total']:.3f}",
     "data/twist/train_info_a002.json variance_decomposition_exp_S_over_alpha", False,
     "no file carries the S-scale (reward-scale) share for beat alignment; the exp-scale bounds are 0.13 / 0.13 / 0.14 (k = 3 prints 0.15). RESULTS.md Phase 2a quotes S-scale sds 0.022/0.017/0.012 across vs 0.044/0.036/0.025 within, whose raw shares are 0.20 / 0.18 / 0.19. A value of exactly <= 0.15 at k = 3 is in no file")
for k, pr in (("1", "0.47"), ("2", "0.62"), ("3", "0.66")):
    check(S, f"5.1 / Table 3: prompt adherence within-prompt share k={k}", pr, M2[f"b2_var_k{k}_share_bias_corrected"], "results/m2/M2_SUMMARY.json b2_var_k*_share_bias_corrected")
for k, pr in (("1", "0.46"), ("2", "0.73"), ("3", "0.84")):
    check(S, f"5.1: prefix-score / terminal correlation within prompt k={k}", pr, M2[f"b2_corr_prefixclap_k{k}_vs_terminal_within_prompt_r"], "results/m2/M2_SUMMARY.json")


def s_share(v):
    sa, sw, M = v["S_sd_across_prefixes_of_mean"], v["S_sd_within_prefix"], v["M"]
    return (sa ** 2 - sw ** 2 / M) / (sa ** 2 - sw ** 2 / M + sw ** 2)


check(S, "5.1: repetition pooled share 'is 0.9'", "0.9", s_share(DEC_REP["1"]), "data/twist_rep/decomp_a002.json S_sd_* fields, bias-corrected share (k = 1)", "k = 1/2/3: " + " / ".join(f"{s_share(DEC_REP[k]):.3f}" for k in ("1", "2", "3")))
text(S, "5.1 / Table 3: repetition within-condition share 'bounded by rollout noise at about 0.5' (<= 0.5, <= 0.5)", "<= 0.5", f"phase3d within-set target sd {D3W['target_within_set_sd_mean']['1']:.3f} / {D3W['target_within_set_sd_mean']['2']:.3f} nats vs MC RMS error at M = 4 {D3['mc_se_nats_from_sets_rms_vs_M8']['M=4']['1']:.2f} / {D3['mc_se_nats_from_sets_rms_vs_M8']['M=4']['2']:.2f} (log-target scale, not a share)",
     "results/phase3d_analysis.json, results/phase3d_wstwist_train.json", False, "no analysis file computes a within-condition S-scale share for repetition; it would have to be produced by lattice_smc/twist/train_sets.py or a new script on data/twist_rep/sets/")
# 5.2
check(S, "5.2 / Table 4: cross-prompt R^2 (4M)", "0.90", CAL_REP["learned_twist_vs_future_target"]["r2"], "results/rep_twist_calibration_a002.json")
check(S, "5.2 / Table 4: within-set correlation with oracle, k = 1", "-0.04", C3W["within_set_corr_learned_oracle_mean"]["1"], "results/phase3c_within_set.json")
check(S, "5.2: within-set correlation k = 2", "0.35", C3W["within_set_corr_learned_oracle_mean"]["2"], "results/phase3c_within_set.json")
check(S, "5.2 / Table 4 / B.3: residual 4.3 nats", "4.3", CAL_REP["learned_twist_vs_future_target"]["rmse"], "results/rep_twist_calibration_a002.json rmse")
check(S, "5.2: weight error factor 10 (median)", "10", B3T["1.0"]["exp_median"], "results/phase3b_tempered_residuals.json exp_median", d=0, tol=1.0)
check(S, "5.2: weight error factor 900 (90th pct)", "900", B3T["1.0"]["exp_p90"], "results/phase3b_tempered_residuals.json exp_p90", d=0, tol=20)
d, lo, hi = ppair(P3A, "lattice_smc", "lattice_smc_notwist", 32)
check(S, "5.2: learned twist lowers reward by 0.05 at N = 32 (4M)", "0.05", -d, "phase3_rep_a002 pairwise (lattice_smc - notwist)", f"{d:+.4f} [{lo:.4f}, {hi:.4f}]")
p = xpair(GR, "L (37.6 M)", "lattice_smc (draw) - lattice_smc_notwist (draw)", "full", 32)
check(S, "5.2: '... at both model sizes' (38M)", "0.05", -p["diff"], xsrc + " L pairs", f"38M effect is {p['diff']:+.4f} [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}], i.e. 0.018, not 0.05 (Table 4 itself prints -0.018)")
# 5.3
d, lo, hi = m3pair("lattice_smc_prefix_b1", "bon_is", 32); check(S, "5.3: prefix score raises terminal CLAP by 0.020 at N = 32", "0.020", d, m3s + " pairs", f"CI [{lo:.4f}, {hi:.4f}]")
text(S, "5.3: beta = 1 ahead of 0.5 ahead of 0.25", "monotone", f"N=32 draw: {mus('lattice_smc_prefix_b1','draw',32)['reward']:.4f} > {mus('lattice_smc_prefix_b0.5','draw',32)['reward']:.4f} > {mus('lattice_smc_prefix_b0.25','draw',32)['reward']:.4f}; paired 1 vs 0.5 {m3pair('lattice_smc_prefix_b1','lattice_smc_prefix_b0.5',32)[0]:+.4f} [{m3pair('lattice_smc_prefix_b1','lattice_smc_prefix_b0.5',32)[1]:.4f}, {m3pair('lattice_smc_prefix_b1','lattice_smc_prefix_b0.5',32)[2]:.4f}]", m3s, True, "1 vs 0.5 does not separate at N = 32 (it does at N = 8)")
o = B3["question"]["oracle_vs_notwist_N32"]; check(S, "5.3 / Table 4 / B.4: oracle +0.058 at N = 32", "0.058", o["diff"], "results/phase3b_analysis.json question.oracle_vs_notwist_N32", f"CI [{o['ci_lo']:.4f}, {o['ci_hi']:.4f}]; 10 prompts x 4 seeds")
rp = D3["rollout_pairs"]
text(S, "5.3: rollouts lose to additional particles at every budget for M <= 4", "6 / 6 negative", ", ".join(f"{p['diff']:+.4f}{'*' if p['separates'] else ''}" for p in rp), "results/phase3d_analysis.json rollout_pairs", all(p["diff"] < 0 for p in rp), "N = 8, M = 1 interval contains zero (stated in B.4)")
check(S, "5.3 / B.3: MC error of within-set targets 'about 2 nats'", "2", D3["mc_se_nats_from_sets_rms_vs_M8"]["M=4"]["1"], "results/phase3d_analysis.json mc_se_nats_from_sets_rms_vs_M8 M=4 k=1 (RMS vs the M = 8 estimate)", d=0, tol=0.5)
check(S, "5.3 / B.3: within-set spread 'about 3' / 2.9 nats", "2.9", D3W["target_within_set_sd_mean"]["1"], "results/phase3d_wstwist_train.json target_within_set_sd_mean k=1")
check(S, "5.3 / B.3: within-set Spearman 0.08 (k=1)", "0.08", D3W["heldout_within_set"]["1"]["spearman"], "results/phase3d_wstwist_train.json")
check(S, "5.3 / B.3: within-set Spearman 0.13 (k=2)", "0.13", D3W["heldout_within_set"]["2"]["spearman"], "results/phase3d_wstwist_train.json")
# Fig 1-3 captions, Fig 2 caption
text(S, "Figure 1 / Figure 3 captions", "N/2 threshold; M continuations", "ESS < N/2 is the implemented threshold (methods/*.py); no other numbers in these captions", "code", True)
text(S, "Figure 2 caption: 400N / 32N, 40 conditions, alpha = 0.02", "400N; 32N; 40; 0.02", "nfe_audit expected 400 N (dance), rows nfe 32 N (music); n_prompts 40; alpha 0.02 grid", "phase analyses; m3/m4 rows", True)

# ================================================================================================= Section 3: appendix
S = "3. Appendix tables"
# Table 3 handled above (rows tagged 5.1 / Table 3). Table 4:
check(S, "Table 4: learned regressor 4M cross-prompt R^2", "0.90", CAL_REP["learned_twist_vs_future_target"]["r2"], "results/rep_twist_calibration_a002.json")
check(S, "Table 4: 4M within-set corr.", "-0.04", C3W["within_set_corr_learned_oracle_mean"]["1"], "results/phase3c_within_set.json")
check(S, "Table 4: 4M residual", "4.3", CAL_REP["learned_twist_vs_future_target"]["rmse"], "results/rep_twist_calibration_a002.json")
pc = next(p for p in P3C["primary_comparisons"] if p["alpha"] == 0.02 and p["pair"] == "lattice_smc - lattice_smc_notwist" and p["N"] == 32)
check(S, "Table 4: 4M effect", "-0.053", pc["diff"], "results/phase3_rep_combined.json primary_comparisons"); check(S, "Table 4: 4M effect CI lo", "-0.083", pc["ci_lo"], "phase3_rep_combined primary_comparisons"); check(S, "Table 4: 4M effect CI hi", "-0.028", pc["ci_hi"], "phase3_rep_combined primary_comparisons")
check(S, "Table 4: 38M cross-prompt R^2", "0.91", CAL_L["learned_twist_vs_future_target"]["r2"], "results/L_rep_twist_calibration_a002.json")
check(S, "Table 4: 38M residual", "3.3", CAL_L["learned_twist_vs_future_target"]["rmse"], "results/L_rep_twist_calibration_a002.json")
p = xpair(GR, "L (37.6 M)", "lattice_smc (draw) - lattice_smc_notwist (draw)", "full", 32)
check(S, "Table 4: 38M effect", "-0.018", p["diff"], xsrc + " L pairs"); check(S, "Table 4: 38M effect CI lo", "-0.032", p["ci_lo"], xsrc); check(S, "Table 4: 38M effect CI hi", "-0.006", p["ci_hi"], xsrc)
check(S, "Table 4: within-set regressor corr.", "0.01", D3W["oracle_sets"]["1"]["pearson_new_vs_oracle"], "results/phase3d_wstwist_train.json oracle_sets k=1 pearson", "Spearman 0.015")
pd = next(p for p in D3["pairs"] if p["N"] == 32)
check(S, "Table 4: within-set regressor effect", "-0.014", pd["diff"], "results/phase3d_analysis.json pairs N=32"); check(S, "Table 4: its CI lo", "-0.033", pd["ci_lo"], "phase3d pairs"); check(S, "Table 4: its CI hi", "-0.002", pd["ci_hi"], "phase3d pairs")
check(S, "Table 4: oracle residual ~0.6 nats", "0.6", C3W["oracle_mc_se_nats_mean"]["1"], "results/phase3c_within_set.json oracle_mc_se_nats_mean k=1")
check(S, "Table 4: oracle effect", "+0.058", o["diff"], "phase3b question.oracle_vs_notwist_N32"); check(S, "Table 4: oracle CI lo", "+0.034", o["ci_lo"], "phase3b"); check(S, "Table 4: oracle CI hi", "+0.083", o["ci_hi"], "phase3b")
text(S, "Table 4: oracle within-set corr. '1'", "1", "by definition (the oracle is the reference); not a measured value", "-", True)
check(S, "B.2 text: between-sequence residual component", "0.27", 1 - C3R["within_share"], "results/phase3c_residual_decomposition.json (1 - within_share)")
check(S, "B.2 text: within-sequence component", "0.73", C3R["within_share"], "results/phase3c_residual_decomposition.json within_share")
# B.3
for b, pr in (("0.5", "2.1"), ("0.25", "1.1"), ("0.1", "0.4")):
    check(S, f"B.3: tempered residual beta={b}", pr, B3T[b]["rmse_nats"], "results/phase3b_tempered_residuals.json")
tp = {}
for p in B3["pairs"]:
    tp.setdefault((p.get("variant"), p.get("vs"), p["N"]), p)
pair_by = {}
for p in B3["pairs"]:
    pair_by[(p.get("variant"), p["N"])] = p
for var, pr in (("tempered beta 0.5", "-0.017"), ("tempered beta 0.25", "+0.003"), ("tempered beta 0.1", "-0.002")):
    pp = pair_by.get((var, 32))
    check(S, f"B.3: effect at N=32, {var}", pr, pp["diff"] if pp else None, "results/phase3b_analysis.json pairs (variant - notwist)")
r01 = next(r for r in B3["rows"] if r["variant"] == "tempered beta 0.1" and r["N"] == 32)
check(S, "B.3: resampling fraction at early boundaries falls to 0.05 (beta 0.1)", "0.05", (r01["frac_resampled_k1"] + r01["frac_resampled_k2"]) / 2, "results/phase3b_analysis.json rows (k1 0.0625, k2 0.05)", tol=0.02)
r10 = next(r for r in B3["rows"] if r["variant"] == "learned (beta 1)" and r["N"] == 32)
check(S, "B.3: ... from 1.00 (beta 1)", "1.00", (r10["frac_resampled_k1"] + r10["frac_resampled_k2"]) / 2, "phase3b rows")
check(S, "B.3: ensemble agrees with itself to about 1 nat", "1", C3E["fresh_member_spread_sd_mean"], "results/phase3c_ensemble.json fresh_member_spread_sd_mean", d=0, tol=0.2)
check(S, "B.3: ensemble mean residual 4.1 nats", "4.1", C3E["fresh"]["ensemble_mean"]["rmse_nats"], "results/phase3c_ensemble.json fresh.ensemble_mean.rmse_nats")
check(S, "B.3: shrink factor 0.996", "0.996", C3E["shrink_factor_fresh_mean"], "results/phase3c_ensemble.json shrink_factor_fresh_mean")
for var, pr in (("ensemble mean (5)", "-0.048"), ("ensemble shrunk (5, c)", "-0.052")):
    pp = pair_by.get((var, 32)); check(S, f"B.3: effect {var}", pr, pp["diff"] if pp else None, "phase3b pairs")
text(S, "B.3: 200 sets of 16 same-condition prefixes each continued 8 times", "200; 16; 8", f"n_sets train {D3W['n_sets_train']} + heldout {D3W['n_sets_heldout']} (100 segments x 2 seeds x k in {{1,2}}); n_particles {D3W['n_particles']}; M = 8 (RESULTS.md Phase 3d)", "results/phase3d_wstwist_train.json", D3W["n_sets_train"] + D3W["n_sets_heldout"] == 400, "the file holds 400 sets = 200 (segment, seed) pairs x 2 chunk indices; '200 sets' counts (segment, seed) pairs")
# B.4
for (N, M, pr) in ((4, 1, "-0.029"), (4, 2, "-0.058"), (4, 4, "-0.071"), (8, 1, "-0.009"), (8, 2, "-0.017"), (8, 4, "-0.040")):
    idx = {(4, 1): 0, (8, 1): 1, (4, 2): 2, (8, 2): 3, (4, 4): 4, (8, 4): 5}[(N, M)]
    pp = rp[idx]; check(S, f"B.4: rollout - matched, N={N}, M={M}", pr, pp["diff"], "results/phase3d_analysis.json rollout_pairs", f"CI [{pp['ci_lo']:.4f}, {pp['ci_hi']:.4f}] separates {pp['separates']}")
for M, pr in (("M=1", "5.5"), ("M=2", "3.7"), ("M=4", "2.2")):
    check(S, f"B.4: MC error {M} (k=1)", pr, D3["mc_se_nats_from_sets_rms_vs_M8"][M]["1"], "results/phase3d_analysis.json mc_se_nats_from_sets_rms_vs_M8")
fr = [r[f"frac_resampled_k{k}"] for r in D3["rollout_rows"] if r["variant"].startswith("rollout") for k in (1, 2)]
text(S, "B.4: rollout resamples at 70 to 97 percent of early boundaries", "70 to 97", f"min {min(fr):.3f}, max {max(fr):.3f} over the six rollout cells x k in (1, 2)", "results/phase3d_analysis.json rollout_rows", 0.695 <= min(fr) and max(fr) <= 0.975, f"minimum is {min(fr):.2f} (N = 4, M = 4, k = 2), i.e. 67 percent")
onfe = {r["N"]: r["oracle_nfe_per_sequence"] for r in B3["oracle_same_sequences"] if "oracle" in r["variant"]}
check(S, "B.4: oracle costs 410x the sequence budget (N=8)", "410", onfe[8] / 3200, "results/phase3b_analysis.json oracle_nfe_per_sequence / (400 N)", d=0)
check(S, "B.4: ... 610x (N=32)", "610", onfe[32] / 12800, "results/phase3b_analysis.json", d=0)
# B.5
for st, pr in (("45", "0.995"), ("35", "0.985"), ("25", "0.974"), ("15", "0.952"), ("5", "0.858")):
    check(S, f"B.5: repetition plug-in R^2 at step {st}", pr, CAL_REP["plugin"][st]["vs_next_chunk_reward_same_continuation"]["r2"], "results/rep_twist_calibration_a002.json plugin[step].vs_next_chunk_reward_same_continuation.r2")
check(S, "B.5: beat alignment plug-in R^2 at step 45", "0.40", CAL_BA["plugin"]["45"]["vs_next_chunk_reward_same_continuation"]["r2"], "results/twist_calibration_a002.json")
b5 = CAL_BA["plugin"]["5"]["vs_next_chunk_reward_same_continuation"]
check(S, "B.5: beat alignment plug-in 'falling monotonically to 0.11 at step 5' (R^2)", "0.11", b5["r2"], "results/twist_calibration_a002.json plugin['5'].r2", f"R^2 at step 5 is {b5['r2']:.3f}; the Pearson r at step 5 is {b5['pearson_r']:.3f} (R^2 by step: " + ", ".join(f"{s}: {CAL_BA['plugin'][s]['vs_next_chunk_reward_same_continuation']['r2']:.3f}" for s in ("45", "35", "25", "15", "5")) + ")")
# A.5
check(S, "A.5: telescoping discrepancy below 1e-13 (music, worst)", "1e-13", max(r.get("telescoping_max_abs_err", 0) for r in M3["rows"] + M4["rows"]), "m3/m4 rows telescoping_max_abs_err (max 7.1e-14); dance: results/telescoping_check_N{8,32}.json max_abs_error_log 0.0 (alpha 0.02, 50 sequences)", d=13, tol=1e-13)
mw = {(r["alpha"], r["method"]): r["max_weight_ratio"] for r in P2B["resampling_activity"] if r["N"] == 32}
check(S, "A.5 / Fig 7 caption: max weight ratio at alpha 0.2 is 3", "3", mw[(0.2, "lattice_smc_notwist")], "phase2b_combined resampling_activity max_weight_ratio", d=0)
check(S, "A.5: max weight ratio at alpha 0.05 is 83", "83", mw[(0.05, "lattice_smc_notwist")], "phase2b_combined (notwist 81.6; lattice_smc 85.8; fk_noise 72.8)", d=0, tol=1.0)
text(S, "A.5 / Fig 7 caption: max weight ratio at alpha 0.02 'about 1e4'", "1e4", f"notwist {mw[(0.02, 'lattice_smc_notwist')]:.0f}, lattice_smc {mw[(0.02, 'lattice_smc')]:.0f}, fk_noise {mw[(0.02, 'fk_noise')]:.0f}", "phase2b_combined", True, "fk_noise's is 1e3")
# C.1 - C.3
check(S, "C.1: seam ratio 4M", "2.87", CHK_E["boundary"]["ratio_pooled_mean_seam_over_median_within"], "results/phase1c_E_checks.json")
check(S, "C.1: seam ratio 38M", "2.27", CHK_L["boundary"]["ratio_pooled_mean_seam_over_median_within"], "results/phase1c_L_checks.json")
check(S, "C.1: seam ratio ground truth", "0.93", CHK_E["boundary"]["gt_ratio_pooled_mean_seam_over_median_within"], "results/phase1c_E_checks.json")
text(S, "C.1: 4M 12,000 steps (latent 192, 4 layers); 38M 36,000 steps (latent 512, 6 layers); batch 64", "12000; 36000; 64", "adopted step 12000 / 9000 (36,000 trained); configs latent 192/4 layers and 512/6 layers; batch 64", "runs/chunk_dispE/adopted.json, runs/chunk_dispE_L/adopted.json, configs/*.yaml", True, "the 38M model was trained for 36,000 steps and the adopted checkpoint is step 9,000 (lowest held-out loss), as the sentence's 'adopting the checkpoint with the lowest held-out loss' allows")
text(S, "C.1: 'Sixty music-and-motion sequences are held out'", "60", "RESULTS.md Phase 0: the extended held-out set is 96 sequences (60 original + 36 added); prompts come from held-out sequences", "RESULTS.md Phase 0 / data/heldout_sequences_extended.json", False, "SPEC's original 60 held-out sequences were extended to 96 before any training")
check(S, "C.1: seam spectral-flux ratio of the raw decode 1.26", "1.26", M2["b2_seam_ratio_qual"], "results/m2/M2_SUMMARY.json b2_seam_ratio_qual (10-prompt qualification set); the 160-sequence base grid gives 1.267")
check(S, "C.1: within-chunk control 1.21", "1.21", M2["b2_seam_control_qual"], "results/m2/M2_SUMMARY.json b2_seam_control_qual (qualification set; base-grid control 1.157)")
check(S, "C.1: tempo adherence of base samples 0.95 within 5 percent", "0.95", M2["b2_tempo_within_5pct_qual"], "results/m2/M2_SUMMARY.json b2_tempo_within_5pct_qual (10-prompt qualification set; 160-clip whole-clip value 0.9625)")
check(S, "C.1: CLAP R@1 0.65 over 10 prompts", "0.65", M2["b2_clap_r1_mean"], "results/m2/M2_SUMMARY.json b2_clap_r1_mean")
check(S, "C.1: chance 0.1", "0.1", M2["b2_clap_r1_chance"], "results/m2/M2_SUMMARY.json")
gapall = max(g["abs_gap_mean"] for g in P2A["gap"])
check(S, "C.2: mean |additive - full| gap 0.01 on every method (BA)", "0.01", gapall, "results/phase2b_a002_analysis.json gap (max over methods 0.0124, min 0.0092)", tol=0.003)
check(S, "C.2: tau = 0.171 m", "0.171", json.load(open(os.path.join(ROOT, "data/rep_tau.json")))["tau"] if os.path.exists(os.path.join(ROOT, "data/rep_tau.json")) else 0.1706, "data/rep_tau.json (RESULTS.md Phase 3: tau = 0.1706 m)")
check(S, "C.2: real dancers score 0.412", "0.412", P3A["gt_reference"]["rep_full"], "results/phase3_rep_a002_analysis.json gt_reference")
check(S, "C.2: base model 0.718", "0.718", dfull(GR, "base", "-", 1), usrc)
text(S, "C.2: additive CLAP counterpart differs from the terminal value by under 2 percent", "2 percent", "RESULTS.md Phase M1: 'the same number up to a factor within 2 % (r = 0.99999)'", "results/m1/M1_REPORT.md", True, "structural statement; not in a JSON field")
text(S, "C.2: 99.4 percent of base clips have the final window closer to their own opening", "99.4", "159 / 160 = 0.994", "results/m4/motif_reference.json", True)
text(S, "C.3: 32 / 128 / 512 decoder and CLAP calls at N = 32", "32; 128; 512", f"{mus('bon_argmax','argmax',32)['decode_calls']:.0f} / {mus('greedy_chunk','argmax',32)['decode_calls']:.0f} / {mus('fk_noise','argmax',32)['decode_calls']:.0f}", "u music_rows", True)
m = re.search(r"Exclusive total: about (\d+) GPU-hours", GPUH)
check(S, "C.3: about 250 GPU-hours", "250", float(m.group(1)), "results/GPU_HOURS.md exclusive total", d=0, tol=5)
# Table 5
S5 = "3. Appendix tables"
p = upair(GB, "lattice_smc_notwist-argmax", "lattice_smc_notwist (draw)", 32); check(S5, "Table 5: argmax - draw BA", "+0.042", p["diff"], "u dance_pairs")
p = upair(GR, "lattice_smc_notwist-argmax", "lattice_smc_notwist (draw)", 32); check(S5, "Table 5: argmax - draw rep", "+0.023", p["diff"], "u dance_pairs")
p = next(p for p in U["music_pairs"] if p["A"] == "lattice_smc_prefix_b1-argmax" and p["B"] == "lattice_smc_prefix_b1 (draw)" and p["N"] == 32); check(S5, "Table 5: argmax - draw CLAP", "+0.012", p["diff"], "u music_pairs")
d, lo, hi = m4pair("lattice_smc_prefix_b1-argmax", "lattice_smc_prefix_b1", 32); check(S5, "Table 5: argmax - draw motif", "+0.027", d, m4s)
rowsT5 = {"Beat alignment, 4M": (GB, wcross, ["-0.032", "-0.029", "-0.015", "-0.002", "+0.013"], "lattice_smc_notwist|argmax|"),
          "Beat alignment, 38M": (GB, xcross, ["-0.032", "-0.027", "-0.012", "+0.002", "+0.015"], "lattice_smc_notwist|argmax|"),
          "Repetition, 4M": (GR, wcross, ["+0.007", "+0.026", "+0.064", "+0.078", "+0.059"], "lattice_smc_notwist|argmax|"),
          "Repetition, 38M": (GR, xcross, ["-0.003", "+0.009", "+0.020", "+0.034", "+0.028"], "lattice_smc_notwist|argmax|"),
          "Prompt adherence": ("music terminal CLAP", wcross, ["-0.012", "-0.003", "+0.002", "+0.004", "+0.009"], "lattice_smc_prefix_b1|argmax|"),
          "Motif recurrence": ("music R_motif", wcross, ["+0.001", "+0.008", "+0.008", "+0.012", "+0.011"], "lattice_smc_prefix_b1|argmax|"),
          "Repetition, 4M, draw rule (caption)": (GR, wcross, ["+0.007", "+0.020", "+0.053", "+0.057", "+0.036"], "lattice_smc_notwist|draw|")}
stars = {"Beat alignment, 4M": [1, 1, 1, 0, 1], "Beat alignment, 38M": [1, 1, 1, 0, 1], "Repetition, 4M": [0, 1, 1, 1, 1], "Repetition, 38M": [0, 0, 1, 1, 1], "Prompt adherence": [1, 0, 0, 1, 1], "Motif recurrence": [0, 0, 1, 1, 1], "Repetition, 4M, draw rule (caption)": [0, 1, 1, 1, 1]}
for name, (grid, fn, prs, key) in rowsT5.items():
    for N, pr, st in zip((2, 4, 8, 16, 32), prs, stars[name]):
        v = fn(grid, f"{key}{N}")
        src = ("results/w/w_analysis.json crossover" if fn is wcross else "results/x/x_analysis.json L crossover") + " (dance: additive scale of the logs)"
        note = f"CI [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]; star printed {bool(st)}, interval excludes zero {v['separates']}"
        if "Beat" in name and N in (8, 32):
            reg = upair(GB, "lattice_smc_notwist-argmax (regenerated)", "greedy_chunk", N) if "4M" in name else xpair(GB, "L (37.6 M)", "lattice_smc_notwist-argmax - greedy_chunk", "full (regenerated)", N)
            note += f"; regenerated full-reward cell {reg['diff']:+.4f}"
            # the draft prints the full-reward cell at N = 8, 32 (Table 1 / 10 values): accept either
            if abs(round(reg["diff"], 3) - float(pr)) < 5e-4:
                check(S5, f"Table 5: {name} N={N}", pr, reg["diff"], src.replace("(dance: additive scale of the logs)", "(regenerated full reward)"), note + f"; additive-scale value {v['diff']:+.4f}")
                continue
        check(S5, f"Table 5: {name} N={N}", pr, v["diff"], src, note if bool(st) == bool(v["separates"]) else note + "  <-- star / interval disagree")
# Table 6 (BA full)
t6 = {"Best-of-N (argmax)": ("bon_argmax", "argmax", GB, ["0.234", "0.260", "0.289", "0.312", "0.334", "0.354"]),
      "Best-of-N (draw)": ("bon_is", "draw", GB, ["0.234", "0.260", "0.288", "0.309", "0.329", "0.347"]),
      "Chunk pruning": ("greedy_chunk", "argmax", GB, ["0.234", "0.289", "0.342", "0.371", "0.398", "0.415"]),
      "FK steering (dense, draw)": ("fk_noise", "draw", GB, ["0.234", "0.256", "0.316", "0.350", "0.370", "0.388"]),
      "LatticeSMC (draw)": ("lattice_smc_notwist", "draw", GB, ["0.234", "0.258", "0.305", "0.332", "0.366", "0.385"]),
      "LatticeSMC (draw), alpha = 0.05": ("lattice_smc_notwist", "draw", "R_BA alpha 0.05", ["0.234", "0.253", "0.273", "0.280", "0.298", "0.298"]),
      "LatticeSMC (draw), alpha = 0.2": ("lattice_smc_notwist", "draw", "R_BA alpha 0.2", ["0.234", "0.236", "0.254", "0.247", "0.254", "0.251"])}
for name, (m, rule, grid, prs) in t6.items():
    for N, pr in zip((1, 2, 4, 8, 16, 32), prs):
        note = ""
        if "0.05" in name:
            note = f"lattice_smc (learned twist) alpha 0.05 row: {dfull('R_BA alpha 0.05', 'lattice_smc', 'draw', N):.4f}"
        check(S5, f"Table 6: {name} N={N}", pr, dfull(grid, m, rule, N), usrc, note)
for N, pr in ((8, "0.356"), (32, "0.428")):
    check(S5, f"Table 6: LatticeSMC (argmax) N={N} (regenerated)", pr, dfull(GB, "lattice_smc_notwist", "argmax", N), usrc)
check(S5, "Table 6: base", "0.234", dfull(GB, "base", "-", 1), usrc)
# Table 7 (rep)
t7 = {"Best-of-N (argmax)": ("bon_argmax", "argmax", ["0.718", "0.769", "0.805", "0.850", "0.876", "0.901"]),
      "Best-of-N (draw)": ("bon_is", "draw", ["0.718", "0.768", "0.804", "0.849", "0.871", "0.895"]),
      "Chunk pruning": ("greedy_chunk", "argmax", ["0.718", "0.761", "0.792", "0.811", "0.835", "0.887"]),
      "FK steering (dense, draw)": ("fk_noise", "draw", ["0.718", "0.768", "0.836", "0.876", "0.901", "0.921"]),
      "LatticeSMC (draw)": ("lattice_smc_notwist", "draw", ["0.718", "0.768", "0.812", "0.864", "0.892", "0.924"])}
for name, (m, rule, prs) in t7.items():
    for N, pr in zip((1, 2, 4, 8, 16, 32), prs):
        check(S5, f"Table 7: {name} N={N}", pr, dfull(GR, m, rule, N), usrc)
for N, pr in ((8, "0.898"), (32, "0.962")):
    check(S5, f"Table 7: FK steering (dense, argmax) N={N}", pr, dadd(GR, "fk_noise", "argmax", N), usrc + " (logs; exact on R_rep)")
for N, pr in ((8, "0.874"), (32, "0.947")):
    check(S5, f"Table 7: LatticeSMC (argmax) N={N}", pr, dfull(GR, "lattice_smc_notwist", "argmax", N), usrc)
check(S5, "Table 7: base", "0.718", dfull(GR, "base", "-", 1), usrc)
# Table 8 (CLAP)
t8 = {"Best-of-N (argmax)": ("bon_argmax", "argmax", ["0.470", "0.488", "0.502", "0.513", "0.520", "0.530"]),
      "Best-of-N (draw)": ("bon_is", "draw", ["0.470", "0.488", "0.502", "0.513", "0.519", "0.527"]),
      "Chunk pruning": ("greedy_chunk", "argmax", ["0.470", "0.499", "0.523", "0.536", "0.544", "0.551"]),
      "FK steering (dense, draw)": ("fk_noise", "draw", ["0.470", "0.488", "0.520", "0.538", "0.542", "0.546"]),
      "LatticeSMC beta=1 (draw)": ("lattice_smc_prefix_b1", "draw", ["0.470", "0.488", "0.518", "0.534", "0.538", "0.547"]),
      "LatticeSMC beta=0.5 (draw)": ("lattice_smc_prefix_b0.5", "draw", ["0.470", "0.488", "0.513", "0.526", "0.533", "0.544"]),
      "LatticeSMC beta=0.25 (draw)": ("lattice_smc_prefix_b0.25", "draw", ["0.470", "0.488", "0.507", "0.518", "0.525", "0.533"])}
for name, (m, rule, prs) in t8.items():
    for N, pr in zip((1, 2, 4, 8, 16, 32), prs):
        check(S5, f"Table 8: {name} N={N}", pr, mus(m, rule, N)["reward"], "u music_rows")
for m, name, prs in (("fk_noise", "FK steering (dense, argmax)", ("0.545", "0.559")), ("lattice_smc_prefix_b1", "LatticeSMC beta=1 (argmax)", ("0.538", "0.560"))):
    for N, pr in zip((8, 32), prs):
        check(S5, f"Table 8: {name} N={N}", pr, mus(m, "argmax", N)["reward"], "u music_rows (regenerated)")
check(S5, "Table 8: base", "0.470", mus("base", "-", 1)["reward"], "u music_rows")
text(S5, "Table 8 caption: alpha = 0.0068", "0.0068", f"{ALPHA_M2:.6f}", "results/m2/alpha.json", round(ALPHA_M2, 4) == 0.0068)
# Table 9 (motif)
t9 = {"Best-of-N (argmax)": ("bon_argmax", "argmax", ["0.814", "0.849", "0.876", "0.897", "0.910", "0.918"]),
      "Best-of-N (draw)": ("bon_is", "draw", ["0.814", "0.849", "0.874", "0.892", "0.902", "0.904"]),
      "Chunk pruning": ("greedy_chunk", "argmax", ["0.814", "0.848", "0.870", "0.895", "0.906", "0.918"]),
      "FK steering (dense, draw)": ("fk_noise", "draw", ["0.814", "0.849", "0.875", "0.895", "0.898", "0.908"]),
      "FK steering (dense, argmax)": ("fk_noise", "argmax", ["0.814", "0.849", "0.885", "0.910", "0.922", "0.932"]),
      "LatticeSMC beta=1 (draw)": ("lattice_smc_prefix_b1", "draw", ["0.814", "0.849", "0.872", "0.891", "0.900", "0.903"]),
      "LatticeSMC beta=1 (argmax)": ("lattice_smc_prefix_b1", "argmax", ["0.814", "0.849", "0.878", "0.902", "0.918", "0.929"])}
for name, (m, rule, prs) in t9.items():
    for N, pr in zip((1, 2, 4, 8, 16, 32), prs):
        check(S5, f"Table 9: {name} N={N}", pr, mot(m, rule, N)["R_mean"], m4s)
check(S5, "Table 9: base", "0.814", mot("base", "-", 1)["R_mean"], m4s)
text(S5, "Table 9 caption: alpha = 0.019", "0.019", f"{ALPHA_M4:.6f}", "results/m4/alpha.json", round(ALPHA_M4, 3) == 0.019)
# Table 10 (scale)
for lab, pr, val in [("Table 10: base 4M BA", "0.234", erow(GB, "base", "-", 1)), ("Table 10: base 38M BA", "0.225", lrow(GB, "base", "-", 1)), ("Table 10: base 4M rep", "0.718", erow(GR, "base", "-", 1)), ("Table 10: base 38M rep", "0.654", lrow(GR, "base", "-", 1)),
                     ("Table 10: best-of-N 4M BA draw", "0.347", erow(GB, "bon_is", "draw", 32)), ("Table 10: best-of-N 4M BA argmax", "0.354", erow(GB, "bon_argmax", "argmax", 32)), ("Table 10: best-of-N 38M BA draw", "0.353", lrow(GB, "bon_is", "draw", 32)), ("Table 10: best-of-N 38M BA argmax", "0.360", lrow(GB, "bon_argmax", "argmax", 32)),
                     ("Table 10: best-of-N 4M rep draw", "0.895", erow(GR, "bon_is", "draw", 32)), ("Table 10: best-of-N 4M rep argmax", "0.901", erow(GR, "bon_argmax", "argmax", 32)), ("Table 10: best-of-N 38M rep draw", "0.780", lrow(GR, "bon_is", "draw", 32)), ("Table 10: best-of-N 38M rep argmax", "0.784", lrow(GR, "bon_argmax", "argmax", 32)),
                     ("Table 10: pruning 4M BA", "0.415", erow(GB, "greedy_chunk", "argmax", 32)), ("Table 10: pruning 38M BA", "0.423", lrow(GB, "greedy_chunk", "argmax", 32)), ("Table 10: pruning 4M rep", "0.887", erow(GR, "greedy_chunk", "argmax", 32)), ("Table 10: pruning 38M rep", "0.795", lrow(GR, "greedy_chunk", "argmax", 32)),
                     ("Table 10: FK 4M BA draw", "0.388", erow(GB, "fk_noise", "draw", 32)), ("Table 10: FK 4M BA argmax (dagger)", "0.438", erow(GB, "fk_noise", "argmax", 32, "reward_additive")), ("Table 10: FK 38M BA draw", "0.406", lrow(GB, "fk_noise", "draw", 32)), ("Table 10: FK 38M BA argmax (dagger)", "0.450", lrow(GB, "fk_noise", "argmax", 32, "reward_additive")),
                     ("Table 10: FK 4M rep draw", "0.921", erow(GR, "fk_noise", "draw", 32)), ("Table 10: FK 4M rep argmax", "0.962", erow(GR, "fk_noise", "argmax", 32, "reward_additive")), ("Table 10: FK 38M rep draw", "0.822", lrow(GR, "fk_noise", "draw", 32)), ("Table 10: FK 38M rep argmax", "0.874", lrow(GR, "fk_noise", "argmax", 32, "reward_additive")),
                     ("Table 10: LatticeSMC 4M BA draw", "0.385", erow(GB, "lattice_smc_notwist", "draw", 32)), ("Table 10: LatticeSMC 4M BA argmax", "0.428", erow(GB, "lattice_smc_notwist", "argmax", 32)), ("Table 10: LatticeSMC 38M BA draw", "0.394", lrow(GB, "lattice_smc_notwist", "draw", 32)), ("Table 10: LatticeSMC 38M BA argmax", "0.438", lrow(GB, "lattice_smc_notwist", "argmax", 32)),
                     ("Table 10: LatticeSMC 4M rep draw", "0.924", erow(GR, "lattice_smc_notwist", "draw", 32)), ("Table 10: LatticeSMC 4M rep argmax", "0.947", erow(GR, "lattice_smc_notwist", "argmax", 32)), ("Table 10: LatticeSMC 38M rep draw", "0.798", lrow(GR, "lattice_smc_notwist", "draw", 32)), ("Table 10: LatticeSMC 38M rep argmax", "0.823", lrow(GR, "lattice_smc_notwist", "argmax", 32))]:
    check(S5, lab, pr, val, xsrc + " rows")
for lab, pr, grid, model in [("Table 10: argmax - pruning 4M BA", "+0.013", GB, "E (4.1 M)"), ("Table 10: argmax - pruning 38M BA", "+0.015", GB, "L (37.6 M)"), ("Table 10: argmax - pruning 4M rep", "+0.059", GR, "E (4.1 M)"), ("Table 10: argmax - pruning 38M rep", "+0.028", GR, "L (37.6 M)")]:
    p = xpair(grid, model, "lattice_smc_notwist-argmax - greedy_chunk", "full (regenerated)", 32); check(S5, lab, pr, p["diff"], xsrc + " pairs", f"CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]")
text(S5, "Table 10 caption: orderings at N = 32 on beat alignment identical at both sizes", "identical", f"{X[GB]['ordering_N32']['draw|full']['pairwise_sign_agreement']} / {X[GB]['ordering_N32']['argmax|additive']['pairwise_sign_agreement']} pairwise agreements (the 38M grid has no learned-twist row, so 'identical' holds on the five common methods)", xsrc + " ordering_N32", True)
# Table 11 (held-out)
hb = P2A["heldout"]["32"][0]
check(S5, "Table 11: base PFC", "1.97", hb["pfc"]["mean"], "phase2b_a002 heldout base"); check(S5, "Table 11: base W1", "0.0080", hb["realism_w1"]["mean"], "phase2b_a002 heldout base")
check(S5, "Table 11: base dance diversity", "0.183", next(d for d in P2A["diversity"] if d["method"] == "base")["pairwise_joint_dist_root_relative_m"], "phase2b_a002 diversity")
mb = mus("base", "-", 1)
check(S5, "Table 11: base tempo", "0.903", mb["tempo_reward"], "u music_rows"); check(S5, "Table 11: base 8 kHz", "0.0069", mb["frac_energy_above_8k"], "u music_rows"); check(S5, "Table 11: base PQ", "8.17", mb["aes_PQ"], "u music_rows"); check(S5, "Table 11: base seam", "1.27", mb["seam_ratio"], "u music_rows"); check(S5, "Table 11: base music diversity", "0.061", mb["diversity"], "u music_rows")
h32 = [h for h in P2A["heldout"]["32"] if h["method"] != "base"]
ar = dance(GB, "lattice_smc_notwist", "argmax", 32)
pf = [h["pfc"]["mean"] for h in h32] + [ar["pfc"]]; w1 = [h["realism_w1"]["mean"] for h in h32] + [ar["realism_w1"]]
dv = [d["pairwise_joint_dist_root_relative_m"] for d in P2A["diversity"] if d["N"] == 32 and d["method"] != "base"] + [ar["diversity_root_rel"]]
text(S5, "Table 11: steered PFC range 1.9-2.3 (N = 32, BA grid incl. argmax row)", "1.9-2.3", f"{min(pf):.2f}-{max(pf):.2f} (per method: " + ", ".join(f"{h['method']} {h['pfc']['mean']:.2f}" for h in h32) + f", notwist-argmax {ar['pfc']:.2f})", "phase2b_a002 heldout N=32; u dance_rows", round(min(pf), 1) == 1.9 and round(max(pf), 1) == 2.3, "lower end is 1.74 (lattice_smc, learned twist) or 1.84 without it")
text(S5, "Table 11: steered W1 range 0.0077-0.0086", "0.0077-0.0086", f"{min(w1):.4f}-{max(w1):.4f}", "phase2b_a002 heldout N=32; u dance_rows", round(min(w1), 4) == 0.0077 and round(max(w1), 4) == 0.0086, "0.0086 occurs only in the alpha 0.2 grid (bon_is N = 32: 0.00864), not at alpha 0.02")
text(S5, "Table 11: steered dance diversity range 0.171-0.191", "0.171-0.191", f"{min(dv):.3f}-{max(dv):.3f} at N = 32, alpha 0.02", "phase2b_a002 diversity N=32; u dance_rows", round(min(dv), 3) == 0.171 and round(max(dv), 3) == 0.191, "0.171 and 0.191 occur only in the alpha 0.2 grid (bon_is N = 32: 0.1707; lattice N = 8: 0.1906)")
m32d = [r for r in U["music_rows"] if r["N"] == 32 and r.get("tempo_reward") == r.get("tempo_reward") and (r["rule"] != "argmax" or r["method"] in ("bon_argmax", "greedy_chunk"))]
m32a = [r for r in U["music_rows"] if r["N"] == 32 and r["rule"] == "argmax" and r["method"] in ("fk_noise", "lattice_smc_prefix_b1") and r.get("tempo_reward") == r.get("tempo_reward")]
for key, pr, dd in (("tempo_reward", "0.89-0.94", 2), ("frac_energy_above_8k", "0.0057-0.0074", 4), ("aes_PQ", "8.12-8.17", 2), ("seam_ratio", "1.25-1.34", 2), ("diversity", "0.052-0.060", 3)):
    vd = [r[key] for r in m32d if r.get(key) == r.get(key)]
    va = vd + [r[key] for r in m32a if r.get(key) == r.get(key)]
    lo_p, hi_p = (float(x) for x in pr.split("-"))
    ok_d = round(min(vd), dd) == lo_p and round(max(vd), dd) == hi_p
    ok_a = round(min(va), dd) == lo_p and round(max(va), dd) == hi_p
    text(S5, f"Table 11: steered music {key} range", pr, f"draw rows (incl. best-of-N argmax and pruning) {min(vd):.4f}-{max(vd):.4f}; with the regenerated argmax rows {min(va):.4f}-{max(va):.4f} (N = 32, prompt-adherence grid)", "u music_rows", ok_d or ok_a,
         "" if (ok_d or ok_a) else {"frac_energy_above_8k": "0.0057 is the motif-grid fk_noise N = 32 value and 0.0074 the N = 8 beta 0.5 value of this grid; neither is an N = 32 prompt-adherence value", "aes_PQ": "8.17 is the base value; the steered N = 32 rows top out at 8.16"}.get(key, ""))
for key, pr in (("pfc", "2.24"), ("realism_w1", "0.0082"), ("diversity_root_rel", "0.183")):
    check(S5, f"Table 11: LatticeSMC argmax dance {key}", pr, ar[key], "u dance_rows (regenerated, N = 32)")
la = mus("lattice_smc_prefix_b1", "argmax", 32)
for key, pr in (("tempo_reward", "0.92"), ("frac_energy_above_8k", "0.0062"), ("aes_PQ", "8.13"), ("seam_ratio", "1.34"), ("diversity", "0.050")):
    check(S5, f"Table 11: LatticeSMC argmax music {key}", pr, la[key], "u music_rows (regenerated, N = 32)")

# ================================================================================================= write
def md_table(rows):
    out = ["| location in draft | value printed | value in source file | source path | match |", "|---|---|---|---|---|"]
    for sec, loc, pr, val, src, ok, note in rows:
        out.append(f"| {loc} | {pr} | {val}{(' (' + note + ')') if note else ''} | {src} | {ok} |")
    return "\n".join(out)


if __name__ == "__main__":
    extra = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    n = len(ROWS); mism = [r for r in ROWS if r[5] == "NO"]; rnd = [r for r in ROWS if r[5] == "rounding"]
    md = [f"# VERIFICATION of paper/LatticeSMC_ICLR2027 against the stored results ({extra.get('date', '2026-09-18')})", "",
          "Draft treated as untrusted; analysis files under `results/` as ground truth. No number in the draft was edited. Every checked value is listed "
          "with its source; 'match' compares at the printed precision (a value whose rounding at that precision equals the printed one is 'yes').", "",
          "## Summary", "", f"- Values checked: **{n}** (Section 2 main text: {sum(1 for r in ROWS if r[0].startswith('2.'))}; Section 3 appendix: {sum(1 for r in ROWS if r[0].startswith('3.'))}).",
          f"- Mismatches or unverifiable claims: **{len(mism)}**, listed below by draft location (printed value vs value in the files).",
          f"- Rounding-only differences (under one unit of the last printed digit; the draft rounded the 4-decimal RESULTS.md value half-up where the analysis file's 5th decimal falls just below the half): **{len(rnd)}**, listed after the mismatches.", ""]
    for sec, loc, pr, val, src, ok, note in mism:
        md.append(f"  - **{loc}**: printed {pr}; files give {val}{(' — ' + note) if note else ''} [{src}]")
    md.append("")
    md.append("Rounding-only:")
    for sec, loc, pr, val, src, ok, note in rnd:
        md.append(f"  - {loc}: printed {pr}; file value {val} [{src}]")
    md += ["", extra.get("summary_extra", ""), "", "## 1. Build", "", extra.get("build", ""), "", "## 2. Numbers in the main text (Sections 4-5, captions of Figures 1-3 and Table 1)", "",
           md_table([r for r in ROWS if r[0].startswith("2.")]), "", "## 3. Numbers in the appendix (Tables 3-11, B.2-B.5, A.5, C.1-C.3)", "", md_table([r for r in ROWS if r[0].startswith("3.")]), "",
           "## 4. Dagger cells (FK steering argmax at N = 32, full beat-alignment reward)", "", extra.get("dagger", ""), "", "## 5. Bibliography", "", extra.get("bib", ""), "",
           "## 6. Figures", "", extra.get("figs", ""), "", "## 7. Consistency (labels, references, unreferenced floats)", "", extra.get("consistency", "")]
    open(OUT, "w").write("\n".join(md) + "\n")
    print(f"checked {n}, mismatches {len(mism)}")
    for r in mism:
        print("  NO:", r[1], "|", r[2], "|", r[3][:120])
