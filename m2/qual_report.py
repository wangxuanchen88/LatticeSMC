"""Apply the Amendment Q/S selection rule, criterion by criterion, to one harness."""
import json, sys
import numpy as np

H = sys.argv[1] if len(sys.argv) > 1 else "b2"
R = fos.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/{H}"
sc = json.load(open(f"{R}/qual_scores.json"))
tp = json.load(open(f"{R}/qual_tempo.json"))
det = json.load(open(f"{R}/det_compare.json"))
rows = sc["rows"]
rids = sc["retrieval_prompt_ids"]

# --- (i) determinism
det_ok = det["latents_all_bitwise_equal"]

# --- (ii) latent prefix bit-exact
import os
lat = []
for r in rows:
    log = json.load(open(f"{R}/samples/{r['dir']}/log.json"))
    lat += log["latent_prefix"]
lat_ok = all(x["latent_prefix_bitexact"] for x in lat)

# --- (iii) seam
seam = float(np.mean([r["seam_mean_ratio"] for r in rows]))
seam_sd = float(np.std([r["seam_mean_ratio"] for r in rows], ddof=1))
seam_lm = float(np.mean([r["seam_mean_ratio_local_max"] for r in rows]))
ctrl = float(np.mean([r["control_mean_ratio"] for r in rows]))
real = sc["real_music_control"]["seam_like_ratio_at_10_20_30s"]

# --- (iv) tempo
tw = [r["adh_within_5pct"] for r in tp["rows"]]
two = [r["adh_within_5pct_octave_corrected"] for r in tp["rows"]]
rel = [r["adh_rel_err"] for r in tp["rows"]]
cv = [r["bt_beat_interval_cv"] for r in tp["rows"]]
tempo_frac = float(np.mean(tw))

# --- (v) CLAP R@1 per seed and mean
per_seed = {}
for s in sorted({r["base_seed"] for r in rows}):
    rr = [r for r in rows if r["base_seed"] == s]
    hit = [int(int(np.argmax(r["cos_vs_retrieval_texts"])) == r["matched_index"]) for r in rr]
    rank = [int(1 + np.sum(np.array(r["cos_vs_retrieval_texts"]) >
                           r["cos_vs_retrieval_texts"][r["matched_index"]])) for r in rr]
    per_seed[s] = {"n": len(rr), "R@1": float(np.mean(hit)),
                   "R@3": float(np.mean([x <= 3 for x in rank])),
                   "median_rank": float(np.median(rank))}
r1 = float(np.mean([v["R@1"] for v in per_seed.values()]))
matched = [r["cos_vs_retrieval_texts"][r["matched_index"]] for r in rows]
unmatched = [np.mean([c for j, c in enumerate(r["cos_vs_retrieval_texts"])
                      if j != r["matched_index"]]) for r in rows]
# genre-level R@1
allp = {p["id"]: p for p in json.load(open(os.environ.get("LATTICESMC_ROOT", ".") + "/m1/prompts.json"))}
gen_hit = [int(allp[rids[int(np.argmax(r["cos_vs_retrieval_texts"]))]]["genre"] == r["genre"]) for r in rows]

hf = float(np.mean([r["frac_energy_above_8k"] for r in rows]))

out = {
  "harness": H, "n_sequences": len(rows), "retrieval_prompt_ids": rids,
  "i_deterministic_latents_bitwise": det_ok,
  "i_deterministic_wavs_bitwise": det["wavs_all_bitwise_equal"],
  "i_n_items": det["n_items"],
  "ii_latent_prefix_bitexact_all": lat_ok, "ii_n_latent_comparisons": len(lat),
  "ii_max_abs_diff": max(x["latent_prefix_max_abs_diff"] for x in lat),
  "iii_seam_mean_ratio": seam, "iii_seam_sd": seam_sd,
  "iii_seam_local_max": seam_lm, "iii_control_mean_ratio": ctrl,
  "iii_real_music_control": real,
  "iv_tempo_within_5pct": tempo_frac,
  "iv_tempo_within_5pct_octave": float(np.mean(two)),
  "iv_median_rel_err": float(np.median(rel)),
  "iv_median_beat_interval_cv": float(np.median(cv)),
  "iv_real_music_tempo": tp["real_music_control"]["tempo_bpm_median_interval"],
  "iv_real_music_cv": tp["real_music_control"]["beat_interval_cv"],
  "v_R@1_per_seed": per_seed, "v_R@1_mean": r1,
  "v_R@3_mean": float(np.mean([v["R@3"] for v in per_seed.values()])),
  "v_matched_mean_cos": float(np.mean(matched)),
  "v_unmatched_mean_cos": float(np.mean(unmatched)),
  "v_matched_minus_unmatched": float(np.mean(matched) - np.mean(unmatched)),
  "v_genre_R@1": float(np.mean(gen_hit)),
  "aux_frac_energy_above_8k": hf,
  "aux_real_music_frac_energy_above_8k": sc["real_music_control"]["frac_energy_above_8k"],
  "aux_nfe_per_chunk": sorted({n for r in rows for n in r["per_chunk_nfe"]}),
  "aux_nfe_per_sequence": sorted({r["nfe_total"] for r in rows}),
  "aux_decode_calls_per_sequence": sorted({r["decode_calls_total"] for r in rows}),
  "aux_wall_per_sequence_mean": float(np.mean([r["seq_wall_s"] for r in rows])),
}
crit = {
  "i_deterministic": bool(det_ok),
  "ii_latent_prefix_bitexact": bool(lat_ok),
  "iii_seam_lt_1.3": bool(seam < 1.3),
  "iv_tempo_ge_0.60": bool(tempo_frac >= 0.60),
  "v_clap_r1_ge_0.30": bool(r1 >= 0.30),
}
out["criteria"] = crit
out["qualifies"] = all(crit.values())
json.dump(out, open(f"{R}/qualification.json", "w"), indent=1)
print(json.dumps(out, indent=1))
