"""R1 Phase 3 (music) qualification gate: K = 8 rolling-context base samples vs the K = 4 base samples of the same 10 prompts
x 2 seeds. Seam ratio (mean over the clip's seams of seam flux / median within-chunk flux, m1/rewards.seam_stats as used by
m3/score_samples.py) and whole-clip tempo adherence (fraction of clips whose tracked tempo is within 5 % of the prompt's;
m3/score_tempo.py). Passes if the seam ratio does not rise by more than 10 % and tempo adherence does not fall by more
than 10 % relative to K = 4. Writes results/r1/qual_k8/qualification.json."""
import json
import os

import numpy as np

ROOT = os.environ.get("LATTICESMC_ROOT", ".")
Q = os.path.join(ROOT, "results", "r1", "qual_k8")
ids = [4, 8, 9, 12, 16, 21, 25, 30, 33, 35]
keys = [f"base/1/p{i:02d}_s{s}" for i in ids for s in (2000, 2001)]
s8 = {r["key"]: r for r in json.load(open(os.path.join(Q, "scores.json")))["rows"]}
t8 = {r["key"]: r for r in json.load(open(os.path.join(Q, "tempo.json")))["rows"]}
s4 = {r["key"]: r for r in json.load(open(os.path.join(ROOT, "results", "m3", "scores.json")))["rows"]}
t4 = {r["key"]: r for r in json.load(open(os.path.join(ROOT, "results", "m3", "tempo.json")))["rows"]}
seam8 = [s8[k]["seam_mean_ratio"] for k in keys if k in s8]
seam4 = [s4[k]["seam_mean_ratio"] for k in keys if k in s4]
tf = "whole_within_5pct"
tempo8 = [t8[k][tf] for k in keys if k in t8 and t8[k].get(tf) is not None]
tempo4 = [t4[k][tf] for k in keys if k in t4 and t4[k].get(tf) is not None]
per_seam8 = [s["ratio_to_median"] for k in keys if k in s8 for s in s8[k].get("seams", [])]
out = {"prompts": ids, "seeds": [2000, 2001], "n_k8": len(seam8), "n_k4": len(seam4),
       "seam_ratio_k8_mean": float(np.mean(seam8)), "seam_ratio_k4_mean": float(np.mean(seam4)), "seam_ratio_rel_change": float(np.mean(seam8) / np.mean(seam4) - 1),
       "seam_ratio_k8_per_seam_mean": [float(np.mean([s8[k]["seams"][j]["ratio_to_median"] for k in keys if k in s8])) for j in range(len(s8[keys[0]]["seams"]))] if s8.get(keys[0], {}).get("seams") else None,
       "tempo_within5_k8": float(np.mean(tempo8)) if tempo8 else None, "tempo_within5_k4": float(np.mean(tempo4)) if tempo4 else None,
       "tempo_rel_change": (float(np.mean(tempo8) / np.mean(tempo4) - 1) if tempo8 and tempo4 and np.mean(tempo4) > 0 else None),
       "clap_terminal_k8_mean": float(np.mean([s8[k]["clap_terminal_recomputed"] for k in keys if k in s8])), "clap_terminal_k4_mean": float(np.mean([s4[k]["clap_terminal_recomputed"] for k in keys if k in s4])),
       "frac_energy_above_8k_k8": float(np.mean([s8[k]["frac_energy_above_8k"] for k in keys if k in s8])), "frac_energy_above_8k_k4": float(np.mean([s4[k]["frac_energy_above_8k"] for k in keys if k in s4]))}
out["pass"] = bool(out["seam_ratio_rel_change"] <= 0.10 and (out["tempo_rel_change"] is None or out["tempo_rel_change"] >= -0.10))
json.dump(out, open(os.path.join(Q, "qualification.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
print("QUAL_PASS" if out["pass"] else "QUAL_FAIL")
