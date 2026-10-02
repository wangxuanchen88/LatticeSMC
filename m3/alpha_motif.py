"""Amendment V: R_motif on the M2 base samples (160) and their item-4 continuations (320): alpha by the Amendment T rule
(mean over prompts of p90 - p10 of R_motif over that prompt's values, / ln 1000) and the ground-truth-free reference
(distribution of R_motif on the base samples; fraction of base clips whose chunk 4 is more similar to its own chunk 1 than
to a random other prompt's chunk 1, default_rng(0)). Written before any steered sample exists. SA3 venv, GPU.

  .venv/bin/python m3/alpha_motif.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1")
import rewards as R  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
OUT = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m4"
os.makedirs(OUT, exist_ok=True)
clap = R.ClapScorer(device="cuda")


def windows(path):
    return clap.window_embeds(R.load_mono48k(path), 4)  # [4, D] unit rows


base = {}
for d in sorted(glob.glob(os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/b2/samples/p*_s*")):
    log = json.load(open(os.path.join(d, "log.json")))
    base[(log["prompt_id"], log["base_seed"])] = windows(os.path.join(d, "seq_chunk03.wav"))
cont = {}
for f in sorted(glob.glob(os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/b2/variance/wavs/*.wav")):
    name = os.path.basename(f)[:-4]  # p04_s2001_k3_m2
    pid, seed = int(name[1:3]), int(name.split("_s")[1][:4])
    cont[name] = (pid, seed, windows(f))
motif = lambda E: float(E[3] @ E[0])  # noqa: E731
vals = {}
for (pid, seed), E in base.items():
    vals.setdefault(pid, []).append(motif(E))
n_cont = 0
for name, (pid, seed, E) in cont.items():
    vals[pid].append(motif(E))
    n_cont += 1
spreads = {pid: float(np.percentile(v, 90) - np.percentile(v, 10)) for pid, v in vals.items()}
alpha = float(np.mean(list(spreads.values())) / np.log(1000))
base_vals = np.array([motif(E) for E in base.values()])
# reference: own chunk 4 vs own chunk 1 against a random other prompt's chunk 1 (same seed), rng(0)
rng = np.random.default_rng(0)
keys = sorted(base)
wins = []
for (pid, seed) in keys:
    others = [k for k in keys if k[0] != pid and k[1] == seed]
    o = others[rng.integers(len(others))]
    wins.append(float(base[(pid, seed)][3] @ base[(pid, seed)][0]) > float(base[(pid, seed)][3] @ base[o][0]))
# also chunk 3 to chunk 1 (the prefix-score variant) on the base samples
s3 = np.array([float(E[2] @ E[0]) for E in base.values()])
ref = {"n_base": len(base), "n_continuations": n_cont, "base_R_motif_mean": float(base_vals.mean()), "base_R_motif_sd": float(base_vals.std(ddof=1)),
       "base_R_motif_quantiles_10_50_90": [float(np.percentile(base_vals, q)) for q in (10, 50, 90)], "base_R_motif_min_max": [float(base_vals.min()), float(base_vals.max())],
       "frac_chunk4_closer_to_own_chunk1_than_random_other_prompt": float(np.mean(wins)), "n_pairs": len(wins),
       "base_s3_chunk3_to_chunk1_mean_sd": [float(s3.mean()), float(s3.std(ddof=1))], "corr_s3_R_motif_base": float(np.corrcoef(s3, base_vals)[0, 1]),
       "across_seed_sd_R_motif": float(np.mean([np.std([motif(base[(p, s)]) for s in (2000, 2001, 2002, 2003)], ddof=1) for p in vals])),
       "across_prompt_sd_R_motif": float(np.std([np.mean([motif(base[(p, s)]) for s in (2000, 2001, 2002, 2003)]) for p in vals], ddof=1))}
json.dump({"rule": "alpha = mean over prompts of (p90 - p10 of R_motif over the prompt's 4 base samples and its item-4 continuations) / ln(1000)",
           "reward": "R_motif = cos(CLAP audio embedding of window 4, window 1) of the 40 s clip; laion_clap music_audioset_epoch_15_esc_90.14.pt, TF32 off",
           "n_total_values": len(base) + n_cont, "mean_spread_p90_minus_p10": float(np.mean(list(spreads.values()))), "alpha": alpha, "per_prompt_spread": spreads,
           "no_steered_sample_exists": True}, open(os.path.join(OUT, "alpha.json"), "w"), indent=1)
json.dump(ref, open(os.path.join(OUT, "motif_reference.json"), "w"), indent=1)
print(json.dumps({"alpha": alpha, "mean_spread": float(np.mean(list(spreads.values()))), **ref}, indent=1))
