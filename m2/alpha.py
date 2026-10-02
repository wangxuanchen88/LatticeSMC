"""Amendment S item 5 -- the alpha rule, computed BEFORE any steered sample exists.

alpha = mean over the 40 prompts of [ p90 - p10 of terminal CLAP over that prompt's
        base samples (4 seeds) and their continuations from item 4 ] / ln(1000)

Only prompts that have item-4 continuations contribute those continuations; every
prompt contributes its 4 base terminal CLAPs.  Percentiles are numpy's default
linear interpolation.  Nothing steered exists at the time this is written.
"""
import os
import json, os
from collections import defaultdict
import numpy as np

R = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/b2"
base = json.load(open(f"{R}/scores.json"))["rows"]
var_p = f"{R}/variance/results.json"
var = json.load(open(var_p))["rows"] if os.path.exists(var_p) else []

vals = defaultdict(list)
prov = defaultdict(lambda: {"base": 0, "continuations": 0})
for r in base:
    vals[r["prompt_id"]].append(float(r["clap_terminal"]))
    prov[r["prompt_id"]]["base"] += 1
for r in var:
    vals[r["prompt_id"]].append(float(r["clap_terminal"]))
    prov[r["prompt_id"]]["continuations"] += 1

pids = sorted(vals)
per = []
for p in pids:
    v = np.asarray(vals[p])
    per.append({"prompt_id": p, "n_values": int(len(v)),
                "n_base": prov[p]["base"], "n_continuations": prov[p]["continuations"],
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                "spread_p90_minus_p10": float(np.percentile(v, 90) - np.percentile(v, 10)),
                "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())})
spreads = np.array([x["spread_p90_minus_p10"] for x in per])
mean_spread = float(spreads.mean())
alpha = mean_spread / float(np.log(1000.0))

out = {
    "rule": "alpha = mean_prompts(p90 - p10 of terminal CLAP over that prompt's base "
            "samples (4 seeds) and its item-4 continuations) / ln(1000)",
    "reward": "terminal CLAP of the full 40 s raw-decode clip, LAION laion_clap "
              "music_audioset_epoch_15_esc_90.14.pt, m1/rewards.py protocol",
    "harness": "b2 = Stable Audio 3 medium + caller-side replacement inpainting",
    "n_prompts": len(pids),
    "n_base_values": sum(x["n_base"] for x in per),
    "n_continuation_values": sum(x["n_continuations"] for x in per),
    "n_total_values": int(sum(x["n_values"] for x in per)),
    "prompts_with_continuations": [x["prompt_id"] for x in per if x["n_continuations"] > 0],
    "n_prompts_with_continuations": sum(1 for x in per if x["n_continuations"] > 0),
    "values_per_prompt_with_continuations": sorted({x["n_values"] for x in per if x["n_continuations"] > 0}),
    "values_per_prompt_without_continuations": sorted({x["n_values"] for x in per if x["n_continuations"] == 0}),
    "percentile_method": "numpy.percentile, linear interpolation",
    "mean_spread_p90_minus_p10": mean_spread,
    "sd_of_spread_over_prompts": float(spreads.std(ddof=1)),
    "min_spread": float(spreads.min()), "max_spread": float(spreads.max()),
    "ln_1000": float(np.log(1000.0)),
    "alpha": alpha,
    "no_steered_sample_exists": True,
    "note": "No steered / resampled / pruned sample exists at the time this file was "
            "written; alpha is computed from base samples and item-4 continuations only.",
    "per_prompt": per,
}
json.dump(out, open(os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/alpha.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_prompt"}, indent=1))
