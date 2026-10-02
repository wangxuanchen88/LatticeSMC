#!/bin/bash
# Phase 1c per-candidate chain (Amendment G): adopt the checkpoint with the lowest held-out monitor
# loss, generate the three sample roots on both GPUs, run the SPEC 2.5 checks.
#   scripts/phase1c_eval.sh chunk_dispE E
set -e
cd "$(dirname "$0")/.."
RUN=$1; TAG=$2; PY=.venv/bin/python
$PY scripts/pick_ckpt.py --run runs/$RUN
CUDA_VISIBLE_DEVICES=0 $PY -m lattice_smc.generate --ckpt runs/$RUN/ckpt_adopted.pt --method base --N 1 --prompts 2 --seeds 2000 --out_root /tmp/latticesmc_scratch/smoke_$TAG 2>&1 | grep -v Warning | tail -2
scripts/phase1_generate.sh runs/$RUN/ckpt_adopted.pt samples_$TAG
CUDA_VISIBLE_DEVICES=0 $PY -m lattice_smc.analyze --phase1 --samples samples_$TAG --compare samples_${TAG}_rerun,samples_${TAG}_swap --tag phase1c_${TAG}_checks 2>&1 | grep -v Warning > logs/analyze_$TAG.log
$PY - <<PYEOF
import json; d = json.load(open("results/phase1c_${TAG}_checks.json")); b = d["boundary"]
print("$TAG ckpt step", d["ckpt_step"], "ratio %.3f %s" % (b["ratio_pooled_mean_seam_over_median_within"], "PASS" if b["pass"] else "FAIL"),
      "root %.2f pose %.2f" % (b["root_translation"]["ratio"], b["root_relative_pose"]["ratio"]), "determinism", [(r["n_bitwise_equal"], r["n_compared"]) for r in d["determinism"]], "nfe", d["nfe"]["all_equal_expected"], "oob", d["root_out_of_range"])
PYEOF
