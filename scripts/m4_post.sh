#!/bin/bash
# Phase M4 post-processing: root-of-trust recompute (R_motif, terminal CLAP held out, 8 kHz, seam), tempo, aesthetics, analysis.
set -e
cd ${LATTICESMC_ROOT:-.}
SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
(cd $SA3 && CUDA_VISIBLE_DEVICES=1 HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/score_samples.py \
  --samples ${LATTICESMC_ROOT:-.}/results/m4/samples --out ${LATTICESMC_ROOT:-.}/results/m4/scores.json) > logs/m4_scores.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 m0/.venv/bin/python m3/score_tempo.py --samples results/m4/samples --out results/m4/tempo.json --Ns 8,32 --device cuda > logs/m4_tempo.log 2>&1 &
CUDA_VISIBLE_DEVICES="" m2/.venv-aes/bin/python m3/score_aes.py --samples results/m4/samples --out results/m4/aesthetics.json --Ns 8,32 > logs/m4_aes.log 2>&1 &
wait
.venv/bin/python m3/analyze_music.py --samples results/m4/samples --out results/m4 --reward motif --tag m4 > logs/m4_analyze.log 2>&1
echo "M4 post done"
