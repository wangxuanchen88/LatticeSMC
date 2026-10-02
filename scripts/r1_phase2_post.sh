#!/bin/bash
# R1 Phase 2 post-processing: dance metrics for the four pruning roots (GPU), music held-out scoring of the four pruning
# roots (SA3 venv on GPU, tempo in the m0 venv, aesthetics on CPU), then the analysis.
cd ${LATTICESMC_ROOT:-.}
PY=.venv/bin/python; SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
until grep -q "phase 2 generation done" logs/r1_phase2.log; do sleep 120; done
for c in ba_M1 ba_M16 rep_M1 rep_M16; do
  CUDA_VISIBLE_DEVICES=0 $PY -m lattice_smc.analyze_phase2 --samples samples_r1_prune_$c --tag r1_prune_$c --metrics_only > logs/r1_prune_${c}_metrics.log 2>&1
done
for r in clap_M1 clap_MN2 motif_M1 motif_MN2; do
  (cd $SA3 && CUDA_VISIBLE_DEVICES=1 HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/score_samples.py --samples ${LATTICESMC_ROOT:-.}/results/r1/samples_prune_$r --out ${LATTICESMC_ROOT:-.}/results/r1/prune_${r}_scores.json) > logs/r1_prune_${r}_scores.log 2>&1
  CUDA_VISIBLE_DEVICES=1 m0/.venv/bin/python m3/score_tempo.py --samples results/r1/samples_prune_$r --out results/r1/prune_${r}_tempo.json --Ns 32 --device cuda > logs/r1_prune_${r}_tempo.log 2>&1
  CUDA_VISIBLE_DEVICES="" m2/.venv-aes/bin/python m3/score_aes.py --samples results/r1/samples_prune_$r --out results/r1/prune_${r}_aesthetics.json --Ns 32 > logs/r1_prune_${r}_aes.log 2>&1
done
$PY scripts/r1_phase2_analysis.py > logs/r1_phase2_analysis.log 2>&1
echo "phase 2 post done $(date)"
