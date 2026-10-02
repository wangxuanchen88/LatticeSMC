#!/bin/bash
# R1 Phase 3 (music) qualification: base samples at K = 8 (rolling 40 s context) on the M2 qualification prompts
# (ids 4 8 9 12 16 21 25 30 33 35, seeds 2000 2001), seam spectral-flux ratio over the seven seams and whole-clip tempo
# adherence, compared with the K = 4 base samples of the same prompts (results/m3 scores.json / tempo.json). Gate: stop if
# the seam ratio rises or tempo adherence falls by more than 10 percent. Usage: scripts/r1_music_qual.sh <gpu>
cd ${LATTICESMC_ROOT:-.}
GPU=${1:-1}; PY=.venv/bin/python; SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
OUT=${LATTICESMC_ROOT:-.}/results/r1/qual_k8
(cd $SA3 && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/music_lattice_k8.py --runs base --Ns 1 --seeds 2000,2001 --prompts 4,8,9,12,16,21,25,30,33,35 --out_root $OUT) > logs/r1_qual_k8_gen.log 2>&1
(cd $SA3 && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/score_samples.py --samples $OUT --out $OUT/scores.json) > logs/r1_qual_k8_scores.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU m0/.venv/bin/python m3/score_tempo.py --samples $OUT --out $OUT/tempo.json --Ns 1 --device cuda > logs/r1_qual_k8_tempo.log 2>&1
$PY scripts/r1_music_qual_compare.py > logs/r1_qual_k8_compare.log 2>&1
cat logs/r1_qual_k8_compare.log
