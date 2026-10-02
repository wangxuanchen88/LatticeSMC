#!/bin/bash
# R1 Phase 2 music rerun (2026-09-20): the 2026-09-19 runs never applied --prune_M (fixed); M = 1 and N/2 at N = 32.
cd ${LATTICESMC_ROOT:-.}
PY=.venv/bin/python; SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
music() { local RW=$2 M=$3 MT=${3/\//}; local AF=results/m2/alpha.json; [ $RW = motif ] && AF=results/m4/alpha.json
  (cd $SA3 && CUDA_VISIBLE_DEVICES=$1 HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/music_lattice.py --reward $RW --alpha_file ${LATTICESMC_ROOT:-.}/$AF \
    --runs greedy_chunk --Ns 32 --seeds 2000,2001 --prune_M $M --out_root ${LATTICESMC_ROOT:-.}/results/r1/samples_prune_${RW}_M${MT}) > logs/r1_prune_${RW}_M${MT}_rerun.log 2>&1; }
echo "music rerun start $(date)"
( music 1 clap 1; music 1 clap N/2; echo "clap done $(date)" ) &
( music 0 motif 1; music 0 motif N/2; echo "motif done $(date)" ) &
wait
echo "music rerun generation done $(date)"
for r in clap_M1 clap_MN2 motif_M1 motif_MN2; do
  (cd $SA3 && CUDA_VISIBLE_DEVICES=1 HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/score_samples.py --samples ${LATTICESMC_ROOT:-.}/results/r1/samples_prune_$r --out ${LATTICESMC_ROOT:-.}/results/r1/prune_${r}_scores.json) > logs/r1_prune_${r}_scores.log 2>&1
  CUDA_VISIBLE_DEVICES=1 m0/.venv/bin/python m3/score_tempo.py --samples results/r1/samples_prune_$r --out results/r1/prune_${r}_tempo.json --Ns 32 --device cuda > logs/r1_prune_${r}_tempo.log 2>&1
  CUDA_VISIBLE_DEVICES="" m2/.venv-aes/bin/python m3/score_aes.py --samples results/r1/samples_prune_$r --out results/r1/prune_${r}_aesthetics.json --Ns 32 > logs/r1_prune_${r}_aes.log 2>&1
done
$PY scripts/r1_phase2_analysis.py > logs/r1_phase2_analysis.log 2>&1
echo "music rerun post done $(date)"
