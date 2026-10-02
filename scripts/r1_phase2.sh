#!/bin/bash
# R1 Phase 2: chunk-pruning M sweep (M in {1, N/2 = 16}) at N = 32, argmax return, all four rewards; resumable.
# GPU 0: the four dance configs, then the motif-recurrence music runs; GPU 1: the prompt-adherence music runs.
cd ${LATTICESMC_ROOT:-.}
PY=.venv/bin/python; SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
music() {  # gpu reward M
  local RW=$2 M=$3 MT=${3/\//}; local AF=results/m2/alpha.json; [ $RW = motif ] && AF=results/m4/alpha.json
  (cd $SA3 && CUDA_VISIBLE_DEVICES=$1 HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/music_lattice.py --reward $RW --alpha_file ${LATTICESMC_ROOT:-.}/$AF \
    --runs greedy_chunk --Ns 32 --seeds 2000,2001 --prune_M $M --out_root ${LATTICESMC_ROOT:-.}/results/r1/samples_prune_${RW}_M${MT}) > logs/r1_prune_${RW}_M${MT}.log 2>&1
}
echo "phase 2 start $(date)"
( for c in r1_prune_ba_M1 r1_prune_ba_M16 r1_prune_rep_M1 r1_prune_rep_M16; do CUDA_VISIBLE_DEVICES=0 $PY -m lattice_smc.generate --grid configs/$c.yaml > logs/$c.log 2>&1; done
  echo "dance done $(date)"; music 0 motif 1; music 0 motif N/2; echo "motif done $(date)" ) &
( music 1 clap 1; music 1 clap N/2; echo "clap done $(date)" ) &
wait
echo "phase 2 generation done $(date)"
