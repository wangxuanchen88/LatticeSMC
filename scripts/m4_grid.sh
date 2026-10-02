#!/bin/bash
# Phase M4 grid (Amendment V, reward motif): one shard per process, resumable, seed-major order.
#   scripts/m4_grid.sh <gpu> <shard i/n>
GPU=$1; SHARD=$2
SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
for SEED in 2000 2001; do
  cd $SA3 && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/music_lattice.py \
    --reward motif --alpha_file ${LATTICESMC_ROOT:-.}/results/m4/alpha.json --runs base,bon,greedy_chunk,lsp1,lsp05,lsp025,fk_noise \
    --Ns 1,2,4,8,16,32 --seeds $SEED --shard $SHARD --out_root ${LATTICESMC_ROOT:-.}/results/m4/samples
done
