#!/bin/bash
# Phase M3 grid (Amendment T): one shard per GPU, resumable (skip-if-exists), under nohup.
#   scripts/m3_grid.sh <gpu> <shard i/n> [seeds] [runs] [Ns]
GPU=$1; SHARD=$2; SEEDS=${3:-2000,2001,2002,2003}; RUNS=${4:-bon,greedy_chunk,lsp1,lsp05,lsp025,fk_noise}; NS=${5:-1,2,4,8,16,32}
SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}
cd $SA3 && CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/music_lattice.py \
  --runs $RUNS --Ns $NS --seeds $SEEDS --shard $SHARD --out_root ${LATTICESMC_ROOT:-.}/results/m3/samples
