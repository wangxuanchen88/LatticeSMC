#!/bin/bash
# Phase 1 base generation for all prompts x 4 seeds, sharded by prompt over both GPUs, then two
# reruns for the determinism check: same shard->GPU assignment (run-to-run) and swapped (cross-GPU).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CK=${1:-runs/chunk_s101/ckpt_final.pt}
R=${2:-samples}   # root prefix: $R, ${R}_rerun, ${R}_swap
run() {  # root, gpu for shard 0, gpu for shard 1
  CUDA_VISIBLE_DEVICES=$2 $PY -m lattice_smc.generate --ckpt $CK --method base --N 1 --prompts all --seeds 2000-2003 --shard 0/2 --out_root $1 > logs/gen_$(basename $1)_s0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$3 $PY -m lattice_smc.generate --ckpt $CK --method base --N 1 --prompts all --seeds 2000-2003 --shard 1/2 --out_root $1 > logs/gen_$(basename $1)_s1.log 2>&1 &
  wait
}
t0=$(date +%s); run $R 0 1;        echo "$R: $(( $(date +%s) - t0 )) s"
t0=$(date +%s); run ${R}_rerun 0 1;  echo "${R}_rerun: $(( $(date +%s) - t0 )) s"
t0=$(date +%s); run ${R}_swap 1 0;   echo "${R}_swap: $(( $(date +%s) - t0 )) s"
ls $R/base/1 | wc -l; ls ${R}_rerun/base/1 | wc -l; ls ${R}_swap/base/1 | wc -l
