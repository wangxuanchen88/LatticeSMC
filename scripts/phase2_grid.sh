#!/bin/bash
# Phase 2 main grid: both GPUs, sharded by prompt, resumable (existing outputs are skipped).
cd "$(dirname "$0")/.."
for s in 0 1; do
  nohup env CUDA_VISIBLE_DEVICES=$s .venv/bin/python -m lattice_smc.generate --grid configs/phase2_grid.yaml --shard $s/2 > logs/grid_s$s.log 2>&1 &
done
echo "grid launched: logs/grid_s0.log logs/grid_s1.log"
