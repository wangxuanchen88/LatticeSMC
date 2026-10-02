#!/bin/bash
# R1 Phase 3b: alpha = 0.01 boundary / dense schedules at N = 32 on beat alignment, K = 6 (GPU 0) and K = 8 (GPU 1), both rules; then metrics.
cd ${LATTICESMC_ROOT:-.}
PY=.venv/bin/python
run() { CUDA_VISIBLE_DEVICES=$1 $PY -m lattice_smc.generate --grid configs/$2.yaml > logs/$2.log 2>&1; }
echo "phase 3b start $(date)"
( run 0 r1_k6_ba_a001; run 0 r1_k6_ba_a001_argmax; echo "k6 done $(date)" ) &
( run 1 r1_k8_ba_a001; run 1 r1_k8_ba_a001_argmax; echo "k8 done $(date)" ) &
wait
echo "phase 3b generation done $(date)"
for K in 6 8; do for s in "" _argmax; do
  LATTICE_PROMPTS=data/prompts_K$K.json CUDA_VISIBLE_DEVICES=$(( K == 6 ? 0 : 1 )) $PY -m lattice_smc.analyze_phase2 --samples samples_r1_k${K}_ba_a001$s --tag r1_k${K}_ba_a001$s --metrics_only > logs/r1_k${K}_ba_a001${s}_metrics.log 2>&1 &
done; done; wait
echo "phase 3b metrics done $(date)"
