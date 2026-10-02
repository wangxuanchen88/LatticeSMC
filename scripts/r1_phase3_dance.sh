#!/bin/bash
# R1 Phase 3 (dance): K = 6 on the 40 paper conditions and K = 8 on the 16 feasible ones; starts after Phase 2's generation.
cd ${LATTICESMC_ROOT:-.}
PY=.venv/bin/python
until grep -q "phase 2 generation done" logs/r1_phase2.log; do sleep 120; done
echo "phase 3 dance start $(date)"
run() { CUDA_VISIBLE_DEVICES=$1 $PY -m lattice_smc.generate --grid configs/$2.yaml > logs/$2.log 2>&1; }
( run 0 r1_k6_ba; run 0 r1_k6_ba_argmax; run 0 r1_k8_ba; run 0 r1_k8_ba_argmax; echo "gpu0 done $(date)" ) &
( run 1 r1_k6_rep; run 1 r1_k6_rep_argmax; run 1 r1_k8_rep; run 1 r1_k8_rep_argmax; echo "gpu1 done $(date)" ) &
wait
echo "phase 3 dance generation done $(date)"
for K in 6 8; do for R in ba rep; do
  LATTICE_PROMPTS=data/prompts_K$K.json CUDA_VISIBLE_DEVICES=$(( K == 6 ? 0 : 1 )) $PY -m lattice_smc.analyze_phase2 --samples samples_r1_k${K}_${R} --tag r1_k${K}_${R} --metrics_only > logs/r1_k${K}_${R}_metrics.log 2>&1 &
  LATTICE_PROMPTS=data/prompts_K$K.json CUDA_VISIBLE_DEVICES=$(( K == 6 ? 1 : 0 )) $PY -m lattice_smc.analyze_phase2 --samples samples_r1_k${K}_${R}_argmax --tag r1_k${K}_${R}_argmax --metrics_only > logs/r1_k${K}_${R}_argmax_metrics.log 2>&1 &
  wait
done; done
echo "phase 3 dance metrics done $(date)"
