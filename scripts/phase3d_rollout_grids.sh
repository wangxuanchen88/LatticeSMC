#!/bin/bash
# Amendment R part 4: rollout lookahead grids (M = 1, 2, 4; N in {4, 8}) then the NFE-matched notwist grid; GPU 0.
cd ${LATTICESMC_ROOT:-.}
for M in 1 2 4; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python -m lattice_smc.generate --grid configs/phase3_rep_grid_roll_M$M.yaml
done
CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python -m lattice_smc.generate --grid configs/phase3_rep_grid_match.yaml
