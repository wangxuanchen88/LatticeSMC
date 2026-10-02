#!/bin/bash
# Amendment U: deterministic regeneration of lattice_smc_notwist with the argmax return rule (same particles, same draws)
cd ${LATTICESMC_ROOT:-.}
GPU=$1; shift
for cfg in "$@"; do CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m lattice_smc.generate --grid configs/$cfg.yaml; done
