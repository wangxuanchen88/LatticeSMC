"""Method registry (SPEC 4). Phase 3's lattice_smc_adaptive is not implemented."""
from lattice_smc.methods import base, bon, fk_noise, greedy_chunk, lattice_smc

METHODS = {"base": base.run, "bon_argmax": bon.run_argmax, "bon_is": bon.run_is, "fk_noise": fk_noise.run,
           "greedy_chunk": greedy_chunk.run, "lattice_smc": lattice_smc.run,
           "lattice_smc_notwist": lattice_smc.run_notwist}
