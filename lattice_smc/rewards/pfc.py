"""PFC over a K-chunk sequence (SPEC 3.2, held-out reward in Phase 2; base-quality check in
Phase 1). Reported x 10000 as in EDGE. R_PFC (negative PFC, additive over frames) is a Phase 2
item and is not implemented here."""
from lattice_smc.rewards.kinematics import pfc

PFC_SCALE = 10000.0


def pfc_scaled(joints):
    return pfc(joints) * PFC_SCALE
