"""Blocking Phase 1 checks of the adopted base model, read from results/base_model_checks.json
(written by `python -m lattice_smc.analyze --phase1`); skipped until that file exists."""
import json
import os

import pytest

from lattice_smc.analyze import BOUNDARY_RATIO_LIMIT, RESULTS_DIR

# The adopted base model's checks (results/base_model_checks.json, a symlink set when a model is
# adopted); falls back to the Phase 1 file before any adoption.
PATH = os.path.join(RESULTS_DIR, "base_model_checks.json")
if not os.path.exists(PATH):
    PATH = os.path.join(RESULTS_DIR, "phase1_checks.json")


@pytest.fixture
def checks():
    if not os.path.exists(PATH):
        pytest.skip("results/phase1_checks.json not written yet")
    return json.load(open(PATH))


def test_nfe_matches_budget(checks):
    assert checks["nfe"]["all_equal_expected"], checks["nfe"]


def test_boundary_continuity(checks):
    b = checks["boundary"]
    assert b["ratio_pooled_mean_seam_over_median_within"] <= BOUNDARY_RATIO_LIMIT, b


def test_determinism_across_runs_and_gpus(checks):
    rows = checks["determinism"]
    assert rows, "no comparison roots were given to analyze --compare"
    for r in rows:
        assert r["n_compared"] > 0, r
        assert r["n_bitwise_equal"] == r["n_compared"], r
