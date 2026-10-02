"""Amendment M unit tests for R_rep: r_1 = r_2 = 0 exactly; r_k in (0, 1]; the additive form
equals the full form exactly; the distance is root-relative and translation-invariant; a chunk
identical to chunk k-2 gives r_k = 1; reward-evaluation counting."""
import os

import numpy as np
import pytest
import torch

from lattice_smc.data.clips import load_normalizer
from lattice_smc.rewards.counters import Counters
from lattice_smc.rewards.rep import TAU_PATH, RepEvaluator, chunk_distance

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def ev():
    if not os.path.exists(TAU_PATH):
        pytest.skip("data/rep_tau.json not computed yet")
    from vis import SMPLSkeleton
    scale, mn = load_normalizer()
    return RepEvaluator(None, SMPLSkeleton(DEV), scale, mn, Counters())


def random_motion(B, T, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, T, 151, generator=g) * 2 - 1).to(DEV)


def test_first_two_chunks_are_exactly_zero_and_terms_in_unit_interval(ev):
    J = ev.joints(random_motion(3, 600, 0))
    assert np.array_equal(ev.r_k(J, 1), np.zeros(3)) and np.array_equal(ev.r_k(J, 2), np.zeros(3))
    for k in (3, 4):
        r = ev.r_k(J, k)
        assert np.all(r > 0) and np.all(r <= 1)


def test_additive_equals_full_exactly(ev):
    J = ev.joints(random_motion(4, 600, 1))
    add = sum(ev.r_k(J, k) for k in range(1, 5))
    full = ev.full(J)
    assert np.array_equal(add, full)


def test_repeat_gives_one_and_translation_invariance(ev):
    m = random_motion(2, 600, 2)
    m[:, 300:450] = m[:, 0:150]  # chunk 3 repeats chunk 1 exactly
    J = ev.joints(m)
    assert np.allclose(ev.r_k(J, 3), 1.0, atol=1e-6)
    m2 = m.clone()
    m2[:, 450:600, 4:7] += 0.3  # translate chunk 4's root only: root-relative distance unchanged
    J2 = ev.joints(m2)
    assert np.allclose(chunk_distance(J2, 4), chunk_distance(J, 4), atol=1e-5)


def test_reward_eval_counting(ev):
    ev.counters.reset()
    J = ev.joints(random_motion(5, 600, 3))
    ev.r_k(J, 1)
    ev.r_k(J, 4)
    ev.full(J)
    assert ev.counters.reward_evals.n == 15
