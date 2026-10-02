"""Phase 2a unit tests for the inference-time methods (SPEC 4), random-init model, K = 2:
  * every method at N = 1 reproduces base bitwise;
  * NFE = N x K x 50 x 2 for every method and N; reward-eval and twist-eval counters are exact;
  * lattice_smc with psi = 1 equals lattice_smc_notwist bitwise;
  * resampling is deterministic under the per-sample generator (two runs bitwise equal), and
    systematic resampling / the final draw follow the weights exactly for degenerate weights.
"""
import numpy as np
import pytest
import torch

from lattice_smc.generate import Context, generate_sequence, set_deterministic
from lattice_smc.methods import METHODS
from lattice_smc.model.sampler import DDIM_STEPS
from lattice_smc.model.small import build_diffusion, build_model
from lattice_smc.rewards.steering import draw_index, ess, systematic_resample
from lattice_smc.twist.model import TwistMLP, zero_twist

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K = 2
ALPHA = 0.1


def make_ctx(twist="random"):
    set_deterministic()
    torch.manual_seed(5)
    model = build_model({}).to(DEV).eval()
    diffusion = build_diffusion(model, DEV).eval()
    cfg = {"loss": {"repr": "disp", "root_weight": 1.0, "seam_lambda": 0.0}}
    if twist == "random":
        torch.manual_seed(11)
        tw = TwistMLP(K).to(DEV).eval()
    elif twist == "zero":
        tw = zero_twist(K, DEV)
    else:
        tw = None
    return Context(model, diffusion, DEV, cfg, alpha=ALPHA, twist=tw)


def music_with_beats():
    g = torch.Generator().manual_seed(3)
    m = torch.randn(K * 150, 35, generator=g)
    m[:, -1] = 0
    m[::17, -1] = 1  # a beat every 17 frames
    return m


@pytest.fixture(scope="module")
def ctx():
    return make_ctx()


@pytest.fixture(scope="module")
def music():
    return music_with_beats()


@pytest.fixture(scope="module")
def base_out(ctx, music):
    return generate_sequence(ctx, music, K, 2000, "base", 1)


@pytest.mark.parametrize("method", sorted(m for m in METHODS if m != "base"))
def test_every_method_at_N1_reproduces_base_bitwise(ctx, music, base_out, method):
    m1, log1 = generate_sequence(ctx, music, K, 2000, method, 1)
    assert torch.equal(m1, base_out[0]), method
    assert log1["nfe"] == base_out[1]["nfe"] == K * DDIM_STEPS * 2


EXPECTED_REWARD_EVALS = {"base": lambda N: 0, "bon_argmax": lambda N: N, "bon_is": lambda N: N,
                         "greedy_chunk": lambda N: K * N, "lattice_smc": lambda N: K * N,
                         "lattice_smc_notwist": lambda N: K * N, "fk_noise": lambda N: 5 * K * N}
EXPECTED_TWIST_EVALS = {"lattice_smc": lambda N: (K - 1) * N}


@pytest.mark.parametrize("N", [1, 2, 4])
@pytest.mark.parametrize("method", sorted(METHODS))
def test_nfe_and_counters_exact(ctx, music, method, N):
    if method == "base" and N != 1:
        pytest.skip("base is N = 1")
    motion, log = generate_sequence(ctx, music, K, 2000, method, N)
    assert log["nfe"] == N * K * DDIM_STEPS * 2, (method, N, log["nfe"])
    assert log["reward_evals"] == EXPECTED_REWARD_EVALS[method](N), (method, N, log["reward_evals"])
    assert log["twist_evals"] == EXPECTED_TWIST_EVALS.get(method, lambda N: 0)(N), (method, N)
    assert motion.shape == (K * 150, 151) and torch.isfinite(motion).all()
    if method != "base":
        assert len(log["rewards"]) == N and np.isfinite(log["final_reward"]) and np.isfinite(log["weighted_mean_reward"])


def test_lattice_with_unit_twist_equals_notwist_bitwise(music):
    c0 = make_ctx("zero")
    for N in [4, 8]:
        a, la = generate_sequence(c0, music, K, 2001, "lattice_smc", N)
        b, lb = generate_sequence(c0, music, K, 2001, "lattice_smc_notwist", N)
        assert torch.equal(a, b)
        assert la["rewards"] == lb["rewards"] and la["weights"] == lb["weights"] and la["chosen"] == lb["chosen"]
        assert la["twist_evals"] == (K - 1) * N and lb["twist_evals"] == 0


@pytest.mark.parametrize("method", ["fk_noise", "lattice_smc", "greedy_chunk", "bon_is"])
def test_resampling_is_deterministic(ctx, music, method):
    a, la = generate_sequence(ctx, music, K, 2002, method, 8)
    b, lb = generate_sequence(ctx, music, K, 2002, method, 8)
    assert torch.equal(a, b)
    assert la["chosen"] == lb["chosen"] and la["rewards"] == lb["rewards"]
    assert la["ess_content"] == lb["ess_content"] and la["ess_noise"] == lb["ess_noise"]
    c, _ = generate_sequence(ctx, music, K, 2003, method, 8)
    assert not torch.equal(a, c)


def test_systematic_resample_and_draw_follow_degenerate_weights():
    g = torch.Generator().manual_seed(0)
    w = np.array([0.0, 0.0, 1.0, 0.0])
    assert systematic_resample(w, g).tolist() == [2, 2, 2, 2]
    assert draw_index(w, g) == 2
    assert abs(ess(np.log(np.array([0.25, 0.25, 0.25, 0.25]))) - 4.0) < 1e-12
    assert abs(ess(np.array([0.0, -50.0, -50.0, -50.0])) - 1.0) < 1e-12
    # a uniform draw stream is reproducible under the generator
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    assert systematic_resample(w, g1).tolist() == systematic_resample(w, g2).tolist()
    parents = systematic_resample(w, torch.Generator().manual_seed(1))
    assert sorted(parents.tolist()) == parents.tolist() and len(parents) == 4  # systematic: monotone
