"""Amendment N: beta = 0 makes lattice_smc equal lattice_smc_notwist bitwise; beta = 1 equals the
default; the oracle mode runs, returns finite log psi with a standard error and counts its NFE
separately (sequence NFE unchanged). Random-init model and twist, K = 3."""
import numpy as np
import torch

from lattice_smc.generate import Context, generate_sequence, set_deterministic
from lattice_smc.model.sampler import DDIM_STEPS
from lattice_smc.model.small import build_diffusion, build_model
from lattice_smc.twist.model import TwistMLP

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K = 3


def make_ctx():
    set_deterministic()
    torch.manual_seed(5)
    model = build_model({}).to(DEV).eval()
    torch.manual_seed(11)
    tw = TwistMLP(K).to(DEV).eval()
    ctx = Context(model, build_diffusion(model, DEV).eval(), DEV, {"loss": {"repr": "disp"}}, alpha=0.1, twist=tw, reward="rep")
    return ctx


def music():
    return torch.randn(K * 150, 35, generator=torch.Generator().manual_seed(3))


def test_beta_zero_equals_notwist_and_beta_one_equals_default():
    ctx = make_ctx()
    m = music()
    b, lb = generate_sequence(ctx, m, K, 2001, "lattice_smc_notwist", 8)
    ctx.twist_beta = 0.0
    a, la = generate_sequence(ctx, m, K, 2001, "lattice_smc", 8)
    assert torch.equal(a, b) and la["weights"] == lb["weights"]
    ctx.twist_beta = 1.0
    c, lc = generate_sequence(ctx, m, K, 2001, "lattice_smc", 8)
    ctx.twist_beta = 0.25
    d, ld = generate_sequence(ctx, m, K, 2001, "lattice_smc", 8)
    assert ld["twist_beta"] == 0.25 and lc["twist_beta"] == 1.0 and ld["twist_evals"] == lc["twist_evals"] == (K - 1) * 8


def test_oracle_mode_counts_separately():
    ctx = make_ctx()
    ctx.twist_mode, ctx.oracle_M = "oracle", 4
    m = music()
    a, la = generate_sequence(ctx, m, K, 2002, "lattice_smc", 4)
    assert la["nfe"] == 4 * K * DDIM_STEPS * 2  # sequence NFE unchanged
    assert la["twist_mode"] == "oracle" and la["twist_evals"] == 0
    # oracle at k = 1 only (k < K - 1 = 2): 4 particles x 4 continuations x 2 chunks x 50 x 2
    assert ctx.oracle_counters.nfe.n == 4 * 4 * (K - 1) * DDIM_STEPS * 2
    assert la["oracle_mc_se"][0]["chunk"] == 1 and np.isfinite(la["oracle_mc_se"][0]["mc_se_mean_nats"])
    b, lb = generate_sequence(ctx, m, K, 2002, "lattice_smc", 4)
    assert torch.equal(a, b)
