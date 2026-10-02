"""Twist MLP (SPEC 5): two hidden layers of 512, output scalar = log psi. Inputs are standardized
with the training-set mean / std stored in the module. Every evaluation adds N to
counters.twist_evals (not NFE)."""
import os

import torch
import torch.nn as nn

from lattice_smc.edge_import import ROOT
from lattice_smc.twist.features import feature_dim, twist_features

TWIST_PATH = os.path.join(ROOT, "data", "twist", "twist.pt")


class TwistMLP(nn.Module):
    def __init__(self, K, hidden=512):
        super().__init__()
        d = feature_dim(K)
        self.K = K
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def set_standardization(self, x):
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def zero_twist(K, device):
    """psi = 1 everywhere (used by tests: lattice_smc with this twist must equal the ablation)."""
    m = TwistMLP(K).to(device)
    with torch.no_grad():
        m.net[-1].weight.zero_()
        m.net[-1].bias.zero_()
    return m.eval()


class TwistEnsemble(nn.Module):
    """Amendment P: mean m and variance v of the member predictions; mode "mean" returns m,
    mode "shrunk" returns m x s2_k / (s2_k + c v) with s2_k the validation target variance
    within chunk index k (read from the one-hot tail of the features) and c a fixed scalar."""

    def __init__(self, members, mode="mean", c=0.0, s2=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.mode, self.c = mode, float(c)
        self.K = members[0].K
        self.register_buffer("s2", torch.tensor(s2 if s2 is not None else [1.0] * (self.K - 1), dtype=torch.float32))

    def mean_var(self, x):
        preds = torch.stack([m(x) for m in self.members])  # [E, N]
        return preds.mean(0), preds.var(0, unbiased=True)

    def forward(self, x):
        m, v = self.mean_var(x)
        if self.mode == "mean":
            return m
        k_idx = x[:, -(self.K - 1):].argmax(-1)
        s2 = self.s2[k_idx]
        return m * s2 / (s2 + self.c * v)


def load_twist(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("type") == "ensemble":
        members = []
        for sd in ck["members"]:
            m = TwistMLP(ck["K"], ck.get("hidden", 512)).to(device)
            m.load_state_dict(sd)
            members.append(m.eval())
        return TwistEnsemble(members, ck["mode"], ck.get("c", 0.0), ck.get("s2")).to(device).eval()
    m = TwistMLP(ck["K"], ck.get("hidden", 512)).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def twist_log_psi(ctx, chunk_abs, music, k, K):
    """log psi_k for N particles' chunk k; counts N twist evaluations. Returns numpy [N]."""
    feats = twist_features(chunk_abs, music, k, K)
    out = ctx.twist(feats)
    ctx.counters.twist_evals.add(feats.shape[0])
    return out.double().cpu().numpy()
