"""R_rep (Amendment M): a long-range, chunk-additive reward. r_k = 0 for k = 1, 2; for k = 3, 4,
r_k = exp(-d(chunk_k, chunk_{k-2}) / tau) with d the mean over frames and joints of the
root-relative joint-position distance between the two chunks frame by frame (Amendment I
joints), tau = median of d over the Phase 2 base samples (data/rep_tau.json, computed once).
R_rep = r_3 + r_4; the additive form equals the full form exactly (no boundary approximation).
Same interface as RewardEvaluator (r_k / full / joints, reward evaluations counted).

  python -m lattice_smc.rewards.rep --compute_tau        # from samples_p2/base/1
"""
import glob
import json
import os

import numpy as np
import torch

from lattice_smc.edge_import import ROOT
from lattice_smc.rewards.kinematics import to_joints

CHUNK_LEN = 150
TAU_PATH = os.path.join(ROOT, "data", "rep_tau.json")


def chunk_distance(joints, k, chunk_len=CHUNK_LEN):
    """d(chunk_k, chunk_{k-2}) for joints [B, >= 150k, 24, 3] (numpy): mean over frames and
    joints of the L2 distance between root-relative joint positions, frame by frame."""
    a = joints[:, (k - 1) * chunk_len: k * chunk_len]
    b = joints[:, (k - 3) * chunk_len: (k - 2) * chunk_len]
    a = a - a[:, :, :1]
    b = b - b[:, :, :1]
    return np.linalg.norm(a - b, axis=-1).mean(axis=(1, 2))


def load_tau():
    return float(json.load(open(TAU_PATH))["tau"])


class RepEvaluator:
    def __init__(self, music, smpl, scale, mn, counters, tau=None):
        self.smpl, self.scale, self.mn, self.counters = smpl, scale, mn, counters
        self.tau = load_tau() if tau is None else float(tau)
        self.beats = np.zeros(0, dtype=np.int64)  # unused; kept for interface parity

    def joints(self, motion):
        return to_joints(motion, self.smpl, self.scale, self.mn, clip_root=False).cpu().numpy()

    def r_k(self, joints_prefix, k):
        """Additive term of chunk k from joints of frames [0, 150k) or longer. Counts B."""
        B = joints_prefix.shape[0]
        self.counters.reward_evals.add(B)
        if k < 3:
            return np.zeros(B)
        return np.exp(-chunk_distance(joints_prefix, k) / self.tau)

    def full(self, joints):
        """R_rep of whole sequences [B, K*150, 24, 3]; equals the sum of the additive terms
        exactly. Counts B."""
        B = joints.shape[0]
        K = joints.shape[1] // CHUNK_LEN
        self.counters.reward_evals.add(B)
        out = np.zeros(B)
        for k in range(3, K + 1):
            out += np.exp(-chunk_distance(joints, k) / self.tau)
        return out


def compute_tau(samples_root="samples_p2/base/1", device="cuda:0"):
    from lattice_smc.data.clips import load_normalizer
    from vis import SMPLSkeleton
    dev = torch.device(device)
    smpl = SMPLSkeleton(dev)
    scale, mn = load_normalizer()
    files = sorted(glob.glob(os.path.join(ROOT, samples_root, "*.npz")))
    mot = torch.from_numpy(np.stack([np.load(f)["motion"] for f in files])).to(dev)
    J = to_joints(mot, smpl, scale, mn, clip_root=False).cpu().numpy()
    d3, d4 = chunk_distance(J, 3), chunk_distance(J, 4)
    d = np.concatenate([d3, d4])
    info = {"tau": float(np.median(d)), "n_sequences": len(files), "n_distances": int(len(d)), "source": samples_root,
            "d_mean": float(d.mean()), "d_quantiles_10_50_90": np.quantile(d, [0.1, 0.5, 0.9]).tolist(),
            "d3_median": float(np.median(d3)), "d4_median": float(np.median(d4)),
            "r_at_median": float(np.exp(-1.0)), "r_quantiles_10_50_90": np.exp(-np.quantile(d, [0.9, 0.5, 0.1]) / np.median(d)).tolist()}
    json.dump(info, open(TAU_PATH, "w"), indent=1)
    print(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute_tau", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.compute_tau:
        compute_tau(device=a.device)
