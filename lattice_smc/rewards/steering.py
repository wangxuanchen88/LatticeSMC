"""Reward evaluation for the inference-time methods (R_BA additive terms, full R_BA), the ESS,
and systematic resampling under a per-sample generator. Every evaluation increments
counters.reward_evals by the number of particles evaluated. Joints come from the Amendment I
pipeline (no root clip)."""
import numpy as np
import torch

from lattice_smc.rewards.ba import kernel_sum, music_beats_from_features
from lattice_smc.rewards.kinematics import kinematic_beats, to_joints

CHUNK_LEN = 150


class RewardEvaluator:
    def __init__(self, music, smpl, scale, mn, counters, sigma=3.0):
        """music [K*150, 35] (tensor or numpy) of the prompt; beats read off the last channel."""
        m = music.detach().cpu().numpy() if torch.is_tensor(music) else np.asarray(music)
        self.beats = music_beats_from_features(m)
        self.n_beats = max(len(self.beats), 1)
        self.smpl, self.scale, self.mn, self.counters, self.sigma = smpl, scale, mn, counters, sigma

    def joints(self, motion):
        """[B, T, 151] normalized (absolute root) -> [B, T, 24, 3] numpy, root un-clipped."""
        return to_joints(motion, self.smpl, self.scale, self.mn, clip_root=False).cpu().numpy()

    def r_k(self, joints_prefix, k):
        """Additive term of chunk k (1-based) for each particle from joints of frames
        [0, 150k) or longer (only the first 150k frames are used). Returns numpy [B]; counts B."""
        B = joints_prefix.shape[0]
        lo, hi = (k - 1) * CHUNK_LEN, k * CHUNK_LEN
        in_chunk = self.beats[(self.beats >= lo) & (self.beats < hi)]
        out = np.zeros(B)
        for b in range(B):
            kin = kinematic_beats(joints_prefix[b, :hi])
            out[b] = kernel_sum(in_chunk, kin, self.sigma) / self.n_beats
        self.counters.reward_evals.add(B)
        return out

    def full(self, joints):
        """Full-sequence R_BA per particle (kinematic beats from the whole sequence). Counts B."""
        B = joints.shape[0]
        out = np.zeros(B)
        for b in range(B):
            out[b] = kernel_sum(self.beats, kinematic_beats(joints[b]), self.sigma) / self.n_beats
        self.counters.reward_evals.add(B)
        return out


def normalized_weights(logw):
    """logw numpy [N] -> normalized weights [N]."""
    w = np.exp(logw - logw.max())
    return w / w.sum()


def ess(logw):
    w = normalized_weights(logw)
    return float(1.0 / np.sum(w ** 2))


def systematic_resample(weights, gen):
    """weights numpy [N] normalized; one uniform draw from the CPU generator gen.
    Returns parent indices [N] (numpy int64)."""
    N = len(weights)
    u0 = torch.rand(1, generator=gen).item()
    positions = (u0 + np.arange(N)) / N
    cum = np.cumsum(weights)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions, side="right").clip(0, N - 1).astype(np.int64)


def draw_index(weights, gen):
    """One index drawn with the given probabilities from gen (inverse CDF, one uniform)."""
    u = torch.rand(1, generator=gen).item()
    cum = np.cumsum(weights)
    cum[-1] = 1.0
    return int(np.searchsorted(cum, u, side="right").clip(0, len(weights) - 1))
