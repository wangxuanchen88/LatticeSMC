# The FK / kinematic-beat / PFC code below is copied from the earlier harness:
# <earlier-harness>/eval.py (working tree; no git commit), sha256
# 138f51a390fb530329bedc174e38f600b850bdcb8e761791bdd6d0ba238848ff; EDGE pin
# 17c3428669ed6733edd9d8c66f7dc62060b8e46d. Modifications: functions take sequences of any length
# (K*150 frames); the T_loss / generation parts of eval.py are not needed here.
"""Joint positions, kinematic beats and PFC for sequences of any length."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from lattice_smc.data.clips import unnormalize
from lattice_smc.edge_import import EDGE_DIR  # noqa: F401
from dataset.quaternion import ax_from_6v  # noqa: E402

KIN_SMOOTH_SIGMA = 5  # Bailando's reference implementation smooths the velocity curve
FPS = 30


@torch.no_grad()
def to_joints(x_norm, smpl, scale, mn, clip_root=True):
    """Normalized [B, T, 151] -> unnormalized -> FK joint positions [B, T, 24, 3] (z up).
    clip_root=False (Amendment I, generated sequences): the root channels are un-normalized
    without EDGE's [-1, 1] clip; rotation channels keep the clip."""
    x = unnormalize(x_norm, scale, mn)
    if not clip_root:
        x = x.clone()
        x[..., 4:7] = (x_norm[..., 4:7] - mn[4:7].to(x.device)) / scale[4:7].to(x.device)
    pos = x[..., 4:7]
    q = ax_from_6v(x[..., 7:].reshape(x.shape[0], x.shape[1], 24, 6))
    return smpl.forward(q, pos)


def root_covariates(x_norm, scale, mn):
    """Amendment J: net root travel in the ground plane (m) and mean per-frame root displacement
    per axis (mm/frame) for [B, T, 151] normalized motion. Returns (travel [B], mean_disp [B, 3])."""
    p = (x_norm[..., 4:7] - mn[4:7].to(x_norm.device)) / scale[4:7].to(x_norm.device)
    travel = (p[:, -1, :2] - p[:, 0, :2]).norm(dim=-1)
    mean_disp = (p[:, 1:] - p[:, :-1]).mean(1) * 1000
    return travel, mean_disp


def realism_w1(accel_sample, accel_gt_pool):
    """1-Wasserstein distance between the pooled per-frame joint-acceleration magnitudes of one
    sample and the pooled ground-truth distribution (scipy)."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(accel_sample, accel_gt_pool))


def kinematic_beats(joints):
    """joints: [T, 24, 3] numpy. Local minima of the (smoothed) mean joint speed."""
    vel = np.linalg.norm(joints[1:] - joints[:-1], axis=-1).mean(-1)  # [T-1]
    vel = gaussian_filter1d(vel, KIN_SMOOTH_SIGMA)
    return argrelextrema(vel, np.less)[0] + 1  # +1: velocity index t is between t and t+1


def pfc(joints):
    """Port of edge/eval/eval_pfc.py::calc_physical_score for one in-memory [T, 24, 3] motion
    (z up). Returns the per-motion score before the x10000 scaling."""
    up_dir, dt = 2, 1 / FPS
    flat = [i for i in range(3) if i != up_dir]
    root_v = (joints[1:, 0] - joints[:-1, 0]) / dt
    root_a = (root_v[1:] - root_v[:-1]) / dt
    root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)
    root_a = np.linalg.norm(root_a, axis=-1)
    root_a = root_a / root_a.max()
    feet = joints[:, [7, 10, 8, 11]]
    foot_v = np.linalg.norm(feet[2:][:, :, flat] - feet[1:-1][:, :, flat], axis=-1)
    left = np.minimum(foot_v[:, 0], foot_v[:, 1])
    right = np.minimum(foot_v[:, 2], foot_v[:, 3])
    return float((left * right * root_a).mean())


def frame_jumps(joints):
    """Mean over joints of the frame-to-frame joint displacement, [T-1]."""
    return np.linalg.norm(joints[1:] - joints[:-1], axis=-1).mean(-1)


def accel_magnitude(joints):
    """Per-frame joint-acceleration magnitude, pooled over joints, [(T-2)*24] (realism statistic)."""
    a = joints[2:] - 2 * joints[1:-1] + joints[:-2]
    return np.linalg.norm(a, axis=-1).reshape(-1)
