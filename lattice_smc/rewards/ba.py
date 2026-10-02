"""R_BA (SPEC 3.2): beat alignment over a K-chunk sequence, full and additive forms.

Full: kinematic beats from the whole sequence's joint velocities; R_BA = mean over music beats of
exp(-d^2 / (2 sigma^2)), d = distance to the nearest kinematic beat. sigma = 3 frames.
Additive: r_k = sum over music beats in chunk k of the same kernel, with d measured to the nearest
kinematic beat of the prefix up to and including chunk k (kinematic beats recomputed on the
first 150k frames); R_BA_additive = sum_k r_k / n_beats. The two differ only where a music beat's
nearest kinematic beat lies in a later chunk (or where the smoothing / extremum detection at the
running right edge shifts a beat); the gap is reported.
The beat_alignment kernel is copied from the earlier harness's eval.py (see kinematics.py header).
"""
import numpy as np

from lattice_smc.rewards.kinematics import kinematic_beats

BA_SIGMA = 3.0
CHUNK_LEN = 150


def music_beats_from_features(music):
    """music [T, 35] numpy; beat one-hot is the last channel."""
    return np.where(music[:, -1] > 0.5)[0].astype(np.int64)


def kernel_sum(music_beats, kin_beats, sigma=BA_SIGMA):
    """sum over music beats of exp(-d^2 / 2 sigma^2); 0 if either set is empty."""
    if len(music_beats) == 0 or len(kin_beats) == 0:
        return 0.0
    d = (music_beats[:, None] - kin_beats[None, :]) ** 2
    return float(np.exp(-d.min(1) / (2 * sigma ** 2)).sum())


def ba_full(joints, music_beats, sigma=BA_SIGMA):
    """joints [T, 24, 3] numpy. None if the music has no beats."""
    if len(music_beats) == 0:
        return None
    return kernel_sum(music_beats, kinematic_beats(joints), sigma) / len(music_beats)


def ba_additive(joints, music_beats, K, sigma=BA_SIGMA, chunk_len=CHUNK_LEN):
    """Returns (R_BA_additive, [r_1..r_K]) or (None, []) if the music has no beats."""
    if len(music_beats) == 0:
        return None, []
    r = []
    for k in range(1, K + 1):
        kin = kinematic_beats(joints[: k * chunk_len])
        in_chunk = music_beats[(music_beats >= (k - 1) * chunk_len) & (music_beats < k * chunk_len)]
        r.append(kernel_sum(in_chunk, kin, sigma))
    return float(sum(r) / len(music_beats)), r


def r_k_increment(joints_prefix, music_beats, k, sigma=BA_SIGMA, chunk_len=CHUNK_LEN):
    """The additive term of chunk k alone, from a prefix of at least k chunks (for steering
    methods in Phase 2; unused in Phase 1)."""
    kin = kinematic_beats(joints_prefix[: k * chunk_len])
    in_chunk = music_beats[(music_beats >= (k - 1) * chunk_len) & (music_beats < k * chunk_len)]
    return kernel_sum(in_chunk, kin, sigma)
