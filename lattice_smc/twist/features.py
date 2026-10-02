"""Twist input (SPEC 5): (a) the last 30 frames of the prefix chunk, root-relative (shifted by
the chunk's own frame-149 root, Amendment C), flattened 30 x 151; (b) mean-pooled music features
of each remaining chunk, 35 x (K - 1), zero-padded; (c) one-hot chunk index k in 1..K-1."""
import torch

from lattice_smc.model.window import root_shift, shift_window

CHUNK_LEN = 150
LAST = 30


def feature_dim(K):
    return LAST * 151 + 35 * (K - 1) + (K - 1)


def twist_features(chunk_abs, music, k, K):
    """chunk_abs [N, 150, 151] (chunk k in absolute coordinates), music [K*150, 35], k in 1..K-1.
    Returns [N, feature_dim(K)] float32 on chunk_abs's device."""
    N = chunk_abs.shape[0]
    dev = chunk_abs.device
    rel = shift_window(chunk_abs, root_shift(chunk_abs))[:, -LAST:].reshape(N, -1)
    music = music.to(dev)
    pooled = torch.zeros(K - 1, 35, device=dev, dtype=rel.dtype)
    for j in range(k + 1, K + 1):
        pooled[j - k - 1] = music[(j - 1) * CHUNK_LEN: j * CHUNK_LEN].mean(0)
    onehot = torch.zeros(K - 1, device=dev, dtype=rel.dtype)
    onehot[k - 1] = 1
    return torch.cat([rel, pooled.reshape(1, -1).expand(N, -1), onehot[None].expand(N, -1)], dim=-1).float()
