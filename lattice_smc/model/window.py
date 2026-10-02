"""Root-relative window (SPEC Amendment C) and root-displacement chunk (Amendment E).

Amendment C: the prefix's frame-149 root position (normalized channels 4:7) is subtracted from
every frame of the 300-frame window before the denoiser sees it and added back to the generated
chunk afterwards. Chunk 1 (no prefix) uses no shift. Arithmetic is done in the window's dtype. In
float64 the round trip is exact on the whole pair cache; in float32 it is exact to one ulp (IEEE
subtraction cannot be exactly inverted when the shift is much larger than the value), see
tests/test_root_shift.py and RESULTS.md Phase 1b.

Amendment E: for the chunk half, root channels hold d_out = (p[t] - p[t-1]) / (scale x s), i.e.
the per-frame root displacement in metres divided by the per-axis std s from
data/root_disp_scale.json (1 normalized unit = 1 / scale metres, EDGE's MinMax scale). Positions
are reconstructed by cumulative sum from an anchor (p[149] in the frame the window lives in). The
cumulative sum of exact normalized displacements reproduces float32-valued data bitwise in float64
(every partial sum is a difference of two float32 numbers); the scaling by 1 / (scale x s) is a
separate diagonal map, exact only to floating-point rounding.
"""
import json
import os

import numpy as np
import torch

from lattice_smc.edge_import import ROOT as REPO_ROOT

CHUNK_LEN = 150
ROOT = slice(4, 7)
DISP_CLIP = 5.0
DEFAULT_LOSS_CFG = {"repr": "relative", "root_weight": 1.0, "seam_lambda": 0.0}

_nz = np.load(os.path.join(REPO_ROOT, "data", "normalizer.npz"))
ROOT_SCALE = torch.from_numpy(_nz["scale"][4:7].astype(np.float64))  # normalized units per metre
_disp = json.load(open(os.path.join(REPO_ROOT, "data", "root_disp_scale.json")))
DISP_S = torch.tensor(_disp["s_m_per_frame"], dtype=torch.float64)  # metres per frame
DISP_FACTOR = ROOT_SCALE * DISP_S  # normalized units per output unit


def loss_cfg_from(config):
    """Loss / representation settings of a run config (or checkpoint config)."""
    cfg = dict(DEFAULT_LOSS_CFG)
    cfg.update((config or {}).get("loss") or {})
    return cfg


def root_shift(window, uncond=None):
    """[B, 300, 151] window (or [B, >=150, 151] prefix) -> shift [B, 3] = root at frame 149;
    zero for unconditional-start rows."""
    s = window[:, CHUNK_LEN - 1, ROOT].clone()
    if uncond is not None:
        s[uncond.to(s.device)] = 0
    return s


def shift_window(x, s):
    """Subtract s [B, 3] from the root channels of every frame of x [B, T, 151]."""
    out = x.clone()
    out[..., ROOT] = out[..., ROOT] - s[:, None].to(out.dtype)
    return out


def unshift_window(x_rel, s):
    out = x_rel.clone()
    out[..., ROOT] = out[..., ROOT] + s[:, None].to(out.dtype)
    return out


def _factor(like):
    return DISP_FACTOR.to(like.device, like.dtype)


def chunk_to_disp(window):
    """[B, 300, 151] positions (any root frame) -> same with the chunk half's root channels
    replaced by scaled displacements d_out[t] = (p[t] - p[t-1]) / factor, t = 150..299."""
    out = window.clone()
    p = window[..., ROOT]
    out[:, CHUNK_LEN:, ROOT] = (p[:, CHUNK_LEN:] - p[:, CHUNK_LEN - 1:-1]) / _factor(window)
    return out


def positions_from_disp_norm(d_norm, anchor):
    """anchor [B, 3] + cumulative sum of normalized displacements d_norm [B, T, 3]."""
    return anchor[:, None].to(d_norm.dtype) + torch.cumsum(d_norm, dim=1)


def positions_from_disp(d_out, anchor):
    """Scaled displacements [B, T, 3] -> absolute normalized root positions from anchor [B, 3]."""
    return positions_from_disp_norm(d_out * _factor(d_out), anchor)


def chunk_from_disp(chunk_disp, anchor):
    """[B, 150, 151] chunk in displacement form -> position form, integrated from anchor."""
    out = chunk_disp.clone()
    out[..., ROOT] = positions_from_disp(chunk_disp[..., ROOT], anchor)
    return out


def clip_x0(x0, cfg):
    """DDIM x0 clip: [-1, 1] on every channel; +-DISP_CLIP on the displacement channels of a
    chunk in displacement form (Amendment E)."""
    if cfg["repr"] != "disp":
        return x0.clamp(-1.0, 1.0)
    out = x0.clamp(-1.0, 1.0)
    out[..., ROOT] = x0[..., ROOT].clamp(-DISP_CLIP, DISP_CLIP)
    return out
