# Copied from the earlier harness: <earlier-harness>/data/clips.py (working tree; that repo has no git commit).
# sha256 of the source file: 4e52fca03916f7ffd9d7a2a6c7d9b1209a298918b9bd49800eb466a776e9a18f; EDGE pin 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications: import path of the earlier harness -> lattice_smc only. Used for load_normalizer / unnormalize.
"""Load cached clips (SPEC 2.4) into memory tensors. Nothing here touches raw AIST++.
Clip ids are unique across splits, so a pool may mix train- and test-split clips
(Amendment A); the cache directory is resolved per clip."""
import json
import os

import numpy as np
import torch

from lattice_smc.edge_import import ROOT

DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
N_SPANS = 10
SPAN_LEN = 15
CLIP_LEN = 150


def load_pool(pool_file):
    return json.load(open(os.path.join(DATA_DIR, pool_file)))


def clip_path(clip_id):
    for split in ("train", "test"):
        p = os.path.join(CACHE_DIR, split, clip_id + ".npz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(clip_id)


class ClipSet:
    """All clips of a pool as dense tensors. Keeps a per-clip span mask ([N, 10] float)."""

    def __init__(self, clip_ids, span_mask_file=None, device="cpu"):
        self.clip_ids = list(clip_ids)
        fields = {k: [] for k in ["motion", "music", "span_has_beat", "span_n_beats", "frame_beat_prox",
                                  "span_n_prox", "beat_frames"]}
        for cid in self.clip_ids:
            z = np.load(clip_path(cid))
            for k in fields:
                fields[k].append(z[k])
        self.beat_frames = fields.pop("beat_frames")  # ragged, stays a list of arrays
        for k, v in fields.items():
            setattr(self, k, torch.from_numpy(np.stack(v)).to(device))
        # motion [N,150,151], music [N,150,35], span_has_beat [N,10] bool, span_n_beats [N,10],
        # frame_beat_prox [N,150] bool, span_n_prox [N,10]
        if span_mask_file is None:
            mask = np.ones((len(self.clip_ids), N_SPANS), dtype=np.float32)
        else:
            m = json.load(open(span_mask_file))
            mask = np.array([m[cid] for cid in self.clip_ids], dtype=np.float32)
        assert mask.shape == (len(self.clip_ids), N_SPANS)
        self.span_mask = torch.from_numpy(mask).to(device)

    def __len__(self):
        return len(self.clip_ids)

    def mask_dict(self):
        return {cid: [bool(v) for v in row] for cid, row in zip(self.clip_ids, self.span_mask.cpu().numpy())}


def load_normalizer():
    z = np.load(os.path.join(DATA_DIR, "normalizer.npz"))
    return torch.from_numpy(z["scale"]), torch.from_numpy(z["min"])


def unnormalize(x, scale, mn):
    """Inverse of EDGE's MinMaxScaler (feature range [-1, 1], clip=True)."""
    x = torch.clip(x, -1, 1)
    return (x - mn.to(x.device)) / scale.to(x.device)
