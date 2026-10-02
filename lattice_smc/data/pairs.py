"""In-memory loaders for the pair cache and the prompt cache. Nothing here touches EDGE files."""
import json
import os

import numpy as np
import torch

from lattice_smc.edge_import import ROOT

DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CHUNK_LEN = 150
WINDOW = 300


class PairSet:
    """motion [P, 300, 151], music [P, 300, 35] as tensors on `device`."""

    def __init__(self, name="pairs_train", device="cpu", n=None):
        z = np.load(os.path.join(CACHE_DIR, f"{name}.npz"))
        sl = slice(None) if n is None else slice(0, n)
        self.pair_ids = [str(s) for s in z["pair_id"][sl]]
        self.motion = torch.from_numpy(z["motion"][sl]).to(device)
        self.music = torch.from_numpy(z["music"][sl]).to(device)

    def __len__(self):
        return len(self.pair_ids)


class PromptSet:
    """All prompts of data/prompts.json: music [n, K*150, 35], gt [n, K*150, 151], beat frames."""

    def __init__(self, path=None, device="cpu"):
        # R1 Phase 3 (2026-09-19): LATTICE_PROMPTS selects another prompt file (K = 6 / 8 lists); default unchanged
        info = json.load(open(path or os.environ.get("LATTICE_PROMPTS") or os.path.join(DATA_DIR, "prompts.json")))
        self.K = int(info["K"])
        self.info = info
        self.prompts = info["prompts"]
        self.prompt_ids = [p["prompt_id"] for p in self.prompts]
        music, gt, beats = [], [], []
        for p in self.prompts:
            z = np.load(os.path.join(CACHE_DIR, "prompts", p["prompt_id"] + ".npz"))
            music.append(z["music"])
            gt.append(z["gt_motion"])
            beats.append(z["beat_frames"])
        self.music = torch.from_numpy(np.stack(music)).to(device)
        self.gt_motion = torch.from_numpy(np.stack(gt)).to(device)
        self.beat_frames = beats

    def __len__(self):
        return len(self.prompts)

    def index(self, prompt_id):
        return self.prompt_ids.index(prompt_id)
