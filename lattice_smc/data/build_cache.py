# Copied from the earlier harness: <earlier-harness>/data/build_cache.py (working tree; that repo has no git commit).
# sha256 of the source file: 4810bbe489dd6395568dd8e7d73491b91ff44e7faa5bd2b1cf2ac152cfc95cb6; EDGE pin 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications: import path of the earlier harness -> lattice_smc only. Kept for provenance of data/normalizer.* and data/manifest.json; not run in this sprint (pairs come from build_pairs.py).
"""Build the non-overlapping 5 s clip cache for one split (SPEC 2, Amendment B fields).

We load EDGE's own AISTPPDataset (which uses EDGE's 0.5 s-stride slices and fits
EDGE's MinMax normalizer on the train split) and keep only the slices whose index is a
multiple of 10: slice 10*j starts at exactly 5*j seconds, so those slices are the
stride-150 windows. This way the motion representation, the normalizer and the baseline
music features (including the beat channel) are byte-for-byte what EDGE computed.

Pools (clips_train.json / clips_heldout.json) are built afterwards by make_pools.py.

Usage:
  python -m <earlier-harness>.data.build_cache --split train
  python -m <earlier-harness>.data.build_cache --split test
"""
import argparse
import json
import os
import pickle
import re
import time

import numpy as np

from lattice_smc.edge_import import EDGE_DATA_DIR, ROOT  # noqa: F401  (sets sys.path)
from dataset.dance_dataset import AISTPPDataset  # noqa: E402

FPS = 30
CLIP_LEN = 150
N_SPANS = 10
SPAN_LEN = CLIP_LEN // N_SPANS
BEAT_PROX_RADIUS = 2  # Amendment B: frames within +-2 of a beat
DATA_DIR = os.path.join(ROOT, "data")
SLICE_RE = re.compile(r"^(.*)_slice(\d+)$")


def beat_meta(beat_frames):
    n = np.zeros(N_SPANS, dtype=np.int64)
    prox = np.zeros(CLIP_LEN, dtype=bool)
    for b in beat_frames:
        n[int(b) // SPAN_LEN] += 1
        prox[max(0, int(b) - BEAT_PROX_RADIUS): int(b) + BEAT_PROX_RADIUS + 1] = True
    span_n_prox = prox.reshape(N_SPANS, SPAN_LEN).sum(1).astype(np.int64)
    return n > 0, n, prox, span_n_prox


def load_edge_dataset(split):
    backup = os.path.join(DATA_DIR, "edge_dataset_backup")
    if split == "train":
        return AISTPPDataset(data_path=EDGE_DATA_DIR, backup_path=backup, train=True,
                             feature_type="baseline")
    normalizer = pickle.load(open(os.path.join(DATA_DIR, "normalizer.pkl"), "rb"))
    return AISTPPDataset(data_path=EDGE_DATA_DIR, backup_path=backup, train=False,
                         feature_type="baseline", normalizer=normalizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    t0 = time.time()

    ds = load_edge_dataset(args.split)
    if args.split == "train":
        pickle.dump(ds.normalizer, open(os.path.join(DATA_DIR, "normalizer.pkl"), "wb"))
        np.savez(os.path.join(DATA_DIR, "normalizer.npz"),
                 scale=ds.normalizer.scaler.scale_.numpy(), min=ds.normalizer.scaler.min_.numpy())

    out_dir = os.path.join(DATA_DIR, "cache", args.split)
    os.makedirs(out_dir, exist_ok=True)
    clips = []
    for i, fname in enumerate(ds.data["filenames"]):
        stem = os.path.splitext(os.path.basename(fname))[0]
        seq, k = SLICE_RE.match(stem).groups()
        k = int(k)
        if k % 10 != 0:
            continue
        j = k // 10
        motion = ds.data["pose"][i].numpy().astype(np.float32)
        music = np.load(fname).astype(np.float32)
        assert motion.shape == (CLIP_LEN, 151), motion.shape
        assert music.shape == (CLIP_LEN, 35), music.shape
        beat_frames = np.where(music[:, -1] > 0.5)[0].astype(np.int64)
        has_beat, n_beats, prox, span_n_prox = beat_meta(beat_frames)
        clip_id = f"{seq}_c{j}"
        np.savez(os.path.join(out_dir, clip_id + ".npz"), motion=motion, music=music,
                 beat_frames=beat_frames, span_has_beat=has_beat, span_n_beats=n_beats,
                 frame_beat_prox=prox, span_n_prox=span_n_prox,
                 source_seq=seq, start_frame=j * CLIP_LEN)
        clips.append({"clip_id": clip_id, "split": args.split, "source_seq": seq,
                      "genre": seq.split("_")[0], "music": seq.split("_")[4],
                      "start_frame": j * CLIP_LEN, "n_beats": int(len(beat_frames)),
                      "beat_frames": beat_frames.tolist(), "span_has_beat": has_beat.tolist(),
                      "span_n_beats": n_beats.tolist(), "span_n_prox": span_n_prox.tolist(),
                      "n_frames_beat_prox": int(prox.sum())})
    clips.sort(key=lambda c: c["clip_id"])

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest[args.split] = {"n_clips_all": len(clips), "clips": clips}
    json.dump(manifest, open(manifest_path, "w"))
    print(json.dumps({"split": args.split, "edge_slices": len(ds), "nonoverlap_clips": len(clips),
                      "sequences_with_clips": len({c["source_seq"] for c in clips}),
                      "clips_with_beats": sum(c["n_beats"] > 0 for c in clips),
                      "frac_spans_with_beat": float(np.mean([np.mean(c["span_has_beat"]) for c in clips])),
                      "frac_frames_beat_prox": sum(c["n_frames_beat_prox"] for c in clips) / (len(clips) * CLIP_LEN),
                      "seconds": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
