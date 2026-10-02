# Copied from the earlier harness: <earlier-harness>/data/make_pools.py (working tree; that repo has no git commit).
# sha256 of the source file: d2bdc63a9a3dd978409d405d0162ee49c2677e16384557acd8241ad9b1f41efb; EDGE pin 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications: import path of the earlier harness -> lattice_smc only. Kept for provenance of data/heldout_sequences.json (60 sequences, 6 per genre, seed 0); not run in this sprint.
"""Amendment A: sequence-level hold-out from the train split, then the two clip pools.

  python -m <earlier-harness>.data.make_pools --per_genre 6 --pool 2000 --seed 0

Writes data/heldout_sequences.json, data/clips_train.json, data/clips_heldout.json and
adds a "pools" entry to data/manifest.json. Sequence choice and pool subsampling both use
numpy's default_rng(seed).
"""
import argparse
import json
import os

import numpy as np

from lattice_smc.edge_import import ROOT

DATA_DIR = os.path.join(ROOT, "data")
CLIP_LEN = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_genre", type=int, default=6)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json")))
    train_clips = manifest["train"]["clips"]
    test_clips = manifest["test"]["clips"]

    rng = np.random.default_rng(args.seed)
    by_genre = {}
    for c in train_clips:
        by_genre.setdefault(c["genre"], set()).add(c["source_seq"])
    heldout_seqs = []
    for g in sorted(by_genre):
        seqs = sorted(by_genre[g])
        heldout_seqs += sorted(rng.choice(seqs, size=args.per_genre, replace=False).tolist())
    heldout_set = set(heldout_seqs)

    heldout_from_train = sorted(c["clip_id"] for c in train_clips if c["source_seq"] in heldout_set)
    heldout_ids = heldout_from_train + sorted(c["clip_id"] for c in test_clips)
    remaining = sorted(c["clip_id"] for c in train_clips if c["source_seq"] not in heldout_set)
    if len(remaining) > args.pool:
        train_ids = sorted(rng.choice(remaining, size=args.pool, replace=False).tolist())
    else:
        train_ids = remaining

    json.dump(heldout_seqs, open(os.path.join(DATA_DIR, "heldout_sequences.json"), "w"), indent=0)
    json.dump(train_ids, open(os.path.join(DATA_DIR, "clips_train.json"), "w"), indent=0)
    json.dump(heldout_ids, open(os.path.join(DATA_DIR, "clips_heldout.json"), "w"), indent=0)

    by_id = {c["clip_id"]: c for c in train_clips + test_clips}
    train_music = {by_id[i]["music"] for i in train_ids}
    heldout_music = {by_id[i]["music"] for i in heldout_ids}
    prox_frames = sum(by_id[i]["n_frames_beat_prox"] for i in heldout_ids)
    info = {"seed": args.seed, "per_genre": args.per_genre, "n_heldout_sequences": len(heldout_seqs),
            "genres": sorted(by_genre), "n_train_sequences_total": len({c["source_seq"] for c in train_clips}),
            "n_train_clips_remaining": len(remaining), "n_train_pool": len(train_ids),
            "n_heldout_from_train": len(heldout_from_train), "n_heldout_from_test": len(test_clips),
            "n_heldout_pool": len(heldout_ids),
            "n_music_pieces_train": len(train_music), "n_music_pieces_heldout": len(heldout_music),
            "n_music_pieces_shared": len(train_music & heldout_music),
            "heldout_frac_frames_beat_prox": prox_frames / (len(heldout_ids) * CLIP_LEN),
            "heldout_frac_spans_with_beat": float(np.mean([np.mean(by_id[i]["span_has_beat"]) for i in heldout_ids])),
            "train_pool_frac_frames_beat_prox": sum(by_id[i]["n_frames_beat_prox"] for i in train_ids) / (len(train_ids) * CLIP_LEN)}
    manifest["pools"] = info
    json.dump(manifest, open(os.path.join(DATA_DIR, "manifest.json"), "w"))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
