"""Prompts (SPEC 3.1): K x 5 s music segments from held-out sequences.

SPEC 3.1 asks for the largest K in {4, 6} with at least 40 prompts from distinct held-out
sequences (the 60 held-out train-split sequences plus the 20 test-split sequences). On AIST++ as
sliced by EDGE that is impossible: 783 of the 952 train sequences are "basic" dances (sBM) of
about 10 s, only the 169 "advanced" ones (sFM) have >= 4 chunks, and the reused held-out set
contains 9 of them (7 with >= 6 chunks). The counts are printed and written to prompts.json.

Deviation applied here (RESULTS.md Phase 0), before any training:
  * The unit of distinctness is the music piece (AIST++ has 60 pieces; 49 have recordings of
    >= 4 chunks, 40 of >= 6 chunks). --n_pieces pieces (default 20, 2 per genre, seed 0) among
    the pieces with >= 6 chunks are the PROMPT pieces; the other long pieces are reserved for
    twist rollouts (Phase 2a), so twist music and prompt music are disjoint at the piece level.
  * For each prompt piece, --per_piece sequences with >= 6 chunks are held out (existing
    held-out sequences of that piece first, then train-split sequences chosen with the rng).
    The union with the original 60 is written to data/heldout_sequences_extended.json and is the
    set excluded from the training pairs. The original file is left untouched.
  * K defaults to 4. With sequences of >= 6 chunks held out, a K = 6 list can be built later
    from the same held-out set without retraining (--K 6 --out prompts_K6.json).
  * Start windows on the 5 s grid: the first sequence of a piece starts at chunk 0, the second
    at the last valid start (n_chunks - K), a third (if needed) in the middle, so two prompts of
    one piece share as little music as the recording allows.

  python -m lattice_smc.data.make_prompts --K 4 --min_prompts 40

Writes data/prompts.json, data/heldout_sequences_extended.json and
data/cache/prompts/{prompt_id}.npz (music [K*150, 35], gt_motion [K*150, 151] normalized,
beat_frames).
"""
import argparse
import json
import os
import time

import numpy as np

from lattice_smc.data.build_pairs import (CHUNK_LEN, DATA_DIR, CACHE_DIR, SLICES_PER_CHUNK, SLICE_RE,
                                          load_edge_dataset, music_path, slice_index)
from lattice_smc.edge_import import EDGE_DATA_DIR

K_CANDIDATES = [4, 6]
LONG_CHUNKS = 6  # a "long" piece / sequence has >= 6 chunks on the 5 s grid, so K = 4 and 6 both fit


def sequences_and_slice_counts(split):
    counts = {}
    for f in os.listdir(os.path.join(EDGE_DATA_DIR, split, "baseline_feats")):
        seq, k = SLICE_RE.match(os.path.splitext(f)[0]).groups()
        counts[seq] = max(counts.get(seq, 0), int(k) + 1)
    return counts


def n_grid_chunks(n_slices):
    return (n_slices - 1) // SLICES_PER_CHUNK + 1


def piece_of(seq):
    return seq.split("_")[4]


def genre_of(seq):
    return seq.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--min_prompts", type=int, default=40)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--n_pieces", type=int, default=20)
    ap.add_argument("--per_piece", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prompts.json")
    args = ap.parse_args()
    t0 = time.time()
    K = args.K

    heldout_orig = sorted(json.load(open(os.path.join(DATA_DIR, "heldout_sequences.json"))))
    counts = {"train": sequences_and_slice_counts("train"), "test": sequences_and_slice_counts("test")}
    split_of = {s: sp for sp in counts for s in counts[sp]}
    chunks_of = {s: n_grid_chunks(n) for sp in counts for s, n in counts[sp].items()}
    heldout_set = set(heldout_orig) | set(counts["test"])

    # SPEC 3.1 rule on the reused held-out set (reported; it fails for both K)
    spec_counts = {k: sum(chunks_of[s] >= k for s in heldout_set) for k in K_CANDIDATES}
    pieces_long = sorted({piece_of(s) for s in chunks_of if chunks_of[s] >= LONG_CHUNKS})
    pieces_ge = {k: sorted({piece_of(s) for s in chunks_of if chunks_of[s] >= k}) for k in K_CANDIDATES}

    rng = np.random.default_rng(args.seed)
    # prompt pieces: n_pieces among the long pieces, as even over genres as possible
    by_genre = {}
    for p in pieces_long:
        by_genre.setdefault("g" + p[1:3], []).append(p)
    for g in by_genre:
        by_genre[g] = [by_genre[g][i] for i in rng.permutation(len(by_genre[g]))]
    prompt_pieces = []
    while len(prompt_pieces) < min(args.n_pieces, len(pieces_long)):
        for g in sorted(by_genre):
            if by_genre[g] and len(prompt_pieces) < args.n_pieces:
                prompt_pieces.append(by_genre[g].pop(0))
    prompt_pieces = sorted(prompt_pieces)
    twist_pieces = sorted(set(pieces_ge[4]) - set(prompt_pieces))

    # held-out sequences per prompt piece: existing held-out ones first, then train-split ones
    chosen = []
    extra_heldout = []
    for p in prompt_pieces:
        cands = sorted(s for s in chunks_of if piece_of(s) == p and chunks_of[s] >= LONG_CHUNKS)
        already = [s for s in cands if s in heldout_set]
        fresh = [s for s in cands if s not in heldout_set]
        fresh = [fresh[i] for i in rng.permutation(len(fresh))]
        take = (already + fresh)[: args.per_piece]
        extra_heldout += [s for s in take if s not in heldout_set]
        chosen += [(p, s) for s in take]
    assert len(chosen) >= args.min_prompts, f"only {len(chosen)} prompts; raise --n_pieces or --per_piece"
    chosen = chosen[: args.n_prompts]
    heldout_ext = sorted(set(heldout_orig) | set(extra_heldout))
    json.dump(heldout_ext, open(os.path.join(DATA_DIR, "heldout_sequences_extended.json"), "w"), indent=0)

    ds = {"train": load_edge_dataset("train")}
    if any(split_of[s] == "test" for _, s in chosen):
        ds["test"] = load_edge_dataset("test")
    idx = {sp: slice_index(d) for sp, d in ds.items()}
    out_dir = os.path.join(CACHE_DIR, "prompts")
    os.makedirs(out_dir, exist_ok=True)
    prompts = []
    rank_in_piece = {}
    for p, seq in chosen:
        r = rank_in_piece.get(p, 0)
        rank_in_piece[p] = r + 1
        n_chunks = chunks_of[seq]
        j = [0, n_chunks - K, (n_chunks - K) // 2][r] if r < 3 else int(rng.integers(n_chunks - K + 1))
        split = split_of[seq]
        slices = [SLICES_PER_CHUNK * (j + c) for c in range(K)]
        music = np.concatenate([np.load(music_path(split, seq, s)) for s in slices]).astype(np.float32)
        gt = np.concatenate([ds[split].data["pose"][idx[split][(seq, s)]].numpy() for s in slices]).astype(np.float32)
        assert music.shape == (K * CHUNK_LEN, 35) and gt.shape == (K * CHUNK_LEN, 151)
        beat_frames = np.where(music[:, -1] > 0.5)[0].astype(np.int64)
        pid = f"{seq}_j{j}"
        np.savez(os.path.join(out_dir, pid + ".npz"), music=music, gt_motion=gt, beat_frames=beat_frames)
        prompts.append({"prompt_id": pid, "source_seq": seq, "split": split, "genre": genre_of(seq),
                        "music": p, "start_chunk": j, "start_frame": j * CHUNK_LEN, "slices": slices,
                        "n_chunks_in_sequence": n_chunks, "n_beats": int(len(beat_frames)),
                        "in_original_heldout": seq in heldout_set,
                        "clip_ids": [f"{seq}_c{j + c}" for c in range(K)]})

    info = {"K": K, "seed": args.seed, "min_prompts": args.min_prompts, "n_prompts": len(prompts),
            "spec_rule_candidates_per_K_original_heldout": {str(k): v for k, v in spec_counts.items()},
            "spec_rule_satisfied": {str(k): v >= args.min_prompts for k, v in spec_counts.items()},
            "n_pieces_total": len({piece_of(s) for s in chunks_of}),
            "n_pieces_with_ge_chunks": {str(k): len(v) for k, v in pieces_ge.items()},
            "n_sequences_with_ge_chunks": {str(k): sum(chunks_of[s] >= k for s in chunks_of) for k in K_CANDIDATES},
            "prompt_pieces": prompt_pieces, "twist_pieces": twist_pieces,
            "n_prompt_pieces": len(prompt_pieces), "n_twist_pieces": len(twist_pieces),
            "heldout_original": len(heldout_orig), "heldout_added": sorted(extra_heldout),
            "n_heldout_added": len(extra_heldout), "n_heldout_extended": len(heldout_ext),
            "n_from_original_heldout_or_test": sum(p["in_original_heldout"] for p in prompts),
            "n_from_test_split": sum(p["split"] == "test" for p in prompts),
            "per_genre": {g: sum(p["genre"] == g for p in prompts) for g in sorted({p["genre"] for p in prompts})},
            "n_distinct_sequences": len({p["source_seq"] for p in prompts}),
            "n_distinct_music_segments": len({(p["music"], p["start_chunk"]) for p in prompts}),
            "mean_beats_per_prompt": float(np.mean([p["n_beats"] for p in prompts])),
            "seconds": round(time.time() - t0, 1), "prompts": prompts}
    json.dump(info, open(os.path.join(DATA_DIR, args.out), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "prompts"}, indent=2))


if __name__ == "__main__":
    main()
