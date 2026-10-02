"""Build the (prefix, chunk) training pairs (SPEC 2.2) from EDGE's 0.5 s-stride slices.

EDGE slice j of a sequence covers 30 fps frames [15j, 15j + 150), so slice j and slice j + 10 are
consecutive 5 s windows and form one pair: frames 0..149 = prefix, 150..299 = chunk, with the
music of the whole 10 s span (the two slices' baseline features concatenated). Motion is EDGE's
normalized 151-dim representation exactly as EDGE's AISTPPDataset produces it (backup pickle
from the earlier harness Phase 0 run, train normalizer), so nothing is recomputed.

Held-out sequences (data/heldout_sequences.json) contribute no training pair. Their pairs go to
a separate file that is used only as a loss monitor during training (never for training, twist
data or steering).

  python -m lattice_smc.data.build_pairs --heldout data/heldout_sequences_extended.json

(the extended held-out list is written by make_prompts.py, which therefore runs first)

Writes data/pairs.json, data/pairs_heldout.json, data/cache/pairs_train.npz,
data/cache/pairs_heldout.npz and adds a "pairs" entry to data/manifest.json.
"""
import argparse
import json
import os
import re
import time

import numpy as np

from lattice_smc.edge_import import EDGE_DATA_DIR, ROOT  # noqa: F401  (sets sys.path)
from dataset.dance_dataset import AISTPPDataset  # noqa: E402

FPS = 30
CHUNK_LEN = 150
SLICE_STRIDE_FRAMES = 15  # 0.5 s at 30 fps
SLICES_PER_CHUNK = CHUNK_LEN // SLICE_STRIDE_FRAMES  # 10
REPR_DIM = 151
MUSIC_DIM = 35
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SLICE_RE = re.compile(r"^(.*)_slice(\d+)$")


def load_edge_dataset(split):
    backup = os.path.join(DATA_DIR, "edge_dataset_backup")
    if split == "train":
        return AISTPPDataset(data_path=EDGE_DATA_DIR, backup_path=backup, train=True,
                             feature_type="baseline")
    import pickle
    normalizer = pickle.load(open(os.path.join(DATA_DIR, "normalizer.pkl"), "rb"))
    return AISTPPDataset(data_path=EDGE_DATA_DIR, backup_path=backup, train=False,
                         feature_type="baseline", normalizer=normalizer)


def slice_index(ds):
    """(sequence, slice number) -> row of ds.data['pose'], from EDGE's file names."""
    idx = {}
    for i, fname in enumerate(ds.data["filenames"]):
        seq, k = SLICE_RE.match(os.path.splitext(os.path.basename(fname))[0]).groups()
        idx[(seq, int(k))] = i
    return idx


def music_path(split, seq, k):
    # Resolved through our own edge/data mount, not the absolute paths stored in the pickle.
    return os.path.join(EDGE_DATA_DIR, split, "baseline_feats", f"{seq}_slice{k}.npy")


def build(ds, split, seqs, idx):
    """All (j, j+10) pairs of the given sequences. Returns (records, motion, music)."""
    recs, motions, musics = [], [], []
    by_seq = {}
    for (seq, k) in idx:
        by_seq.setdefault(seq, []).append(k)
    for seq in sorted(seqs):
        for k in sorted(by_seq.get(seq, [])):
            if (seq, k + SLICES_PER_CHUNK) not in idx:
                continue
            i0, i1 = idx[(seq, k)], idx[(seq, k + SLICES_PER_CHUNK)]
            motion = np.concatenate([ds.data["pose"][i0].numpy(), ds.data["pose"][i1].numpy()]).astype(np.float32)
            music = np.concatenate([np.load(music_path(split, seq, k)),
                                    np.load(music_path(split, seq, k + SLICES_PER_CHUNK))]).astype(np.float32)
            assert motion.shape == (2 * CHUNK_LEN, REPR_DIM), motion.shape
            assert music.shape == (2 * CHUNK_LEN, MUSIC_DIM), music.shape
            recs.append({"pair_id": f"{seq}_p{k}", "source_seq": seq, "genre": seq.split("_")[0],
                         "music": seq.split("_")[4], "prefix_slice": k, "chunk_slice": k + SLICES_PER_CHUNK,
                         "start_frame": k * SLICE_STRIDE_FRAMES,
                         "n_beats_chunk": int((music[CHUNK_LEN:, -1] > 0.5).sum())})
            motions.append(motion)
            musics.append(music)
    return recs, np.stack(motions), np.stack(musics)


def seam_check(motion, channels=slice(4, None)):
    """The seam (frame 149 -> 150) must look like any other frame step if the slice arithmetic
    is right. Ratio of mean |delta| at the seam to the mean within-window |delta|, on the given
    channels. The 4 contact channels are excluded by default: EDGE's AISTPPDataset zero-pads the
    foot velocity at the last frame of every slice, so all four contact labels are 1 at frame
    149 of every prefix and frame 299 of every chunk (an EDGE artifact, recorded in RESULTS.md),
    which makes the contact-channel seam jump about 4.7x the within-window jump."""
    d = np.abs(np.diff(motion[..., channels], axis=1)).mean(-1)  # [P, 299]
    seam = d[:, CHUNK_LEN - 1].mean()
    within = np.concatenate([d[:, :CHUNK_LEN - 1], d[:, CHUNK_LEN:]], 1).mean()
    return float(seam / within)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=os.path.join(DATA_DIR, "heldout_sequences_extended.json"))
    args = ap.parse_args()
    t0 = time.time()
    heldout = set(json.load(open(args.heldout)))

    ds = load_edge_dataset("train")
    idx = slice_index(ds)
    all_seqs = {seq for seq, _ in idx}
    train_seqs = sorted(all_seqs - heldout)
    heldout_seqs = sorted(all_seqs & heldout)
    assert len(heldout_seqs) == len(heldout), (len(heldout_seqs), len(heldout))

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = {}
    for name, seqs in [("train", train_seqs), ("heldout", heldout_seqs)]:
        recs, motion, music = build(ds, "train", seqs, idx)
        ratio = seam_check(motion)
        ratio_all = seam_check(motion, slice(None))
        ratio_contacts = seam_check(motion, slice(0, 4))
        assert 0.8 < ratio < 1.25, f"seam ratio {ratio} (channels 4:): slice arithmetic is wrong"
        np.savez(os.path.join(CACHE_DIR, f"pairs_{name}.npz"), motion=motion, music=music,
                 pair_id=np.array([r["pair_id"] for r in recs]))
        fn = "pairs.json" if name == "train" else "pairs_heldout.json"
        json.dump(recs, open(os.path.join(DATA_DIR, fn), "w"))
        per_genre = {}
        for r in recs:
            per_genre[r["genre"]] = per_genre.get(r["genre"], 0) + 1
        out[name] = {"n_pairs": len(recs), "n_sequences": len({r["source_seq"] for r in recs}),
                     "n_sequences_listed": len(seqs), "per_genre": per_genre,
                     "seam_ratio_no_contacts": round(ratio, 4), "seam_ratio_all_channels": round(ratio_all, 4),
                     "seam_ratio_contacts_only": round(ratio_contacts, 4),
                     "pairs_with_chunk_beats": sum(r["n_beats_chunk"] > 0 for r in recs),
                     "mean_beats_per_chunk": float(np.mean([r["n_beats_chunk"] for r in recs])),
                     "cache_bytes": motion.nbytes + music.nbytes}
    out["edge_train_slices"] = len(ds)
    out["n_train_sequences_total"] = len(all_seqs)
    out["n_heldout_sequences"] = len(heldout)
    out["seconds"] = round(time.time() - t0, 1)

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest["pairs"] = out
    json.dump(manifest, open(manifest_path, "w"))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
