"""R1 Phase 3: K = 6 (all 40 paper prompts) and K = 8 (the 16 paper prompts whose source sequence has >= 8 chunks) prompt
lists built from the SAME source sequences as data/prompts.json, start chunk j = min(paper start, n_chunks - K) so the
longer segment contains the paper's 20 s segment where possible. Writes data/prompts_K{6,8}.json and the cache npz
data/cache/prompts/{seq}_K{K}j{j}.npz. Does not touch data/prompts.json or data/heldout_sequences_extended.json.

  .venv/bin/python scripts/r1_make_prompts_k.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", "."))
import numpy as np  # noqa: E402

from lattice_smc.data.build_pairs import load_edge_dataset, music_path, slice_index  # noqa: E402
from lattice_smc.data.make_prompts import SLICES_PER_CHUNK, genre_of  # noqa: E402
from lattice_smc.data.pairs import CACHE_DIR, CHUNK_LEN, DATA_DIR  # noqa: E402

t0 = time.time()
base = json.load(open(os.path.join(DATA_DIR, "prompts.json")))
ds = {"train": load_edge_dataset("train"), "test": load_edge_dataset("test")}
idx = {sp: slice_index(d) for sp, d in ds.items()}
out_dir = os.path.join(CACHE_DIR, "prompts")
for K in (6, 8):
    prompts = []
    for p in base["prompts"]:
        n = p["n_chunks_in_sequence"]
        if n < K:
            continue
        j = min(p["start_chunk"], n - K)
        seq, split = p["source_seq"], p["split"]
        slices = [SLICES_PER_CHUNK * (j + c) for c in range(K)]
        music = np.concatenate([np.load(music_path(split, seq, s)) for s in slices]).astype(np.float32)
        gt = np.concatenate([ds[split].data["pose"][idx[split][(seq, s)]].numpy() for s in slices]).astype(np.float32)
        assert music.shape == (K * CHUNK_LEN, 35) and gt.shape == (K * CHUNK_LEN, 151)
        beat_frames = np.where(music[:, -1] > 0.5)[0].astype(np.int64)
        pid = f"{seq}_K{K}j{j}"
        np.savez(os.path.join(out_dir, pid + ".npz"), music=music, gt_motion=gt, beat_frames=beat_frames)
        prompts.append({"prompt_id": pid, "paper_prompt_id": p["prompt_id"], "source_seq": seq, "split": split, "genre": genre_of(seq), "music": p["music"],
                        "start_chunk": j, "start_frame": j * CHUNK_LEN, "slices": slices, "n_chunks_in_sequence": n, "n_beats": int(len(beat_frames)),
                        "in_original_heldout": p["in_original_heldout"], "clip_ids": [f"{seq}_c{j + c}" for c in range(K)],
                        "contains_paper_segment": bool(j <= p["start_chunk"] and p["start_chunk"] + 4 <= j + K)})
    info = {"K": K, "built_from": "data/prompts.json (same source sequences; j = min(paper start, n_chunks - K))", "n_prompts": len(prompts),
            "prompt_pieces": sorted({p["music"] for p in prompts}), "n_prompt_pieces": len({p["music"] for p in prompts}),
            "per_genre": {g: sum(p["genre"] == g for p in prompts) for g in sorted({p["genre"] for p in prompts})},
            "n_containing_paper_segment": sum(p["contains_paper_segment"] for p in prompts), "mean_beats_per_prompt": float(np.mean([p["n_beats"] for p in prompts])),
            "seconds": round(time.time() - t0, 1), "prompts": prompts}
    json.dump(info, open(os.path.join(DATA_DIR, f"prompts_K{K}.json"), "w"), indent=1)
    print(K, {k: v for k, v in info.items() if k not in ("prompts", "prompt_pieces")})
