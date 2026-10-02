"""Twist rollouts (SPEC 5): base-model sequences on training-pool music segments of the twist
pieces (never held-out music), prefixes at chunk index k in 1..K-1, M = 8 base-model
continuations to the end per prefix, per-continuation future reward sum_{j > k} r_j (additive
R_BA terms, Amendment I pipeline). Targets log mean exp(S / alpha) are formed at training time so
the rollouts do not depend on alpha.

Segments: every (train-split sequence of a twist piece with >= K chunks, grid start j) pair,
music from EDGE's baseline features (same as the prompts). Base sequence i uses segment i mod
n_segments with noise seed 10000 + i ("train" set) or 30000 + i ("fresh" set); continuation
particles of prefix (i, k) use noise seed 20000 + 4i + k (train) or 40000 + 4i + k (fresh), so
no noise stream is shared with the prefix or the prompts (seeds 2000..2003).

For the fresh set the plug-in noise-axis potential is also recorded: on continuation 0 of each
prefix, r_{k+1} of the x0 prediction after DDIM steps 45, 35, 25, 15 and 5 of chunk k + 1.

  python -m lattice_smc.twist.collect --ckpt runs/chunk_dispE/ckpt_adopted.pt --set train --n_prefixes 4000 --M 8 --shard 0/2
  python -m lattice_smc.twist.collect --ckpt ... --set fresh --n_prefixes 500 --M 8 --shard 0/2

Writes data/twist/segments.json, data/twist/rollouts/{set}/{i:05d}.npz (one per base sequence).
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lattice_smc.data.build_pairs import music_path  # noqa: E402
from lattice_smc.data.make_prompts import n_grid_chunks, piece_of, sequences_and_slice_counts  # noqa: E402
from lattice_smc.data.pairs import CHUNK_LEN, DATA_DIR  # noqa: E402
from lattice_smc.edge_import import ROOT  # noqa: E402
from lattice_smc.generate import Context, set_deterministic  # noqa: E402
from lattice_smc.methods.common import evaluator  # noqa: E402
from lattice_smc.model.sampler import chunk_noise, sample_chunk, window_music  # noqa: E402
from lattice_smc.model.small import load_checkpoint  # noqa: E402

TWIST_DIR = os.path.join(DATA_DIR, "twist")  # default; Amendment M uses data/twist_rep
PLUGIN_STEPS = (45, 35, 25, 15, 5)  # 1-based DDIM step indices


def build_segments(K):
    prompts = json.load(open(os.path.join(DATA_DIR, "prompts.json")))
    twist_pieces = set(prompts["twist_pieces"])
    heldout = set(json.load(open(os.path.join(DATA_DIR, "heldout_sequences_extended.json"))))
    counts = sequences_and_slice_counts("train")
    segs = []
    for seq in sorted(counts):
        if seq in heldout or piece_of(seq) not in twist_pieces:
            continue
        n = n_grid_chunks(counts[seq])
        for j in range(n - K + 1):
            segs.append({"seq": seq, "piece": piece_of(seq), "start_chunk": j, "slices": [10 * (j + c) for c in range(K)]})
    return segs


def load_segments(K, twist_dir=TWIST_DIR):
    path = os.path.join(twist_dir, "segments.json")
    if os.path.exists(path):
        info = json.load(open(path))
        return info["segments"], torch.from_numpy(np.load(os.path.join(twist_dir, "segments_music.npy")))
    segs = build_segments(K)
    music = np.stack([np.concatenate([np.load(music_path("train", s["seq"], sl)) for sl in s["slices"]]) for s in segs]).astype(np.float32)
    os.makedirs(twist_dir, exist_ok=True)
    np.save(os.path.join(twist_dir, "segments_music.npy"), music)
    json.dump({"K": K, "n_segments": len(segs), "n_pieces": len({s["piece"] for s in segs}),
               "n_sequences": len({s["seq"] for s in segs}), "n_distinct_music": len({(s["piece"], s["start_chunk"]) for s in segs}),
               "segments": segs}, open(path, "w"), indent=1)
    return segs, torch.from_numpy(music)


def rollout_sequence(ctx, music, K, i, seed_base, seed_cont_base, M, plugin):
    """Base sequence i on music [K*150, 35]; prefixes k = 1..K-1 with M continuations each."""
    dev = ctx.device
    ev = evaluator(ctx, music)
    chunks = []
    prev = None
    for k in range(1, K + 1):
        x = sample_chunk(ctx.model, ctx.diffusion, prev, window_music(music, k, 1), chunk_noise(seed_base, k, range(1), device=dev),
                         ctx.counters, cfg=ctx.loss_cfg)
        chunks.append(x)
        prev = x
    base_motion = torch.cat(chunks, dim=1)  # [1, K*150, 151]
    r_base = np.array([ev.r_k(ev.joints(base_motion), k)[0] for k in range(1, K + 1)])
    out = {"base_motion": base_motion[0].cpu().numpy(), "r_base": r_base, "prefix_k": [], "r_future": [], "plugin": []}
    for k in range(1, K):
        seed_c = seed_cont_base + 4 * i + k
        cont = [c.expand(M, -1, -1).contiguous() for c in chunks[:k]]  # the M particles share the prefix
        r_fut = np.zeros((M, K - k))
        plug = np.full(len(PLUGIN_STEPS), np.nan)
        for j in range(k + 1, K + 1):
            hook = None
            if plugin and j == k + 1:
                def hook(step, time_, x0_abs, k=k, j=j, cont=cont):
                    if step + 1 in PLUGIN_STEPS:
                        J = ev.joints(torch.cat([c[:1] for c in cont] + [x0_abs[:1]], dim=1))
                        plug[PLUGIN_STEPS.index(step + 1)] = ev.r_k(J, j)[0]
                    return None
            x = sample_chunk(ctx.model, ctx.diffusion, cont[-1], window_music(music, j, M), chunk_noise(seed_c, j, range(M), device=dev),
                             ctx.counters, on_x0_pred=hook, cfg=ctx.loss_cfg)
            cont.append(x)
            r_fut[:, j - k - 1] = ev.r_k(ev.joints(torch.cat(cont, dim=1)), j)
        out["prefix_k"].append(k)
        out["r_future"].append(r_fut)
        out["plugin"].append(plug)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default="train", choices=["train", "fresh"])
    ap.add_argument("--n_prefixes", type=int, default=4000)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reward", default="ba", choices=["ba", "rep"])
    ap.add_argument("--twist_dir", default=None, help="default data/twist (ba) or data/twist_rep (rep)")
    ap.add_argument("--reverse", action="store_true", help="iterate indices in reverse (a second worker meeting the first in the middle)")
    args = ap.parse_args()
    set_deterministic()
    dev = torch.device(args.device)
    model, diffusion, ck = load_checkpoint(args.ckpt, dev)
    ctx = Context(model, diffusion, dev, ck["config"], reward=args.reward)
    K = 4
    twist_dir = args.twist_dir or (TWIST_DIR if args.reward == "ba" else os.path.join(DATA_DIR, "twist_rep"))
    segs, music_all = load_segments(K, twist_dir)
    n_seq = -(-args.n_prefixes // (K - 1))  # ceil
    seed_base, seed_cont = (10000, 20000) if args.set == "train" else (30000, 40000)
    out_dir = os.path.join(twist_dir, "rollouts", args.set)
    os.makedirs(out_dir, exist_ok=True)
    si, sn = (int(v) for v in args.shard.split("/"))
    todo = [i for i in range(n_seq) if i % sn == si]
    if args.reverse:
        todo = todo[::-1]
    print(f"set={args.set} reward={args.reward} dir={twist_dir} segments={len(segs)} sequences={n_seq} prefixes={n_seq * (K - 1)} (first {args.n_prefixes} used) "
          f"M={args.M} shard={args.shard} jobs={len(todo)} device={dev}", flush=True)
    t0 = time.time()
    n_done = 0
    for i in todo:
        path = os.path.join(out_dir, f"{i:05d}.npz")
        if os.path.exists(path):
            continue
        seg = i % len(segs)
        ctx.counters.reset()
        res = rollout_sequence(ctx, music_all[seg].to(dev), K, i, seed_base + i, seed_cont, args.M, plugin=(args.set == "fresh"))
        np.savez(path, base_motion=res["base_motion"], r_base=res["r_base"], prefix_k=np.array(res["prefix_k"]),
                 r_future=np.array([np.pad(r, ((0, 0), (0, K - 1 - r.shape[1])), constant_values=np.nan) for r in res["r_future"]]),
                 plugin=np.array(res["plugin"]), segment=seg, seed_base=seed_base + i, seed_cont=seed_cont, M=args.M, reward=args.reward,
                 nfe=ctx.counters.nfe.n, reward_evals=ctx.counters.reward_evals.n)
        n_done += 1
        if n_done % 20 == 0:
            print(f"{n_done}/{len(todo)} sequences, {time.time() - t0:.0f}s, last nfe={ctx.counters.nfe.n}", flush=True)
    print(f"done: {n_done} sequences in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
