"""M3 scoring of the returned clips (runs in the Stable Audio 3 venv, GPU): for every sequence directory
under --samples (method/N/pXX_sYYYY): (a) root-of-trust: the terminal CLAP recomputed from the saved
decoded audio `seq_chunk03.wav` with the M2 protocol (m1/rewards, non-overlapping 10 s windows), compared
with the `final_reward` in log.json; (b) the held-out 8 kHz energy fraction; (c) the seam flux ratio.
Resumable: rows already in --out are kept.

  .venv/bin/python m3/score_samples.py --samples results/m3/samples --out results/m3/scores.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1")
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m2")
import rewards as R  # noqa: E402
from score_clap import spec_stats  # noqa: E402



def returned_wav(sd):
    """The returned clip: seq_chunk03.wav in the K = 4 grids, seq_chunk{K-1:02d}.wav in the R1 K = 8 runs."""
    p = os.path.join(sd, "seq_chunk03.wav")
    if os.path.exists(p):
        return p
    c = sorted(x for x in os.listdir(sd) if x.startswith("seq_chunk") and x.endswith(".wav") and "argmax" not in x)
    return os.path.join(sd, c[-1])

def seq_dirs(root):
    out = []
    for method in sorted(os.listdir(root)):
        mdir = os.path.join(root, method)
        if not os.path.isdir(mdir):
            continue
        for N in sorted(os.listdir(mdir), key=int):
            for d in sorted(os.listdir(os.path.join(mdir, N))):
                sd = os.path.join(mdir, N, d)
                if os.path.exists(os.path.join(sd, "log.json")):
                    out.append((method, int(N), d, sd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_seam", action="store_true")
    a = ap.parse_args()
    rows = {}
    if os.path.exists(a.out):
        rows = {r["key"]: r for r in json.load(open(a.out))["rows"]}
    # the in-run scores were computed with the flags Stable Audio 3 sets at load time (TF32 off for matmul and cudnn);
    # the recompute uses the same flags so that the root-of-trust comparison is a test of the audio, not of kernel precision
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    clap = R.ClapScorer(device=a.device)
    import soundfile as sf
    todo = [x for x in seq_dirs(a.samples) if f"{x[0]}/{x[1]}/{x[2]}" not in rows]
    print(f"{len(rows)} scored, {len(todo)} to do", flush=True)
    for i, (method, N, d, sd) in enumerate(todo):
        log = json.load(open(os.path.join(sd, "log.json")))
        wav = returned_wav(sd)
        m48 = R.load_mono48k(wav)
        t = clap.text_embed([log["prompt"]])[0]
        E = clap.window_embeds(m48, log["K"])
        mv = E.mean(axis=0)
        term = float((mv / np.linalg.norm(mv)) @ t)
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        motif = float(E[log["K"] - 1] @ E[0])  # Amendment V: R_motif recomputed from the saved audio (root of trust for the m4 grid)
        logged_is_motif = log.get("reward") == "motif"
        row = {"key": f"{method}/{N}/{d}", "method": method, "N": N, "prompt_id": log["prompt_id"], "base_seed": log["base_seed"],
               "final_reward_logged": log["final_reward"], "reward_logged_name": log.get("reward", "clap"), "clap_terminal_recomputed": term,
               "motif_recomputed": motif, "s3_recomputed": float(E[2] @ E[0]),
               "clap_abs_diff": abs(term - log["final_reward"]) if not logged_is_motif else None,
               "motif_abs_diff": abs(motif - log["final_reward"]) if logged_is_motif else None, "clap_per_window": [float(x) for x in (E @ t)],
               **spec_stats(mono, sr)}
        if not a.no_seam:
            seam = R.seam_stats(mono, sr, [10.0 * i for i in range(1, log["K"])])
            row["seam_mean_ratio"] = seam["mean_ratio"]
        rows[row["key"]] = row
        if i % 50 == 0:
            print(i, row["key"], round(term, 4), row["clap_abs_diff"], row["motif_abs_diff"], flush=True)
            json.dump({"rows": list(rows.values())}, open(a.out, "w"))
    json.dump({"rows": list(rows.values()), "clap_ckpt": R.CLAP_REPO}, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(rows), flush=True)


if __name__ == "__main__":
    main()
