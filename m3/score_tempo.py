"""M3 held-out tempo reward (runs in the M0 analysis venv, beat_this): for every sequence directory under
--samples whose N is in --Ns (plus base): beat_this with explicit 22.05 kHz resampling on the whole
returned 40 s clip and on each of its four 10 s windows (windows of the FINAL clip, since the returned
sample is one decoded clip; M2's per-chunk numbers used the causal decodes and are not directly
comparable). Per-chunk score 1[within 5 % of the prompted BPM]; additive reward = mean over chunks.
HELD OUT: never used for steering. Resumable.

  m0/.venv/bin/python m3/score_tempo.py --samples results/m3/samples --out results/m3/tempo.json --Ns 8,32 [--device cuda]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1")
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m3")
import rewards as R  # noqa: E402
from score_samples import seq_dirs  # noqa: E402



def returned_wav(sd):
    """The returned clip: seq_chunk03.wav in the K = 4 grids, seq_chunk{K-1:02d}.wav in the R1 K = 8 runs."""
    p = os.path.join(sd, "seq_chunk03.wav")
    if os.path.exists(p):
        return p
    c = sorted(x for x in os.listdir(sd) if x.startswith("seq_chunk") and x.endswith(".wav") and "argmax" not in x)
    return os.path.join(sd, c[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ns", default="8,32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk_sec", type=float, default=10.0)
    a = ap.parse_args()
    Ns = {int(x) for x in a.Ns.split(",")}
    rows = {}
    if os.path.exists(a.out):
        rows = {r["key"]: r for r in json.load(open(a.out))["rows"]}
    if a.device != "cpu":
        from beat_this.inference import Audio2Beats
        R._BT["min"] = Audio2Beats(checkpoint_path="final0", device=a.device, dbn=False)
    import soundfile as sf
    todo = [x for x in seq_dirs(a.samples) if (x[1] in Ns or x[0] == "base") and f"{x[0]}/{x[1]}/{x[2]}" not in rows]
    print(f"{len(rows)} scored, {len(todo)} to do", flush=True)
    for i, (method, N, d, sd) in enumerate(todo):
        log = json.load(open(os.path.join(sd, "log.json")))
        bpm = log["bpm"]
        y, sr = sf.read(returned_wav(sd), dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        bt = R.beatthis_tempo(mono, sr, dbn=False, resample_to=22050)
        adh = R.tempo_adherence(bt.get("tempo_bpm_median_interval"), bpm)
        pc = []
        for j in range(log["K"]):
            seg = mono[int(j * a.chunk_sec * sr): int((j + 1) * a.chunk_sec * sr)]
            b = R.beatthis_tempo(seg, sr, dbn=False, resample_to=22050)
            ad = R.tempo_adherence(b.get("tempo_bpm_median_interval"), bpm)
            pc.append({"chunk": j + 1, "n_beats": b["n_beats"], "tempo_bpm": b.get("tempo_bpm_median_interval"),
                       "score": (1.0 if ad["within_5pct"] else 0.0) if ad else None,
                       "rel_err": ad["rel_err"] if ad else None})
        sc = [x["score"] for x in pc if x["score"] is not None]
        rows[f"{method}/{N}/{d}"] = {"key": f"{method}/{N}/{d}", "method": method, "N": N, "prompt_id": log["prompt_id"], "base_seed": log["base_seed"],
                                     "requested_bpm": bpm, "whole_tempo_bpm": bt.get("tempo_bpm_median_interval"),
                                     "whole_within_5pct": adh["within_5pct"] if adh else None, "whole_beat_cv": bt.get("beat_interval_cv"),
                                     "per_chunk": pc, "tempo_reward_additive": float(np.mean(sc)) if len(sc) == log["K"] else None,
                                     "tempo_reward_partial": float(np.mean(sc)) if sc else None, "n_chunks_measurable": len(sc)}
        if i % 50 == 0:
            print(i, f"{method}/{N}/{d}", rows[f"{method}/{N}/{d}"]["tempo_reward_partial"], flush=True)
            json.dump({"rows": list(rows.values())}, open(a.out, "w"))
    json.dump({"rows": list(rows.values()), "convention": "windows of the final returned clip"}, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(rows), flush=True)


if __name__ == "__main__":
    main()
