"""M2 tempo: beat_this (explicit 22.05 kHz resample) on the whole 40 s clip and on
each 10 s chunk.

Per-chunk convention (stated, not implied): chunk j is measured on the window
[j*10, (j+1)*10) s of `seq_chunk{j:02d}.wav`, i.e. of the decode produced at the
moment that chunk was generated.  This is the causal quantity an additive per-chunk
reward would see.  The same window taken from the final 40 s clip is also reported
(`*_final`) as a cross-check.
"""
import os
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1")
import rewards as R

REAL = "<earlier-harness>/edge/data/train/wavs/gBR_sFM_cAll_d06_mBR1_ch15.wav"

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--chunk_sec", type=float, default=10.0)
ap.add_argument("--per_chunk", action="store_true")
ap.add_argument("--also_final_windows", action="store_true")
a = ap.parse_args()

import soundfile as sf
rows = []
for d in sorted(os.listdir(a.samples)):
    sd = os.path.join(a.samples, d)
    lp = os.path.join(sd, "log.json")
    if not os.path.isdir(sd) or not os.path.exists(lp):
        continue
    log = json.load(open(lp))
    bpm = log["bpm"]
    fin = os.path.join(sd, f"seq_chunk{a.K-1:02d}.wav")
    y, sr = sf.read(fin, dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    bt = R.beatthis_tempo(mono, sr, dbn=False, resample_to=22050)
    adh = R.tempo_adherence(bt.get("tempo_bpm_median_interval"), bpm)
    row = {"dir": d, "prompt_id": log["prompt_id"], "base_seed": log["base_seed"],
           "requested_bpm": bpm,
           **{f"bt_{k}": v for k, v in bt.items()},
           **({f"adh_{k}": v for k, v in adh.items()} if adh else {})}
    if a.per_chunk:
        pc = []
        for j in range(a.K):
            w = os.path.join(sd, f"seq_chunk{j:02d}.wav")
            yy, ss = sf.read(w, dtype="float32", always_2d=True)
            seg = yy[int(j * a.chunk_sec * ss): int((j + 1) * a.chunk_sec * ss)].mean(axis=1)
            b = R.beatthis_tempo(seg, ss, dbn=False, resample_to=22050)
            ad = R.tempo_adherence(b.get("tempo_bpm_median_interval"), bpm)
            e = {"chunk": j, "n_beats": b["n_beats"],
                 "tempo_bpm": b.get("tempo_bpm_median_interval"),
                 "beat_interval_cv": b.get("beat_interval_cv")}
            if ad:
                e.update({"rel_err": ad["rel_err"], "within_5pct": ad["within_5pct"],
                          "rel_err_octave_corrected": ad["rel_err_octave_corrected"],
                          "within_5pct_octave_corrected": ad["within_5pct_octave_corrected"],
                          "score": 1.0 if ad["within_5pct"] else 0.0})
            else:
                e.update({"rel_err": None, "within_5pct": None, "score": None})
            if a.also_final_windows:
                segf = mono[int(j * a.chunk_sec * sr): int((j + 1) * a.chunk_sec * sr)]
                bf = R.beatthis_tempo(segf, sr, dbn=False, resample_to=22050)
                adf = R.tempo_adherence(bf.get("tempo_bpm_median_interval"), bpm)
                e["tempo_bpm_final"] = bf.get("tempo_bpm_median_interval")
                e["within_5pct_final"] = adf["within_5pct"] if adf else None
            pc.append(e)
        row["per_chunk"] = pc
        sc = [x["score"] for x in pc if x["score"] is not None]
        row["tempo_reward_additive"] = float(np.mean(sc)) if len(sc) == a.K else None
        row["tempo_reward_additive_partial"] = float(np.mean(sc)) if sc else None
        row["n_chunks_measurable"] = len(sc)
    rows.append(row)
    print("tempo", d, row.get("bt_tempo_bpm_median_interval"),
          row.get("tempo_reward_additive"), flush=True)

yr, srr = sf.read(REAL, dtype="float32", always_2d=True)
mr = yr[: int(a.K * a.chunk_sec * srr)].mean(axis=1)
real = R.beatthis_tempo(mr, srr, dbn=False, resample_to=22050)
json.dump({"rows": rows, "real_music_control": real}, open(a.out, "w"), indent=1)
print("wrote", a.out)
