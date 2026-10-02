"""M2 CLAP scoring + seam flux + spectral statistics, on the RAW decode.

For each sequence directory:
  * prefix CLAP after chunk k (k = 1..K): CLAP of `seq_chunk{k-1:02d}.wav`, which is
    the decode (with the sequence's pinned decoder seed) of the latent after chunk k-1
    and is exactly k x 10 s long.  k = K is the terminal CLAP of the full 40 s clip.
    No extra decoder call is made: these files were written by the generator.
  * cosine of the terminal clip against every prompt text in the retrieval set (R@1).
  * seam spectral-flux ratio on the raw 40 s decode, the half-chunk-offset control,
    and >8 kHz energy / centroid / flatness / RMS / peak.
Embedding protocol is m1/rewards.py verbatim (non-overlapping 10 s windows, unit-norm
mean), so numbers are comparable to M1 / M1b.
"""
import os
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1")
import rewards as R

REAL = "<earlier-harness>/edge/data/train/wavs/gBR_sFM_cAll_d06_mBR1_ch15.wav"


def spec_stats(mono, sr, cutoff=8000.0):
    import librosa
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tot = S.sum()
    return {"frac_energy_above_8k": float(S[f > cutoff].sum() / tot) if tot > 0 else None,
            "spectral_centroid_hz": float(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr).mean()),
            "spectral_flatness": float(librosa.feature.spectral_flatness(S=np.sqrt(S)).mean()),
            "rms": float(np.sqrt((mono ** 2).mean())), "peak": float(np.abs(mono).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=os.environ.get("LATTICESMC_ROOT", ".") + "/m1/prompts.json")
    ap.add_argument("--retrieval_ids", type=int, nargs="*", default=None,
                    help="prompt ids forming the retrieval text set; default = ids present in --samples")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--chunk_sec", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_seam", action="store_true")
    a = ap.parse_args()

    allp = {p["id"]: p for p in json.load(open(a.prompts))}
    dirs = [d for d in sorted(os.listdir(a.samples))
            if os.path.isdir(os.path.join(a.samples, d))
            and os.path.exists(os.path.join(a.samples, d, "log.json"))]
    logs = {d: json.load(open(os.path.join(a.samples, d, "log.json"))) for d in dirs}
    rids = a.retrieval_ids if a.retrieval_ids else sorted({logs[d]["prompt_id"] for d in dirs})
    texts = [allp[i]["text"] for i in rids]

    clap = R.ClapScorer(device=a.device)
    T = clap.text_embed(texts)                     # [P, D] unit

    import soundfile as sf
    rows = []
    for d in dirs:
        sd = os.path.join(a.samples, d)
        log = logs[d]
        pid = log["prompt_id"]
        ti = rids.index(pid) if pid in rids else None
        t_own = clap.text_embed([allp[pid]["text"]])[0]

        # prefix CLAP after each chunk k = 1..K
        prefix_clap, prefix_add = [], []
        for k in range(1, a.K + 1):
            wav = os.path.join(sd, f"seq_chunk{k-1:02d}.wav")
            if not os.path.exists(wav):
                raise FileNotFoundError(wav)
            m48 = R.load_mono48k(wav)
            E = clap.window_embeds(m48, k)          # k unit rows
            mv = E.mean(axis=0)
            prefix_clap.append(float((mv / np.linalg.norm(mv)) @ t_own))
            prefix_add.append(float(np.mean(E @ t_own)))
            if k == a.K:
                Efin = E
        term = float(prefix_clap[-1])
        mvf = Efin.mean(axis=0)
        tvec = mvf / np.linalg.norm(mvf)
        cos_all = (tvec @ T.T).tolist()

        row = {"dir": d, "prompt_id": pid, "genre": log.get("genre", allp[pid]["genre"]),
               "base_seed": log["base_seed"], "requested_bpm": log["bpm"],
               "decoder_seed": log.get("decoder_seed"),
               "clap_terminal": term,
               "clap_prefix_k": prefix_clap,               # k = 1..K
               "clap_prefix_additive_k": prefix_add,
               "clap_per_chunk_window": [float(x) for x in (Efin @ t_own)],
               "clap_additive": float(np.mean(Efin @ t_own)),
               "cos_vs_retrieval_texts": cos_all, "matched_index": ti,
               "seq_wall_s": log.get("wall_s"),
               "per_chunk_wall_s": [c["wall_s"] for c in log["per_chunk"]],
               "per_chunk_nfe": [c["nfe"] for c in log["per_chunk"]],
               "per_chunk_decode_calls": [c.get("decode_calls") for c in log["per_chunk"]],
               "nfe_total": log.get("nfe_total"),
               "decode_calls_total": log.get("decode_calls_total")}

        if not a.no_seam:
            wav = os.path.join(sd, f"seq_chunk{a.K-1:02d}.wav")
            y, sr = sf.read(wav, dtype="float32", always_2d=True)
            mono = y.mean(axis=1)
            seam = R.seam_stats(mono, sr, [a.chunk_sec * i for i in range(1, a.K)])
            ctrl = R.seam_stats(mono, sr, [a.chunk_sec * i + a.chunk_sec / 2 for i in range(1, a.K)])
            row.update({
                "seam_mean_ratio": seam["mean_ratio"],
                "seam_mean_ratio_local_max": seam["mean_ratio_local_max"],
                "seam_per": [s["ratio_to_median"] for s in seam["seams"]],
                "control_mean_ratio": ctrl["mean_ratio"],
                "median_within_chunk_flux": seam["median_within_chunk_flux"],
                "p95_within_chunk_flux": seam["p95_within_chunk_flux"],
                **spec_stats(mono, sr)})
        rows.append(row)
        print("scored", d, round(term, 4), flush=True)

    out = {"rows": rows, "retrieval_prompt_ids": rids, "texts": texts,
           "clap_ckpt": R.CLAP_REPO, "clap_revision": R.CLAP_REVISION, "K": a.K}
    if not a.no_seam:
        yr, srr = sf.read(REAL, dtype="float32", always_2d=True)
        mr = yr[: int(a.K * a.chunk_sec * srr)].mean(axis=1)
        out["real_music_control"] = {
            "path": REAL,
            "seam_like_ratio_at_10_20_30s": R.seam_stats(
                mr, srr, [a.chunk_sec * i for i in range(1, a.K)])["mean_ratio"],
            **spec_stats(mr, srr)}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
