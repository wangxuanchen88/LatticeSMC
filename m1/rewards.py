"""M1 rewards: CLAP (terminal + additive), tempo adherence (beat_this), seam flux.

Dependency split (three interpreters, none of them modified):
  * CLAP needs `laion_clap` -> Stable-Audio-3 venv
    (<stable-audio-3>/.venv/bin/python, python 3.10, torch 2.7.1)
  * seam flux needs librosa  -> available in all three venvs
  * beat_this needs the M0 analysis venv
    (<repo>/m0/.venv/bin/python, beat-this 1.1.0)
Imports are therefore lazy and per-function.

CLAP checkpoint — WHY NOT the HF `laion/larger_clap_music` port:
that checkpoint (revision a0b4534a14f58e20944452dff00a22a06ce629d1, both from the local HF
cache and from a fresh download of the same revision) is *not trained*: every LayerNorm
weight is exactly 1.0 and every LayerNorm bias exactly 0.0, `logit_scale_a` is 0.0274
(a trained CLAP has ~2.7-4.6), and every weight matrix with the same fan-in has the same
std.  Functionally it gives cos ~= -0.003 for every (audio, text) pair with no ability to
tell dance music from solo piano.  The same failure was independently recorded on this
machine in stable-audio-3-main/flowmuse/eval/clap_wrapper.py.
We therefore use the original LAION checkpoint that `larger_clap_music` is the port of:
  laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
  ckpt = lukewys/laion_clap :: music_audioset_epoch_15_esc_90.14.pt
  (local snapshot b3708341862f581175dba5c356a4ebf74a9b6651, 2352471003 bytes)

How a 40 s clip is embedded — this matters and is fixed here:
laion_clap's audio front end has max_len = 480000 samples (10 s at 48 kHz).  Fed anything
longer with `data_truncating="rand_trunc"` (the non-fusion default) it takes a *random*
480000-sample crop (laion_clap/training/data.py::get_audio_features), which is both lossy
and non-deterministic.  So we never hand it more than 10 s: the 40 s clip is cut into four
non-overlapping 10 s windows (exactly the four chunk spans), each window is exactly 480000
samples so no crop and no pad happens, each is embedded and L2-normalised, and the clip
embedding is the L2-normalised mean of those four unit vectors.
  terminal_clap  = cos( normalize(mean_k e_k), t )
  additive_clap  = mean_k cos(e_k, t)
These are algebraically linked: terminal = additive / ||mean_k e_k||, with e_k unit.  The
link is reported in the analysis, not hidden.
  center_clap    = cos(e(15-25 s), t)   -- an auxiliary terminal statistic that uses a
                   single window instead of all four, so it is not a mean of the chunk terms.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

CLAP_CKPT = ("<stable-audio-3>/.hf-cache/hub/"
             "models--lukewys--laion_clap/snapshots/"
             "b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt")
CLAP_AMODEL = "HTSAT-base"
CLAP_REPO = "lukewys/laion_clap :: music_audioset_epoch_15_esc_90.14.pt"
CLAP_REVISION = "b3708341862f581175dba5c356a4ebf74a9b6651"
CLAP_SR = 48000
CLAP_WINDOW_SAMPLES = 480000  # 10 s at 48 kHz == laion_clap max_len


# ------------------------------------------------------------------ CLAP ----


class ClapScorer:
    def __init__(self, device: str = "cuda", ckpt: str = CLAP_CKPT):
        import torch
        import laion_clap

        self.torch = torch
        self.device = torch.device(device)
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel=CLAP_AMODEL)
        self.model.load_ckpt(ckpt=ckpt)
        self.model = self.model.to(self.device).eval()
        self._text_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    # -- text --
    def text_embed(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            with self.torch.no_grad():
                e = self.model.get_text_embedding(missing, use_tensor=False)
            e = self._unit(np.asarray(e, dtype=np.float64))
            for t, v in zip(missing, e):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts])

    # -- audio --
    def _windows(self, mono48k: np.ndarray, n_windows: int) -> np.ndarray:
        wins = []
        for i in range(n_windows):
            s = i * CLAP_WINDOW_SAMPLES
            w = mono48k[s: s + CLAP_WINDOW_SAMPLES]
            if len(w) < CLAP_WINDOW_SAMPLES:  # never expected for a 40 s clip
                w = np.pad(w, (0, CLAP_WINDOW_SAMPLES - len(w)))
            wins.append(np.asarray(w, dtype=np.float32))
        return np.stack(wins)

    def window_embeds(self, mono48k: np.ndarray, n_windows: int = 4) -> np.ndarray:
        """Unit-norm CLAP embedding of each non-overlapping 10 s window."""
        X = self._windows(mono48k, n_windows)
        with self.torch.no_grad():
            e = self.model.get_audio_embedding_from_data(x=X, use_tensor=False)
        return self._unit(np.asarray(e, dtype=np.float64))

    def embed_window(self, mono48k: np.ndarray, start_sample: int) -> np.ndarray:
        w = mono48k[start_sample: start_sample + CLAP_WINDOW_SAMPLES]
        if len(w) < CLAP_WINDOW_SAMPLES:
            w = np.pad(w, (0, CLAP_WINDOW_SAMPLES - len(w)))
        with self.torch.no_grad():
            e = self.model.get_audio_embedding_from_data(
                x=np.asarray(w, dtype=np.float32)[None], use_tensor=False)
        return self._unit(np.asarray(e, dtype=np.float64))[0]

    def score_clip(self, mono48k: np.ndarray, text: str, n_windows: int = 4) -> Dict:
        t = self.text_embed([text])[0]
        e = self.window_embeds(mono48k, n_windows)                 # [K, D], unit rows
        per_chunk = (e @ t).tolist()                               # cos per 10 s chunk
        m = e.mean(axis=0)
        m_norm = float(np.linalg.norm(m))
        terminal = float((m / m_norm) @ t)
        center_start = (len(mono48k) - CLAP_WINDOW_SAMPLES) // 2
        center = float(self.embed_window(mono48k, center_start) @ t)
        return {
            "clap_terminal": terminal,
            "clap_additive_mean_of_chunks": float(np.mean(per_chunk)),
            "clap_per_chunk": [float(x) for x in per_chunk],
            "clap_mean_window_embed_norm": m_norm,
            "clap_center_window": center,
            "clap_checkpoint": CLAP_REPO,
            "clap_revision": CLAP_REVISION,
        }


def load_mono48k(path: str) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != CLAP_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=CLAP_SR)
    return y


# ------------------------------------------------------------- seam flux ----

HOP_S = 0.010  # identical to M0's analyze.py


def spectral_flux(y: np.ndarray, sr: int, n_fft: int = 2048):
    import librosa
    hop = int(round(HOP_S * sr))
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    L = librosa.amplitude_to_db(S, ref=1.0, amin=1e-8, top_db=None)
    flux = np.sqrt(((L[:, 1:] - L[:, :-1]) ** 2).sum(axis=0))
    times = (np.arange(len(flux)) + 0.5) * hop / sr
    return flux, times, hop


def seam_stats(y: np.ndarray, sr: int, seam_times: List[float], guard_s: float = 0.10) -> Dict:
    """M0's statistic, unchanged: seam-frame flux / median within-chunk flux."""
    flux, times, hop = spectral_flux(y, sr)
    guard = np.zeros(len(flux), dtype=bool)
    for t in seam_times:
        guard |= np.abs(times - t) <= guard_s
    within = flux[~guard]
    med = float(np.median(within))
    out = []
    for t in seam_times:
        i = int(np.argmin(np.abs(times - t)))
        local = flux[max(0, i - 1): i + 2]
        out.append({
            "seam_time_s": float(t),
            "seam_flux": float(flux[i]),
            "ratio_to_median": float(flux[i] / med) if med > 0 else None,
            "ratio_local_max_to_median": float(local.max() / med) if med > 0 else None,
        })
    return {
        "median_within_chunk_flux": med,
        "p95_within_chunk_flux": float(np.percentile(within, 95)),
        "seams": out,
        "mean_ratio": float(np.mean([s["ratio_to_median"] for s in out])),
        "mean_ratio_local_max": float(np.mean([s["ratio_local_max_to_median"] for s in out])),
    }


# ------------------------------------------------------------------ tempo ----

_BT = {}


def beatthis_tempo(y: np.ndarray, sr: int, dbn: bool = False, resample_to: Optional[int] = 22050) -> Dict:
    """Tempo from beat_this.

    beat_this's own `Audio2Frames.signal2spect` already resamples with soxr when
    sr != 22050 (beat_this/inference.py) and its LogMelSpect is built for 22050 Hz.
    `resample_to` lets us do the resampling ourselves (librosa) and hand beat_this an
    already-22050 Hz signal, so the two paths can be compared explicitly.
    """
    from beat_this.inference import Audio2Beats
    if sr != resample_to and resample_to is not None:
        import librosa
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=resample_to)
        sr = resample_to
    key = ("dbn" if dbn else "min")
    if key not in _BT:
        _BT[key] = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=dbn)
    beats, downbeats = _BT[key](np.asarray(y, dtype=np.float64), sr)
    beats = np.asarray(beats)
    downbeats = np.asarray(downbeats)
    ib = np.diff(beats)
    out = {
        "sr_fed": int(sr), "dbn": dbn,
        "n_beats": int(len(beats)), "n_downbeats": int(len(downbeats)),
        "median_beat_interval_s": float(np.median(ib)) if len(ib) > 1 else None,
        "tempo_bpm_median_interval": float(60.0 / np.median(ib)) if len(ib) > 1 else None,
        "tempo_bpm_mean_interval": float(60.0 / ib.mean()) if len(ib) > 1 else None,
        "beat_interval_cv": float(ib.std() / ib.mean()) if len(ib) > 2 else None,
    }
    if len(downbeats) > 1:
        idb = np.diff(downbeats)
        out["median_downbeat_interval_s"] = float(np.median(idb))
    return out


def tempo_adherence(tempo_bpm: Optional[float], requested_bpm: float) -> Optional[Dict]:
    """Octave-aware relative error against the requested tempo."""
    if tempo_bpm is None or tempo_bpm <= 0:
        return None
    best = None
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        err = abs(tempo_bpm * mult - requested_bpm) / requested_bpm
        if best is None or err < best[1]:
            best = (mult, err)
    return {
        "rel_err": abs(tempo_bpm - requested_bpm) / requested_bpm,
        "octave_multiplier": best[0],
        "rel_err_octave_corrected": best[1],
        "within_5pct": bool(abs(tempo_bpm - requested_bpm) / requested_bpm <= 0.05),
        "within_5pct_octave_corrected": bool(best[1] <= 0.05),
    }
