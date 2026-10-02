"""M2 / B2 harness: Stable Audio 3 `medium` + caller-side replacement inpainting.

Identical to m1b/sa3_harness.py except for Amendment S item 1:

  * the SAME decoder is stochastic (`noise_regularize` in models/bottleneck.py and
    `mask_noise=0.01` in models/autoencoders.py both draw from the *global* torch
    RNG), so the decode of a fixed latent depends on the RNG state at decode time.
    Here `pretransform.decode` is wrapped at runtime (this process only, no repo
    file touched) so that `torch.manual_seed(decoder_seed)` is called immediately
    before every decode.  `decoder_seed` is fixed per (prompt, base_seed) -- the
    same value for all K chunks of a sequence -- and recorded in the log.
  * the RAW decode is always used; no waveform splice is written.

Seeding the global RNG at decode time cannot disturb the sampler: every
`generate()` call re-seeds the global RNG itself (model.py:242) before drawing
its noise, and the decode happens after the last sampler step.

Note (measured in M1b sec 3.5, not re-argued here): pinning the decoder seed makes
the decode *reproducible*, it does NOT make the decoded prefix bit-exact across
chunks, because the number/shape of the decoder's noise draws grows with the
latent length.  The fixed-prefix criterion is therefore evaluated on the LATENT,
which is the particle state (Amendment S item 1).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SA3_ROOT = os.environ.get("STABLE_AUDIO_ROOT", os.path.join(os.environ.get("LATTICESMC_ROOT", "."), "third_party", "stable-audio-3"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def derive_seed(prompt_idx: int, base_seed: int, chunk_idx: int, particle_idx: int) -> int:
    """Unchanged from M1b so that M2 base samples reproduce M1b's where they overlap."""
    key = f"m1b|{prompt_idx}|{base_seed}|{chunk_idx}|{particle_idx}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31 - 1)


def derive_decoder_seed(prompt_idx: int, base_seed: int) -> int:
    """Fixed per (prompt, seed); the same for every chunk of that sequence."""
    key = f"m2dec|{prompt_idx}|{base_seed}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31 - 1)


class _Ctx:
    """Per-call state shared with the patched samplers (single-threaded use)."""
    prefix_latent: Optional[torch.Tensor] = None   # [1, C, P] clean latent to clamp
    n_prefix: int = 0
    last_latent: Optional[torch.Tensor] = None     # final sampled latent, captured
    steps_run: int = 0
    decoder_seed: Optional[int] = None             # pinned before every decode
    decode_calls: int = 0


CTX = _Ctx()


def _install_patches():
    sys.path.insert(0, SA3_ROOT)
    from stable_audio_3.inference import sampling as S
    if getattr(S, "_m2_patched", False):
        return S

    def sample_flow_pingpong(model, x, sigmas, callback=None, disable_tqdm=True, **extra):
        t = sigmas.to(x.device)
        per_elem = t.dim() == 2
        num_steps = t.shape[-1] - 1
        P, pref = CTX.n_prefix, CTX.prefix_latent
        for i in range(num_steps):
            if per_elem:
                t_curr = t[:, i].to(x.dtype); t_next = t[:, i + 1].to(x.dtype)
                tcb = t_curr.view(-1, 1, 1); tnb = t_next.view(-1, 1, 1)
                t_curr_tensor = t_curr
            else:
                t_curr = t[i].to(x.dtype); t_next = t[i + 1].to(x.dtype)
                tcb = t_curr; tnb = t_next
                t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            denoised = x - tcb * model(x, t_curr_tensor, **extra)
            if P > 0 and pref is not None:
                denoised[..., :P] = pref.to(denoised.dtype)
            x = (1 - tnb) * denoised + tnb * torch.randn_like(x)
        if P > 0 and pref is not None:
            x[..., :P] = pref.to(x.dtype)
        CTX.last_latent = x.detach().clone()
        CTX.steps_run = num_steps
        return x

    def sample_discrete_euler(model, x, sigmas, callback=None, disable_tqdm=True, **extra):
        t = sigmas.to(x.device)
        per_elem = t.dim() == 2
        num_steps = t.shape[-1] - 1
        P, pref = CTX.n_prefix, CTX.prefix_latent
        for i in range(num_steps):
            if per_elem:
                t_curr = t[:, i].to(x.dtype); t_next = t[:, i + 1].to(x.dtype)
                tcb = t_curr.view(-1, 1, 1); tnb = t_next.view(-1, 1, 1)
                t_curr_tensor = t_curr
            else:
                t_curr = t[i].to(x.dtype); t_next = t[i + 1].to(x.dtype)
                tcb = t_curr; tnb = t_next
                t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            v = model(x, t_curr_tensor, **extra)
            denoised = x - tcb * v
            if P > 0 and pref is not None:
                denoised[..., :P] = pref.to(denoised.dtype)
            r = tnb / torch.clamp(tcb, min=1e-8)
            x = r * x + (1 - r) * denoised
        if P > 0 and pref is not None:
            x[..., :P] = pref.to(x.dtype)
        CTX.last_latent = x.detach().clone()
        CTX.steps_run = num_steps
        return x

    S.sample_flow_pingpong = sample_flow_pingpong
    S.sample_discrete_euler = sample_discrete_euler
    S._m2_patched = True
    return S


def _wrap_decoder(pretransform):
    """Pin the global RNG immediately before every decode, and count decodes."""
    if getattr(pretransform, "_m2_decode_wrapped", False):
        return
    orig = pretransform.decode

    def wrapped(latents, *a, **kw):
        if CTX.decoder_seed is not None:
            torch.manual_seed(int(CTX.decoder_seed))
            torch.cuda.manual_seed_all(int(CTX.decoder_seed))
        CTX.decode_calls += 1
        return orig(latents, *a, **kw)

    pretransform.decode = wrapped
    pretransform._m2_decode_wrapped = True


class NFECounter:
    """Rows through DiffusionTransformer._forward (CFG doubled batch == 2 rows)."""

    def __init__(self, dit):
        self.nfe = 0
        self.n_calls = 0
        self._dit = dit
        self._orig = dit._forward

        def wrapped(x, *a, **kw):
            self.nfe += int(x.shape[0])
            self.n_calls += 1
            return self._orig(x, *a, **kw)

        dit._forward = wrapped

    def reset(self):
        self.nfe = 0
        self.n_calls = 0


@dataclass
class ChunkState:
    wav_path: str
    latent: Optional[torch.Tensor]        # [1, C, T] full sampled latent
    audio: Optional[np.ndarray]           # [2, N] float32 decoded (RAW), truncated
    n_chunks: int
    duration_s: float
    per_chunk: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ChunkState":
        return ChunkState(self.wav_path,
                          None if self.latent is None else self.latent.clone(),
                          None if self.audio is None else self.audio.copy(),
                          self.n_chunks, self.duration_s, list(self.per_chunk))


class SA3Harness:
    def __init__(self, model_name: str = "medium", steps: int = 8, cfg_scale: float = 1.0,
                 chunk_sec: float = 10.0, device: str = "cuda",
                 duration_padding_sec: float = 6.0):
        os.chdir(SA3_ROOT)
        os.environ.setdefault("HF_HOME", os.path.join(SA3_ROOT, ".hf-cache"))
        _install_patches()
        from stable_audio_3.model import StableAudioModel
        t0 = time.time()
        self.m = StableAudioModel.from_pretrained(model_name, device=device)
        self.load_s = time.time() - t0
        self.model_name = model_name
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.chunk_sec = chunk_sec
        self.duration_padding_sec = duration_padding_sec
        self.sr = self.m.model.sample_rate
        self.ds = self.m.model.pretransform.downsampling_ratio
        self.objective = self.m.model.diffusion_objective
        self.sampler_type = "pingpong" if self.objective == "rf_denoiser" else "euler"
        self.counter = NFECounter(self.m.model.model.model)
        _wrap_decoder(self.m.model.pretransform)

    def n_prefix_frames(self, prefix_sec: float) -> int:
        return int(prefix_sec * self.sr) // self.ds

    def _gen(self, prompt: str, total_sec: float, seed: int,
             prefix_latent: Optional[torch.Tensor], n_prefix: int, decoder_seed: int):
        CTX.prefix_latent = prefix_latent
        CTX.n_prefix = n_prefix
        CTX.last_latent = None
        CTX.decoder_seed = int(decoder_seed)
        dc0 = CTX.decode_calls
        nfe0, calls0 = self.counter.nfe, self.counter.n_calls
        t0 = time.time()
        audio = self.m.generate(
            prompt=prompt, duration=total_sec, steps=self.steps,
            cfg_scale=self.cfg_scale, batch_size=1, seed=seed,
            duration_padding_sec=self.duration_padding_sec,
            sampler_type=self.sampler_type, disable_tqdm=True,
        )
        torch.cuda.synchronize()
        wall = time.time() - t0
        lat = CTX.last_latent
        CTX.prefix_latent = None
        CTX.n_prefix = 0
        CTX.decoder_seed = None
        return {
            "wall_s": wall,
            "nfe": self.counter.nfe - nfe0,
            "dit_calls": self.counter.n_calls - calls0,
            "decode_calls": CTX.decode_calls - dc0,
            "latent": lat,
            "audio": audio[0].detach().float().cpu().numpy(),
        }

    def start(self, prompt, prompt_idx, base_seed, work_dir, stem, particle_idx=0,
              decoder_seed=None):
        import soundfile as sf
        os.makedirs(work_dir, exist_ok=True)
        seed = derive_seed(prompt_idx, base_seed, 0, particle_idx)
        dseed = derive_decoder_seed(prompt_idx, base_seed) if decoder_seed is None else decoder_seed
        out = self._gen(prompt, self.chunk_sec, seed, None, 0, dseed)
        dst = os.path.join(work_dir, f"{stem}_chunk00.wav")
        sf.write(dst, out["audio"].T, self.sr, subtype="FLOAT")
        rec = {"k": 0, "start_s": 0.0, "end_s": self.chunk_sec, "seed": seed,
               "decoder_seed": int(dseed),
               "wall_s": out["wall_s"], "nfe": out["nfe"], "dit_calls": out["dit_calls"],
               "decode_calls": out["decode_calls"],
               "n_prefix_frames": 0, "latent_shape": list(out["latent"].shape),
               "wav": dst}
        return ChunkState(dst, out["latent"], out["audio"], 1, self.chunk_sec, [rec])

    def extend(self, state: ChunkState, prompt, prompt_idx, base_seed, work_dir, stem,
               particle_idx=0, decoder_seed=None):
        import soundfile as sf
        k = state.n_chunks
        seed = derive_seed(prompt_idx, base_seed, k, particle_idx)
        dseed = derive_decoder_seed(prompt_idx, base_seed) if decoder_seed is None else decoder_seed
        total = state.duration_s + self.chunk_sec
        P = self.n_prefix_frames(state.duration_s)
        pref = state.latent[..., :P].clone()
        out = self._gen(prompt, total, seed, pref, P, dseed)
        dst = os.path.join(work_dir, f"{stem}_chunk{k:02d}.wav")
        sf.write(dst, out["audio"].T, self.sr, subtype="FLOAT")
        rec = {"k": k, "start_s": state.duration_s, "end_s": total, "seed": seed,
               "decoder_seed": int(dseed),
               "wall_s": out["wall_s"], "nfe": out["nfe"], "dit_calls": out["dit_calls"],
               "decode_calls": out["decode_calls"],
               "n_prefix_frames": P, "prefix_sec_clamped": P * self.ds / self.sr,
               "latent_shape": list(out["latent"].shape), "wav": dst}
        return ChunkState(dst, out["latent"], out["audio"], k + 1, total,
                          state.per_chunk + [rec])

    def generate_sequence(self, prompt, prompt_idx, base_seed, work_dir, stem, K=4):
        st = self.start(prompt, prompt_idx, base_seed, work_dir, stem)
        for _ in range(1, K):
            st = self.extend(st, prompt, prompt_idx, base_seed, work_dir, stem)
        return st
