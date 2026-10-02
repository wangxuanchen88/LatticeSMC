"""Phase M3 (Amendment T): the Feynman-Kac lattice on the music testbed (harness B2 = Stable
Audio 3 medium + caller-side replacement inpainting, fixed decoder seed per (prompt, seed)).

Runs inside the Stable Audio 3 venv (python 3.10, torch 2.7.1); no repo file of Stable Audio 3
is modified: the ping-pong sampler is replaced at runtime (this process only) by a version that
(a) draws every noise from a per-particle-slot CUDA generator seeded exactly like the M2 base
grid (slot p of chunk k: derive_seed(prompt, seed, k, p); slot 0 therefore reproduces the M2
base samples bitwise at N = 1), (b) clamps each particle's own prefix latent after every step
(replacement inpainting), and (c) calls a hook after every step with the clamped x0 prediction so
fk_noise can weigh and resample particles on the noise axis.

Particle state = the latent [C, L_k] after chunk k (the prefix for chunk k + 1 is its first P
frames). NFE = transformer forward rows (cfg 1.0: one row per particle per step, 8 steps per
chunk). Decoder calls (one per particle per decode, batch 1 with the pinned decoder seed, so a
decode is a function of (latent, prompt, seed) only) and CLAP calls (clips scored) are counted
separately. Methods (SPEC 11i): base, bon_argmax / bon_is (same particle set), greedy_chunk,
lattice_smc_prefix(beta), fk_noise. Rewards: terminal CLAP R = s_K, prefix CLAP s_k (m1/rewards
protocol: non-overlapping 10 s windows, unit-norm mean, cosine with the prompt text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m2")
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sa3_harness_m2 import SA3_ROOT, NFECounter, _wrap_decoder, CTX as DEC_CTX, derive_decoder_seed, derive_seed  # noqa: E402

PROMPTS = os.environ.get("LATTICESMC_ROOT", ".") + "/m1/prompts.json"
ALPHA_FILE = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/alpha.json"
K_DEFAULT, CHUNK_SEC, STEPS = 4, 10.0, 8
FK_STEPS = (1, 3, 5, 7)  # 0-based step index after which fk_noise evaluates its potential (every second of the 8 steps)
METHOD_INDEX = {"base": 0, "bon_argmax": 1, "bon_is": 2, "fk_noise": 3, "greedy_chunk": 4, "lattice_smc_prefix": 2}
# lattice_smc_prefix shares bon_is's resampling generator so that beta = 0 reproduces bon_is bitwise (SPEC 11i)


# ----------------------------------------------------------------------------- resampling ----

def ess(logw):
    w = np.exp(logw - logw.max())
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def normalized_weights(logw):
    w = np.exp(logw - logw.max())
    return w / w.sum()


def systematic_resample(w, g):
    N = len(w)
    u = (torch.rand(1, generator=g).item() + np.arange(N)) / N
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, u).astype(np.int64)


def draw_index(w, g):
    u = torch.rand(1, generator=g).item()
    c = np.cumsum(w)
    c[-1] = 1.0
    return int(np.searchsorted(c, u))


def resample_generator(seed, method):
    return torch.Generator().manual_seed(int(seed) * 131 + METHOD_INDEX[method])


# --------------------------------------------------------------------------------- sampler ----

class _SamplerCtx:
    prefix = None      # [N, C, P] per-particle clean prefix latent, or None
    n_prefix = 0
    gens = None        # per-slot CUDA generators
    hook = None        # hook(step_index, denoised [N, C, L]) -> parent indices or None
    last = None


SCTX = _SamplerCtx()


def _install_sampler():
    sys.path.insert(0, SA3_ROOT)
    from stable_audio_3.inference import sampling as S

    def sample_flow_pingpong(model, x, sigmas, callback=None, disable_tqdm=True, **extra):
        t = sigmas.to(x.device)
        per_elem = t.dim() == 2
        num_steps = t.shape[-1] - 1
        N = x.shape[0]
        for i in range(num_steps):
            if per_elem:
                t_curr, t_next = t[:, i].to(x.dtype), t[:, i + 1].to(x.dtype)
                tcb, tnb, t_curr_tensor = t_curr.view(-1, 1, 1), t_next.view(-1, 1, 1), t_curr
            else:
                t_curr, t_next = t[i].to(x.dtype), t[i + 1].to(x.dtype)
                tcb, tnb = t_curr, t_next
                t_curr_tensor = t_curr * torch.ones((N,), dtype=x.dtype, device=x.device)
            denoised = x - tcb * model(x, t_curr_tensor, **extra)
            if SCTX.n_prefix > 0:
                denoised[..., :SCTX.n_prefix] = SCTX.prefix.to(denoised.dtype)
            if SCTX.hook is not None:
                parents = SCTX.hook(i, denoised)
                if parents is not None:  # noise-axis resampling: particles move, slots keep their noise streams
                    pt = torch.as_tensor(parents, device=x.device)
                    denoised = denoised[pt]
                    if per_elem:
                        tnb = tnb[pt]
                    if SCTX.n_prefix > 0:
                        SCTX.prefix = SCTX.prefix[pt]
            noise = torch.cat([torch.randn(x[p:p + 1].shape, dtype=x.dtype, device=x.device, generator=SCTX.gens[p]) for p in range(N)])
            x = (1 - tnb) * denoised + tnb * noise
        if SCTX.n_prefix > 0:
            x[..., :SCTX.n_prefix] = SCTX.prefix.to(x.dtype)
        SCTX.last = x.detach().clone()
        return x

    S.sample_flow_pingpong = sample_flow_pingpong
    S._m3_patched = True
    return S


# --------------------------------------------------------------------------------- harness ----

class MusicLattice:
    def __init__(self, device="cuda", model_name="medium"):
        os.chdir(SA3_ROOT)
        os.environ.setdefault("HF_HOME", os.path.join(SA3_ROOT, ".hf-cache"))
        self.S = _install_sampler()
        from stable_audio_3.model import StableAudioModel
        import rewards as R
        t0 = time.time()
        self.m = StableAudioModel.from_pretrained(model_name, device=device)
        self.device = device
        self.sr = self.m.model.sample_rate
        self.ds = self.m.model.pretransform.downsampling_ratio
        self.io_channels = self.m.model.io_channels
        self.model_dtype = next(self.m.model.model.parameters()).dtype
        self.pt_dtype = next(self.m.model.pretransform.parameters()).dtype
        assert self.m.model.diffusion_objective == "rf_denoiser"
        self.counter = NFECounter(self.m.model.model.model)
        _wrap_decoder(self.m.model.pretransform)
        self.clap = R.ClapScorer(device=device)
        self.R = R
        self.clap_calls = 0
        self.reward = "clap"  # Amendment T: prefix / terminal CLAP; Amendment V: "motif" (chunk-to-chunk-1 CLAP audio similarity)
        self.load_s = time.time() - t0

    # -- sizes --
    def n_prefix_frames(self, prefix_sec):
        return int(prefix_sec * self.sr) // self.ds

    def latent_len(self, total_sec):
        cond = [{"prompt": "", "seconds_total": total_sec}]
        return self.m._adapt_sample_size(cond, 5292032, 6.0) // self.ds

    # -- one chunk for N particles --
    def sample_chunk(self, prompt, prompt_id, base_seed, k, N, prefix, hook=None):
        """k: 0-based chunk index; prefix: [N, C, P] latents of the particles' own prefixes (None for k = 0).
        Returns the latents [N, C, L] of the (k + 1) x 10 s window (prefix clamped)."""
        total = (k + 1) * CHUNK_SEC
        cond, _ = self.m._build_conditioning_dicts(prompt, None, total, N)
        L = self.latent_len(total)
        gens = []
        noise = []
        for p in range(N):
            g = torch.Generator(device=self.device).manual_seed(derive_seed(prompt_id, base_seed, k, p))
            noise.append(torch.randn([1, self.io_channels, L], device=self.device, generator=g))
            gens.append(g)
        noise = torch.cat(noise).type(self.model_dtype)
        ct = self.m.model.conditioner(cond, self.device)
        mask = torch.zeros((N, 1, L), device=self.device)
        ct["inpaint_mask"] = [mask]
        ct["inpaint_masked_input"] = [torch.zeros((N, self.io_channels, L), device=self.device)]
        ci = self.m.model.get_conditioning_inputs(ct)
        ci = {kk: v.type(self.model_dtype) if v is not None else v for kk, v in ci.items()}
        SCTX.prefix = None if prefix is None else prefix.to(self.model_dtype)
        SCTX.n_prefix = 0 if prefix is None else prefix.shape[-1]
        SCTX.gens, SCTX.hook, SCTX.last = gens, hook, None
        try:
            lat = self.S.sample_diffusion(
                model=self.m.model.model, noise=noise, cond_inputs=ci, diffusion_objective="rf_denoiser", steps=STEPS,
                cfg_scale=1.0, conditioning=cond, sample_rate=self.sr, pretransform=self.m.model.pretransform,
                mask_padding_attention=True, use_effective_length_for_schedule=True, headroom_seconds=6.0,
                dist_shift=self.m.model.sampling_dist_shift, sampler_type="pingpong", batch_cfg=True, rescale_cfg=True,
                apg_scale=1.0, decode=False, disable_tqdm=True)
        finally:
            SCTX.prefix, SCTX.n_prefix, SCTX.gens, SCTX.hook = None, 0, None, None
        return lat.detach()

    # -- decode + CLAP --
    def decode(self, latent1, prompt_id, base_seed, duration_sec):
        """Batch-1 raw decode with the pinned decoder seed (M2 protocol): float32 [2, T], clamped, truncated."""
        DEC_CTX.decoder_seed = derive_decoder_seed(prompt_id, base_seed)
        try:
            with torch.no_grad():
                a = self.m.model.pretransform.decode(latent1.to(self.pt_dtype))
        finally:
            DEC_CTX.decoder_seed = None
        a = a.to(torch.float32).clamp(-1, 1)[0, :, :int(duration_sec * self.sr)]
        return a.cpu().numpy()

    def prefix_clap(self, audio, text, n_windows):
        """Prefix CLAP of a decoded k x 10 s clip: unit-norm mean of the k window embeddings, cosine with the prompt."""
        import librosa
        mono = audio.mean(axis=0)
        m48 = librosa.resample(mono, orig_sr=self.sr, target_sr=self.R.CLAP_SR)
        E = self.clap.window_embeds(m48, n_windows)
        self.clap_calls += 1
        mv = E.mean(axis=0)
        mv = mv / np.linalg.norm(mv)
        t = self.clap.text_embed([text])[0]
        return float(mv @ t), mv

    def motif_score(self, audio, n_windows):
        """Amendment V: s_k = cosine similarity of the CLAP audio embedding of window k to that of window 1 (k >= 3; 0 below);
        the returned embedding is the unit mean of the window embeddings (same object as the CLAP reward's, for diversity)."""
        import librosa
        mono = audio.mean(axis=0)
        m48 = librosa.resample(mono, orig_sr=self.sr, target_sr=self.R.CLAP_SR)
        E = self.clap.window_embeds(m48, n_windows)
        self.clap_calls += 1
        mv = E.mean(axis=0)
        mv = mv / np.linalg.norm(mv)
        return float(E[n_windows - 1] @ E[0]), mv

    def score_particles(self, latents, prompt, prompt_id, base_seed, k):
        """Decode every particle's latent after chunk k (0-based) and score its prefix score s_{k+1}: prefix CLAP (reward
        "clap") or the chunk-(k+1)-to-chunk-1 similarity (reward "motif"; identically 0 for k + 1 < 3, no decode made)."""
        N = latents.shape[0]
        if self.reward == "motif" and k + 1 < 3:
            return np.zeros(N), [None] * N, [None] * N
        s, emb, audio = np.zeros(N), [], []
        for p in range(N):
            a = self.decode(latents[p:p + 1], prompt_id, base_seed, (k + 1) * CHUNK_SEC)
            s[p], e = (self.prefix_clap(a, prompt, k + 1) if self.reward == "clap" else self.motif_score(a, k + 1))
            emb.append(e)
            audio.append(a)
        return s, emb, audio


# --------------------------------------------------------------------------------- methods ----

def run_bon(H, prompt, prompt_id, base_seed, K, N, alpha, g_is):
    """N independent particles; bon_argmax (argmax of R) and bon_is (draw from softmax(R / alpha))."""
    prefix, lat = None, None
    for k in range(K):
        lat = H.sample_chunk(prompt, prompt_id, base_seed, k, N, prefix)
        prefix = lat[..., :H.n_prefix_frames((k + 1) * CHUNK_SEC)].clone()
    R, emb, audio = H.score_particles(lat, prompt, prompt_id, base_seed, K - 1)
    w = normalized_weights(R / alpha)
    idx_is = draw_index(w, g_is)
    idx_arg = int(np.argmax(R))
    common = {"rewards": R.tolist(), "ess_content": [{"chunk": K, "ess": ess(R / alpha), "resampled": False, "final_draw": True}], "ess_noise": []}
    return lat, R, emb, audio, {"bon_argmax": {**common, "weights": (np.arange(N) == idx_arg).astype(float).tolist(), "chosen": idx_arg,
                                              "final_reward": float(R[idx_arg]), "weighted_mean_reward": float(R.mean())},
                                "bon_is": {**common, "weights": w.tolist(), "chosen": idx_is, "final_reward": float(R[idx_is]),
                                           "weighted_mean_reward": float(np.dot(w, R))}}


def run_greedy(H, prompt, prompt_id, base_seed, K, N, alpha):
    """Top-M by prefix CLAP after each chunk (M = max(1, N // 4)), each kept particle expanded to N / M slots."""
    M = max(1, N // 4)
    if getattr(H, "prune_M", None):  # R1 Phase 2 (2026-09-19): M sweep; default unchanged
        M = int(N // 2) if H.prune_M == "N/2" else int(H.prune_M)
    prefix, lat, events, s_hist = None, None, [], []
    for k in range(K):
        lat = H.sample_chunk(prompt, prompt_id, base_seed, k, N, prefix)
        s, emb, audio = H.score_particles(lat, prompt, prompt_id, base_seed, k)
        s_hist.append(s.tolist())
        if k < K - 1:
            keep = np.argsort(-s, kind="stable")[:M]
            parents = np.repeat(keep, N // M)
            if len(parents) < N:
                parents = np.concatenate([parents, keep[:N - len(parents)]])
            events.append({"chunk": k + 1, "ess": float(M), "resampled": True, "kept": keep.tolist()})
            pt = torch.as_tensor(parents, device=lat.device)
            lat = lat[pt]
        prefix = lat[..., :H.n_prefix_frames((k + 1) * CHUNK_SEC)].clone()
    R = s
    idx = int(np.argmax(R))
    events.append({"chunk": K, "ess": float(N), "resampled": False, "final_draw": True})
    return lat, R, emb, audio, {"rewards": R.tolist(), "weights": (np.arange(N) == idx).astype(float).tolist(), "chosen": idx,
                                "final_reward": float(R[idx]), "weighted_mean_reward": float(R.mean()), "prefix_scores": s_hist, "M": M,
                                "ess_content": events, "ess_noise": []}


def run_lattice_prefix(H, prompt, prompt_id, base_seed, K, N, alpha, beta, g):
    """Intermediate potentials exp(beta (s_k - s_{k-1}) / alpha) for k < K, terminal exp((R - beta s_{K-1}) / alpha);
    systematic resampling when ESS < N / 2; final sample drawn by weight. beta = 0 is bon_is."""
    prefix, lat, events, s_hist = None, None, [], []
    logw, s_prev, cum = np.zeros(N), np.zeros(N), np.zeros(N)  # cum: lineage sum of log potentials (telescoping check)
    for k in range(K):
        lat = H.sample_chunk(prompt, prompt_id, base_seed, k, N, prefix)
        s, emb, audio = H.score_particles(lat, prompt, prompt_id, base_seed, k)
        s_hist.append(s.tolist())
        inc = (beta * (s - s_prev) / alpha) if k < K - 1 else ((s - beta * s_prev) / alpha)
        logw, cum, s_prev = logw + inc, cum + inc, s
        if k < K - 1:
            e = ess(logw)
            resampled = e < N / 2
            events.append({"chunk": k + 1, "ess": e, "resampled": bool(resampled)})
            if resampled:
                parents = systematic_resample(normalized_weights(logw), g)
                pt = torch.as_tensor(parents, device=lat.device)
                lat, s_prev, cum = lat[pt], s_prev[parents], cum[parents]
                logw = np.zeros(N)
        prefix = lat[..., :H.n_prefix_frames((k + 1) * CHUNK_SEC)].clone()
    R = s
    w = normalized_weights(logw)
    idx = draw_index(w, g)
    events.append({"chunk": K, "ess": ess(logw), "resampled": False, "final_draw": True})
    tel = float(np.max(np.abs(cum - R / alpha)))  # lineage sum of all log potentials must equal R / alpha
    return lat, R, emb, audio, {"rewards": R.tolist(), "weights": w.tolist(), "chosen": idx, "final_reward": float(R[idx]),
                                "weighted_mean_reward": float(np.dot(w, R)), "prefix_scores": s_hist, "beta": beta,
                                "ess_content": events, "ess_noise": [], "telescoping_max_abs_err": tel}


def run_fk_noise(H, prompt, prompt_id, base_seed, K, N, alpha, g):
    """Noise-axis FK steering: after DDIM-analogue steps 2, 4, 6, 8 of every chunk the potential
    exp((s_hat - s_prev) / alpha), s_hat = prefix CLAP of the decoded clamped x0 prediction (prefix + chunk);
    systematic resampling when ESS < N / 2, weights carried across chunk boundaries (no boundary resampling)."""
    state = {"logw": np.zeros(N), "s_prev": np.zeros(N), "cum": np.zeros(N), "events": [], "s_hat": None, "emb": None, "audio": None, "hats": []}
    prefix, lat = None, None
    for k in range(K):
        def hook(i, denoised, k=k):
            if i not in FK_STEPS:
                return None
            s_hat, emb, audio = H.score_particles(denoised, prompt, prompt_id, base_seed, k)
            inc = (s_hat - state["s_prev"]) / alpha
            state["logw"] = state["logw"] + inc
            state["cum"] = state["cum"] + inc
            state["s_prev"] = s_hat
            state["s_hat"], state["emb"], state["audio"] = s_hat, emb, audio
            state["hats"].append({"chunk": k + 1, "step": i + 1, "s_hat": s_hat.tolist()})
            e = ess(state["logw"])
            resampled = e < N / 2
            state["events"].append({"chunk": k + 1, "step": i + 1, "ess": e, "resampled": bool(resampled)})
            if not resampled:
                return None
            parents = systematic_resample(normalized_weights(state["logw"]), g)
            state["logw"] = np.zeros(N)
            state["s_prev"], state["cum"] = state["s_prev"][parents], state["cum"][parents]
            state["s_hat"] = state["s_hat"][parents]
            state["emb"] = [state["emb"][j] for j in parents] if state["emb"] is not None else None
            state["audio"] = [state["audio"][j] for j in parents] if state["audio"] is not None else None
            return parents
        lat = H.sample_chunk(prompt, prompt_id, base_seed, k, N, prefix, hook=hook)
        prefix = lat[..., :H.n_prefix_frames((k + 1) * CHUNK_SEC)].clone()
    # the last hook's x0 prediction is the final latent (t_next = 0), so s_hat at step 8 of chunk K is R
    R, emb, audio = state["s_hat"], state["emb"], state["audio"]
    w = normalized_weights(state["logw"])
    idx = draw_index(w, g)
    state["events"].append({"chunk": K, "step": STEPS, "ess": ess(state["logw"]), "resampled": False, "final_draw": True})
    tel = float(np.max(np.abs(state["cum"] - R / alpha)))
    return lat, R, emb, audio, {"rewards": R.tolist(), "weights": w.tolist(), "chosen": idx, "final_reward": float(R[idx]),
                                "weighted_mean_reward": float(np.dot(w, R)), "ess_content": [], "ess_noise": state["events"],
                                "x0_scores": state["hats"], "telescoping_max_abs_err": tel}


# ------------------------------------------------------------------------------------ driver ----

def save_sequence(out_dir, H, lat, R, emb, audio, log, extra):
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    idx = log["chosen"]
    sf.write(os.path.join(out_dir, "seq_chunk03.wav"), audio[idx].T, H.sr, subtype="FLOAT")
    np.save(os.path.join(out_dir, "final_latent.npy"), lat[idx].float().cpu().numpy())
    np.save(os.path.join(out_dir, "clap_embed.npy"), np.asarray(emb[idx], dtype=np.float32))
    json.dump({**extra, **log}, open(os.path.join(out_dir, "log.json"), "w"), indent=1)


def generate(H, method, prompt_rec, base_seed, N, K, alpha, beta=None):
    """One (method, N, prompt, seed) sequence. Returns dict method_name -> (lat, R, emb, audio, log)."""
    prompt, pid = prompt_rec["text"], prompt_rec["id"]
    H.counter.reset()
    DEC_CTX.decode_calls = 0
    H.clap_calls = 0
    t0 = time.time()
    if method in ("bon", "base"):
        g = resample_generator(base_seed, "bon_is")
        lat, R, emb, audio, logs = run_bon(H, prompt, pid, base_seed, K, N, alpha, g)
        outs = {"base": logs["bon_argmax"]} if method == "base" else logs
    elif method == "greedy_chunk":
        lat, R, emb, audio, log = run_greedy(H, prompt, pid, base_seed, K, N, alpha)
        outs = {"greedy_chunk": log}
    elif method == "lattice_smc_prefix":
        g = resample_generator(base_seed, "lattice_smc_prefix")
        lat, R, emb, audio, log = run_lattice_prefix(H, prompt, pid, base_seed, K, N, alpha, beta, g)
        outs = {f"lattice_smc_prefix_b{beta:g}": log}
    elif method == "fk_noise":
        g = resample_generator(base_seed, "fk_noise")
        lat, R, emb, audio, log = run_fk_noise(H, prompt, pid, base_seed, K, N, alpha, g)
        outs = {"fk_noise": log}
    else:
        raise ValueError(method)
    torch.cuda.synchronize()
    extra = {"prompt_id": pid, "genre": prompt_rec["genre"], "prompt": prompt, "bpm": prompt_rec["tempo_bpm"], "base_seed": base_seed,
             "K": K, "N": N, "alpha": alpha, "nfe": int(H.counter.nfe), "dit_calls": int(H.counter.n_calls),
             "decode_calls": int(DEC_CTX.decode_calls), "clap_calls": int(H.clap_calls), "wall": round(time.time() - t0, 3),
             "decoder_seed": derive_decoder_seed(pid, base_seed),
             "slot_seeds": [[derive_seed(pid, base_seed, k, p) for p in range(N)] for k in range(K)],
             "resample_generator_seed": (int(base_seed) * 131 + METHOD_INDEX[{"bon": "bon_is", "base": "base"}.get(method, method)]),
             "gpu": torch.cuda.get_device_name(0), "steps": STEPS, "cfg_scale": 1.0, "sampler": "pingpong",
             "model": "stable-audio-3 medium", "decode": "raw, batch 1 per particle, pinned decoder seed", "reward": H.reward}
    return {name: (lat, R, emb, audio, log, extra) for name, log in outs.items()}


METHOD_RUNS = {"base": ("base", None), "bon": ("bon", None), "greedy_chunk": ("greedy_chunk", None), "fk_noise": ("fk_noise", None),
               "lsp1": ("lattice_smc_prefix", 1.0), "lsp05": ("lattice_smc_prefix", 0.5), "lsp025": ("lattice_smc_prefix", 0.25),
               "lsp0": ("lattice_smc_prefix", 0.0)}
RUN_OUTPUTS = {"base": ["base"], "bon": ["bon_argmax", "bon_is"], "greedy_chunk": ["greedy_chunk"], "fk_noise": ["fk_noise"],
               "lsp1": ["lattice_smc_prefix_b1"], "lsp05": ["lattice_smc_prefix_b0.5"], "lsp025": ["lattice_smc_prefix_b0.25"],
               "lsp0": ["lattice_smc_prefix_b0"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3/samples")
    ap.add_argument("--runs", default="bon,greedy_chunk,lsp1,lsp05,lsp025,fk_noise")
    ap.add_argument("--Ns", default="1,2,4,8,16,32")
    ap.add_argument("--seeds", default="2000,2001,2002,2003")
    ap.add_argument("--prompts", default="all", help="'all' or comma-separated prompt ids")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--alpha_file", default=ALPHA_FILE)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--reward", default="clap", choices=["clap", "motif"])
    ap.add_argument("--prune_M", default=None, help="R1 Phase 2: chunk-pruning M (int or N/2); default max(1, N // 4)")
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else float(json.load(open(args.alpha_file))["alpha"])
    prompts = json.load(open(PROMPTS))
    if args.prompts != "all":
        keep = {int(x) for x in args.prompts.split(",")}
        prompts = [p for p in prompts if p["id"] in keep]
    i_shard, n_shard = (int(x) for x in args.shard.split("/"))
    prompts = [p for j, p in enumerate(prompts) if j % n_shard == i_shard]
    Ns = [int(x) for x in args.Ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    runs = args.runs.split(",")
    H = MusicLattice()
    H.reward = args.reward
    H.prune_M = args.prune_M  # R1 Phase 2 (fixed 2026-09-20: the flag was parsed but never applied on 2026-09-19)
    print(f"loaded in {H.load_s:.1f}s reward={H.reward} alpha={alpha} prompts={len(prompts)} shard={args.shard}", flush=True)
    t_all, n_new = time.time(), 0
    for prec in prompts:
        for seed in seeds:
            for N in Ns:
                for run in runs:
                    method, beta = METHOD_RUNS[run]
                    if method == "base" and N != 1:
                        continue
                    outs = RUN_OUTPUTS[run]
                    dirs = {o: os.path.join(args.out_root, o, str(N), f"p{prec['id']:02d}_s{seed}") for o in outs}
                    if all(os.path.exists(os.path.join(d, "log.json")) for d in dirs.values()):
                        continue
                    res = generate(H, method, prec, seed, N, args.K, alpha, beta)
                    for name, (lat, R, emb, audio, log, extra) in res.items():
                        save_sequence(dirs[name], H, lat, R, emb, audio, {**log, "method": name}, extra)
                        print(f"{name} N={N} p{prec['id']:02d} s{seed} nfe={extra['nfe']} dec={extra['decode_calls']} clap={extra['clap_calls']} "
                              f"R={log['final_reward']:.4f} wall={extra['wall']:.1f}s", flush=True)
                    n_new += 1
    print(f"done: {n_new} runs in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
