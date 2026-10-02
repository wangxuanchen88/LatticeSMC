"""R1 Phase 3 (music): K = 8 (80 s) on harness B2 with a ROLLING CONTEXT. Chunk k (0-based) is generated inside a window of
CTX_SEC = 40 s: for k < 4 the window is the (k + 1) x 10 s clip as in the paper's harness; for k >= 4 it is the last 40 s,
i.e. the particle's previous three chunks are the fixed prefix latents (clamped at every flow step, as in the paper) and
the fourth 10 s is generated. The full 80 s latent is the concatenation of the eight 10 s chunk latents. Scores: the
prefix score s_k is the prefix CLAP of the decoded rolling window (the (k + 1) x 10 s clip for k < 4, the last 40 s for
k >= 4, renormalized mean of its 10 s window embeddings); the terminal reward R is the CLAP of the full 80 s decode as the
renormalized mean of its eight window embeddings. Seeds: the same derive_seed(prompt, seed, k, slot) streams and the same
pinned decoder seed as the paper's harness (the first four chunks of the N = 1 run equal the K = 4 base sample). Methods:
best-of-N (draw and argmax), chunk pruning (M = max(1, N // 4)) and the boundary schedule with beta = 1 (draw and argmax
logged from the same particles). Counters: DiT rows (NFE), decoder calls, CLAP calls, wall.

  <sa3 venv> python m3/music_lattice_k8.py --runs bon,greedy_chunk,lsp1 --Ns 8,32 --seeds 2000,2001 --out_root results/r1/samples_music_k8
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m3")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import music_lattice as ML  # noqa: E402
from music_lattice import CHUNK_SEC, STEPS, DEC_CTX, METHOD_INDEX, SCTX, derive_decoder_seed, derive_seed, draw_index, ess, normalized_weights, resample_generator, systematic_resample  # noqa: E402

CTX_SEC = 40.0
K8 = 8


class RollingLattice(ML.MusicLattice):
    def sample_chunk_rolling(self, prompt, prompt_id, base_seed, k, N, full):
        """full: [N, C, n_prefix_frames(10 k)] latents of the particles' own sequences so far (None for k = 0), in the
        paper harness's frame accounting (n_prefix_frames(t) = int(t * sr) // ds, so 10 s = 107 or 108 frames).
        k < 4: the paper's sample_chunk with the full sequence as prefix; the new full sequence is the window's first
        n_prefix_frames(10 (k + 1)) frames (exactly the paper's prefix rule). k >= 4: the window is the last 40 s, prefix =
        the last n_prefix_frames(30) frames of full, and the generated 10 s are frames [n30, n40) of the window."""
        n30, n40 = self.n_prefix_frames(3 * CHUNK_SEC), self.n_prefix_frames(4 * CHUNK_SEC)
        if k < 4:
            lat = self.sample_chunk(prompt, prompt_id, base_seed, k, N, full)
            return lat, lat[..., :self.n_prefix_frames((k + 1) * CHUNK_SEC)].clone()
        prefix = full[..., -n30:]
        cond, _ = self.m._build_conditioning_dicts(prompt, None, CTX_SEC, N)
        L = self.latent_len(CTX_SEC)
        gens, noise = [], []
        for p in range(N):
            g = torch.Generator(device=self.device).manual_seed(derive_seed(prompt_id, base_seed, k, p))
            noise.append(torch.randn([1, self.io_channels, L], device=self.device, generator=g))
            gens.append(g)
        noise = torch.cat(noise).type(self.model_dtype)
        ct = self.m.model.conditioner(cond, self.device)
        ct["inpaint_mask"] = [torch.zeros((N, 1, L), device=self.device)]
        ct["inpaint_masked_input"] = [torch.zeros((N, self.io_channels, L), device=self.device)]
        ci = self.m.model.get_conditioning_inputs(ct)
        ci = {kk: v.type(self.model_dtype) if v is not None else v for kk, v in ci.items()}
        SCTX.prefix, SCTX.n_prefix, SCTX.gens, SCTX.hook, SCTX.last = prefix.to(self.model_dtype), n30, gens, None, None
        try:
            lat = self.S.sample_diffusion(model=self.m.model.model, noise=noise, cond_inputs=ci, diffusion_objective="rf_denoiser", steps=STEPS,
                                          cfg_scale=1.0, conditioning=cond, sample_rate=self.sr, pretransform=self.m.model.pretransform,
                                          mask_padding_attention=True, use_effective_length_for_schedule=True, headroom_seconds=6.0,
                                          dist_shift=self.m.model.sampling_dist_shift, sampler_type="pingpong", batch_cfg=True, rescale_cfg=True,
                                          apg_scale=1.0, decode=False, disable_tqdm=True).detach()
        finally:
            SCTX.prefix, SCTX.n_prefix, SCTX.gens, SCTX.hook = None, 0, None, None
        return lat, torch.cat([full, lat[..., n30:n40]], dim=-1)

    def rolling_prefix_score(self, full, k, prompt, prompt_id, base_seed):
        """Prefix CLAP of the decoded rolling prefix after chunk k: the (k + 1) x 10 s clip for k < 4, the last 40 s for k >= 4."""
        N = full.shape[0]
        n_win = min(k + 1, 4)
        win = full if k < 4 else full[..., -self.n_prefix_frames(4 * CHUNK_SEC):]
        s, emb, audio = np.zeros(N), [], []
        for p in range(N):
            a = self.decode(win[p:p + 1], prompt_id, base_seed, n_win * CHUNK_SEC)
            s[p], e = self.prefix_clap(a, prompt, n_win)
            emb.append(e)
            audio.append(a)
        return s, emb, audio

    def terminal_score(self, full, prompt, prompt_id, base_seed):
        """Terminal CLAP of the full decode (858 latent frames = 79.7 s; the eighth CLAP window is zero-padded by 0.3 s) as the
        renormalized mean of eight 10 s window embeddings."""
        N = full.shape[0]
        R, emb, audio = np.zeros(N), [], []
        for p in range(N):
            a = self.decode(full[p:p + 1], prompt_id, base_seed, K8 * CHUNK_SEC)
            R[p], e = self.prefix_clap(a, prompt, K8)
            emb.append(e)
            audio.append(a)
        return R, emb, audio


def run_bon(H, prompt, pid, seed, N, alpha, g_is):
    full = None
    for k in range(K8):
        _, full = H.sample_chunk_rolling(prompt, pid, seed, k, N, full)
    R, emb, audio = H.terminal_score(full, prompt, pid, seed)
    w = normalized_weights(R / alpha)
    idx_is, idx_arg = draw_index(w, g_is), int(np.argmax(R))
    common = {"rewards": R.tolist(), "ess_content": [{"chunk": K8, "ess": ess(R / alpha), "resampled": False, "final_draw": True}], "ess_noise": []}
    return full, R, emb, audio, {"bon_argmax": {**common, "weights": (np.arange(N) == idx_arg).astype(float).tolist(), "chosen": idx_arg, "final_reward": float(R[idx_arg]), "weighted_mean_reward": float(R.mean())},
                                 "bon_is": {**common, "weights": w.tolist(), "chosen": idx_is, "final_reward": float(R[idx_is]), "weighted_mean_reward": float(np.dot(w, R))}}


def run_greedy(H, prompt, pid, seed, N, alpha):
    M = max(1, N // 4)
    full, events, s_hist = None, [], []
    for k in range(K8):
        _, full = H.sample_chunk_rolling(prompt, pid, seed, k, N, full)
        if k < K8 - 1:
            s, _, _ = H.rolling_prefix_score(full, k, prompt, pid, seed)
            s_hist.append(s.tolist())
            order = np.argsort(-s, kind="stable")
            parents = np.repeat(order[:M], N // M)
            events.append({"chunk": k + 1, "ess": float(M), "kept": order[:M].tolist(), "resampled": True})
            full = full[torch.as_tensor(parents, device=full.device)]
    R, emb, audio = H.terminal_score(full, prompt, pid, seed)
    idx = int(np.argmax(R))
    events.append({"chunk": K8, "ess": float(N), "resampled": False, "final_draw": True})
    return full, R, emb, audio, {"rewards": R.tolist(), "weights": (np.arange(N) == idx).astype(float).tolist(), "chosen": idx, "final_reward": float(R[idx]),
                                 "weighted_mean_reward": float(R.mean()), "prefix_scores": s_hist, "M": M, "ess_content": events, "ess_noise": []}


def run_lattice(H, prompt, pid, seed, N, alpha, beta, g):
    full, events, s_hist = None, [], []
    logw, s_prev, cum = np.zeros(N), np.zeros(N), np.zeros(N)
    for k in range(K8):
        _, full = H.sample_chunk_rolling(prompt, pid, seed, k, N, full)
        if k < K8 - 1:
            s, _, _ = H.rolling_prefix_score(full, k, prompt, pid, seed)
        else:
            s, emb, audio = H.terminal_score(full, prompt, pid, seed)
        s_hist.append(s.tolist())
        inc = (beta * (s - s_prev) / alpha) if k < K8 - 1 else ((s - beta * s_prev) / alpha)
        logw, cum, s_prev = logw + inc, cum + inc, s
        if k < K8 - 1:
            e = ess(logw)
            resampled = e < N / 2
            events.append({"chunk": k + 1, "ess": e, "resampled": bool(resampled)})
            if resampled:
                parents = systematic_resample(normalized_weights(logw), g)
                full, s_prev, cum = full[torch.as_tensor(parents, device=full.device)], s_prev[parents], cum[parents]
                logw = np.zeros(N)
    R = s
    w = normalized_weights(logw)
    idx = draw_index(w, g)
    events.append({"chunk": K8, "ess": ess(logw), "resampled": False, "final_draw": True})
    tel = float(np.max(np.abs(cum - R / alpha)))
    return full, R, emb, audio, {"rewards": R.tolist(), "weights": w.tolist(), "chosen": idx, "final_reward": float(R[idx]), "weighted_mean_reward": float(np.dot(w, R)),
                                 "prefix_scores": s_hist, "beta": beta, "ess_content": events, "ess_noise": [], "telescoping_max_abs_err": tel,
                                 "argmax_index": int(np.argmax(R)), "argmax_reward": float(R.max())}


def save(out_dir, H, full, R, emb, audio, log, extra, save_argmax=False):
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    idx = log["chosen"]
    sf.write(os.path.join(out_dir, "seq_chunk07.wav"), audio[idx].T, H.sr, subtype="FLOAT")
    np.save(os.path.join(out_dir, "final_latent.npy"), full[idx].float().cpu().numpy())
    np.save(os.path.join(out_dir, "clap_embed.npy"), np.asarray(emb[idx], dtype=np.float32))
    if save_argmax:  # the argmax particle's clip and embedding, so the search rule needs no replay
        ia = int(np.argmax(R))
        sf.write(os.path.join(out_dir, "seq_chunk07_argmax.wav"), audio[ia].T, H.sr, subtype="FLOAT")
        np.save(os.path.join(out_dir, "clap_embed_argmax.npy"), np.asarray(emb[ia], dtype=np.float32))
    json.dump({**extra, **log}, open(os.path.join(out_dir, "log.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--runs", default="bon,greedy_chunk,lsp1")
    ap.add_argument("--Ns", default="8,32")
    ap.add_argument("--seeds", default="2000,2001")
    ap.add_argument("--prompts", default="all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--alpha_file", default=ML.ALPHA_FILE)
    a = ap.parse_args()
    alpha = float(json.load(open(a.alpha_file))["alpha"])
    prompts = json.load(open(ML.PROMPTS))
    if a.prompts != "all":
        keep = {int(x) for x in a.prompts.split(",")}
        prompts = [p for p in prompts if p["id"] in keep]
    i_s, n_s = (int(x) for x in a.shard.split("/"))
    prompts = [p for j, p in enumerate(prompts) if j % n_s == i_s]
    H = RollingLattice()
    H.reward = "clap"
    print(f"loaded in {H.load_s:.1f}s K={K8} rolling ctx {CTX_SEC}s alpha={alpha} prompts={len(prompts)} shard={a.shard}", flush=True)
    outs_of = {"bon": ["bon_argmax", "bon_is"], "base": ["base"], "greedy_chunk": ["greedy_chunk"], "lsp1": ["lattice_smc_prefix_b1"]}
    t_all, n_new = time.time(), 0
    for prec in prompts:
        for seed in (int(x) for x in a.seeds.split(",")):
            for N in (int(x) for x in a.Ns.split(",")):
                for run in a.runs.split(","):
                    if run == "base" and N != 1:
                        continue
                    dirs = {o: os.path.join(a.out_root, o, str(N), f"p{prec['id']:02d}_s{seed}") for o in outs_of[run]}
                    if all(os.path.exists(os.path.join(d, "log.json")) for d in dirs.values()):
                        continue
                    prompt, pid = prec["text"], prec["id"]
                    H.counter.reset(); DEC_CTX.decode_calls = 0; H.clap_calls = 0
                    t0 = time.time()
                    if run in ("bon", "base"):
                        full, R, emb, audio, logs = run_bon(H, prompt, pid, seed, N, alpha, resample_generator(seed, "bon_is"))
                        outs = {"base": logs["bon_argmax"]} if run == "base" else logs
                    elif run == "greedy_chunk":
                        full, R, emb, audio, log = run_greedy(H, prompt, pid, seed, N, alpha); outs = {"greedy_chunk": log}
                    else:
                        full, R, emb, audio, log = run_lattice(H, prompt, pid, seed, N, alpha, 1.0, resample_generator(seed, "lattice_smc_prefix")); outs = {"lattice_smc_prefix_b1": log}
                    torch.cuda.synchronize()
                    extra = {"prompt_id": pid, "genre": prec["genre"], "prompt": prompt, "bpm": prec["tempo_bpm"], "base_seed": seed, "K": K8, "N": N, "alpha": alpha,
                             "context_sec": CTX_SEC, "nfe": int(H.counter.nfe), "dit_calls": int(H.counter.n_calls), "decode_calls": int(DEC_CTX.decode_calls),
                             "clap_calls": int(H.clap_calls), "wall": round(time.time() - t0, 3), "decoder_seed": derive_decoder_seed(pid, seed),
                             "resample_generator_seed": int(seed) * 131 + METHOD_INDEX[{"bon": "bon_is", "base": "base", "greedy_chunk": "greedy_chunk", "lsp1": "lattice_smc_prefix"}[run]],
                             "gpu": torch.cuda.get_device_name(0), "steps": STEPS, "cfg_scale": 1.0, "sampler": "pingpong", "model": "stable-audio-3 medium",
                             "decode": "raw, batch 1 per particle, pinned decoder seed; full 80 s decode for the terminal reward", "reward": "clap"}
                    for name, log in outs.items():
                        save(dirs[name], H, full, R, emb, audio, {**log, "method": name}, extra, save_argmax=(name == "lattice_smc_prefix_b1"))
                        print(f"{name} N={N} p{pid:02d} s{seed} nfe={extra['nfe']} dec={extra['decode_calls']} clap={extra['clap_calls']} R={log['final_reward']:.4f} wall={extra['wall']:.1f}s", flush=True)
                    n_new += 1
    print(f"done: {n_new} runs in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
