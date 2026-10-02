"""Amendment U, music: deterministic regeneration of the argmax particle's clip for lattice_smc_prefix(beta = 1) and
fk_noise at N in {8, 32}, by REPLAY: the stored log carries every prefix CLAP score (lattice: `prefix_scores` per
chunk, pre-resampling) / every x0 score (fk_noise: `x0_scores` per event, pre-resampling), so the particle system is
re-run with the DiT only, the logged scores injected in place of decode + CLAP, the same weights, ESS tests and
systematic-resampling draws (same generator seed). The replay is verified against the stored run: the final latent of
the logged chosen particle must equal `final_latent.npy` bitwise (max abs diff reported) and the replayed final weights
and draw must match the log. Rewards are read from the log (the particle scores at the last event / chunk); only the
argmax particle is decoded (pinned decoder seed, batch 1, the M2 / M3 protocol) and its
terminal CLAP recomputed from that decode as the root of trust (must equal the logged reward of that particle up to
the CLAP recompute, reported).

Runs in the Stable Audio 3 venv. Output: results/m3/samples_argmax/{method}/{N}/p{id}_s{seed}/ with seq_chunk03.wav,
final_latent.npy, clap_embed.npy, log.json (rewards, chosen = argmax, final_reward, replay checks, counts).

  .venv/bin/python m3/replay_argmax.py --methods lattice_smc_prefix_b1,fk_noise --Ns 8,32 --shard 0/2
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

# Amendment W: the same replay on the M4 (R_motif) grid; --src / --out select the root, the reward is read from
# each log ("clap" or "motif") and the argmax particle's reward is recomputed with the matching score function.
SRC = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3/samples"
OUT = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3/samples_argmax"


def replay_lattice(H, log, prompt, pid, seed, K, N, alpha, beta):
    g = ML.resample_generator(seed, "lattice_smc_prefix")
    prefix, lat = None, None
    logw, s_prev = np.zeros(N), np.zeros(N)
    scores = [np.asarray(s, dtype=np.float64) for s in log["prefix_scores"]]
    for k in range(K):
        lat = H.sample_chunk(prompt, pid, seed, k, N, prefix)
        s = scores[k]
        inc = (beta * (s - s_prev) / alpha) if k < K - 1 else ((s - beta * s_prev) / alpha)
        logw, s_prev = logw + inc, s
        if k < K - 1:
            if ML.ess(logw) < N / 2:
                parents = ML.systematic_resample(ML.normalized_weights(logw), g)
                pt = torch.as_tensor(parents, device=lat.device)
                lat, s_prev = lat[pt], s_prev[parents]
                logw = np.zeros(N)
        prefix = lat[..., :H.n_prefix_frames((k + 1) * ML.CHUNK_SEC)].clone()
    w = ML.normalized_weights(logw)
    idx_draw = ML.draw_index(w, g)
    return lat, w, idx_draw


def replay_fk(H, log, prompt, pid, seed, K, N, alpha):
    g = ML.resample_generator(seed, "fk_noise")
    hats = {(h["chunk"], h["step"]): np.asarray(h["s_hat"], dtype=np.float64) for h in log["x0_scores"]}
    state = {"logw": np.zeros(N), "s_prev": np.zeros(N)}
    prefix, lat = None, None
    for k in range(K):
        def hook(i, denoised, k=k):
            if i not in ML.FK_STEPS:
                return None
            s_hat = hats[(k + 1, i + 1)]
            state["logw"] = state["logw"] + (s_hat - state["s_prev"]) / alpha
            state["s_prev"] = s_hat
            if ML.ess(state["logw"]) >= N / 2:
                return None
            parents = ML.systematic_resample(ML.normalized_weights(state["logw"]), g)
            state["logw"] = np.zeros(N)
            state["s_prev"] = state["s_prev"][parents]
            return parents
        lat = H.sample_chunk(prompt, pid, seed, k, N, prefix, hook=hook)
        prefix = lat[..., :H.n_prefix_frames((k + 1) * ML.CHUNK_SEC)].clone()
    w = ML.normalized_weights(state["logw"])
    idx_draw = ML.draw_index(w, g)
    return lat, w, idx_draw


def main():
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="lattice_smc_prefix_b1,fk_noise")
    ap.add_argument("--Ns", default="8,32")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    SRC, OUT = a.src, a.out
    i_shard, n_shard = (int(x) for x in a.shard.split("/"))
    import soundfile as sf
    H = ML.MusicLattice()
    todo = []
    for m in a.methods.split(","):
        for N in (int(x) for x in a.Ns.split(",")):
            for d in sorted(os.listdir(os.path.join(SRC, m, str(N)))):
                todo.append((m, N, d))
    todo = [t for j, t in enumerate(todo) if j % n_shard == i_shard]
    if os.environ.get("REPLAY_SMOKE"):
        todo = [t for t in todo if t[2] in ("p00_s2000", "p04_s2000")]
        OUT_SMOKE = True
    else:
        OUT_SMOKE = False
    print(f"{len(todo)} sequences in shard {a.shard}", flush=True)
    for m, N, d in todo:
        out_dir = os.path.join(OUT + ("_smoke" if OUT_SMOKE else ""), m, str(N), d)
        if os.path.exists(os.path.join(out_dir, "log.json")):
            continue
        src = os.path.join(SRC, m, str(N), d)
        log = json.load(open(os.path.join(src, "log.json")))
        prompt, pid, seed, K, alpha = log["prompt"], log["prompt_id"], log["base_seed"], log["K"], log["alpha"]
        H.reward = log.get("reward", "clap")
        H.counter.reset()
        ML.DEC_CTX.decode_calls = 0
        H.clap_calls = 0
        t0 = time.time()
        if m.startswith("lattice_smc_prefix"):
            lat, w, idx_draw = replay_lattice(H, log, prompt, pid, seed, K, N, alpha, float(log["beta"]))
        else:
            lat, w, idx_draw = replay_fk(H, log, prompt, pid, seed, K, N, alpha)
        R = np.asarray(log["rewards"], dtype=np.float64)
        idx = int(np.argmax(R))
        stored = np.load(os.path.join(src, "final_latent.npy"))
        chosen_lat = lat[log["chosen"]].float().cpu().numpy()
        replay_ok = bool(np.array_equal(chosen_lat, stored))
        max_diff = float(np.abs(chosen_lat.astype(np.float64) - stored).max())
        weights_ok = bool(np.allclose(w, np.asarray(log["weights"]), atol=1e-12)) and idx_draw == log["chosen"]
        audio = H.decode(lat[idx:idx + 1], pid, seed, K * ML.CHUNK_SEC)
        R_re, emb = H.prefix_clap(audio, prompt, K) if H.reward == "clap" else H.motif_score(audio, K)
        torch.cuda.synchronize()
        os.makedirs(out_dir, exist_ok=True)
        sf.write(os.path.join(out_dir, "seq_chunk03.wav"), audio.T, H.sr, subtype="FLOAT")
        np.save(os.path.join(out_dir, "final_latent.npy"), lat[idx].float().cpu().numpy())
        np.save(os.path.join(out_dir, "clap_embed.npy"), np.asarray(emb, dtype=np.float32))
        new = {k: v for k, v in log.items() if k not in ("prefix_scores", "x0_scores")}
        new.update({"method": m + "_argmax", "return_rule": "argmax", "chosen": idx, "final_reward": float(R[idx]), "weights": w.tolist(),
                    "replay": {"chosen_latent_bitwise": replay_ok, "chosen_latent_max_abs_diff": max_diff, "weights_and_draw_match": weights_ok,
                               "argmax_reward_logged": float(R[idx]), "argmax_reward_recomputed_from_decode": R_re, "abs_diff": abs(R_re - float(R[idx])),
                               "draw_chosen": log["chosen"], "argmax_equals_draw": idx == log["chosen"]},
                    "nfe_replay": int(H.counter.nfe), "decode_calls_replay": int(ML.DEC_CTX.decode_calls), "clap_calls_replay": int(H.clap_calls),
                    "wall_replay": round(time.time() - t0, 3)})
        json.dump(new, open(os.path.join(out_dir, "log.json"), "w"), indent=1)
        print(f"{m} N={N} {d} replay_ok={replay_ok} maxdiff={max_diff:.2e} weights_ok={weights_ok} argmax={idx} draw={log['chosen']} "
              f"R_argmax={R[idx]:.4f} recomputed={R_re:.4f} wall={new['wall_replay']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
