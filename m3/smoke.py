"""M3 smoke test (SPEC 11i): (1) N = 1 reproduces the M2 base sample bitwise (latent and wav); (2) every method at
N = 1 equals base; (3) lattice_smc_prefix(beta = 0) == bon_is bitwise at N = 4; (4) two runs are bitwise equal;
(5) NFE = 8 K N rows for every method; (6) wall per sequence at N = 8 for every method on 2 prompts x 1 seed."""
import os
import json, os, sys, time
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m3")
import numpy as np, torch, soundfile as sf
import music_lattice as ML

OUT = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3/smoke"
os.makedirs(OUT, exist_ok=True)
prompts = json.load(open(ML.PROMPTS))
alpha = float(json.load(open(ML.ALPHA_FILE))["alpha"])
H = ML.MusicLattice()
rep = {"alpha": alpha, "load_s": H.load_s}
K = 4

def key(res):  # bitwise fingerprint of a result: final latent of every particle, rewards, chosen
    lat, R, emb, audio, log, extra = res
    return lat.float().cpu().numpy().tobytes(), np.asarray(R).tobytes(), log["chosen"], extra["nfe"]

# (1) N = 1 vs M2 base sample, prompt 0 and 4, seed 2000
m2 = []
for pid in (0, 4):
    prec = prompts[pid]
    r = ML.generate(H, "base", prec, 2000, 1, K, alpha)["base"]
    lat, R, emb, audio, log, extra = r
    d = fos.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/b2/samples/p{pid:02d}_s2000"
    L2 = np.load(os.path.join(d, "seq_chunk3_latents.npy"))
    w2, _ = sf.read(os.path.join(d, "seq_chunk03.wav"), dtype="float32", always_2d=True)
    sc2 = None
    m2.append({"prompt_id": pid, "latent_bitwise": bool(np.array_equal(lat.float().cpu().numpy(), L2)),
               "latent_max_abs_diff": float(np.abs(lat.float().cpu().numpy().astype(np.float64) - L2).max()),
               "wav_bitwise": bool(np.array_equal(audio[0].T, w2)), "wav_max_abs_diff": float(np.abs(audio[0].T.astype(np.float64) - w2).max()),
               "terminal_clap_m3": float(R[0]), "nfe": extra["nfe"], "decode_calls": extra["decode_calls"], "wall": extra["wall"]})
rep["n1_vs_m2_base"] = m2
print(json.dumps(m2, indent=1), flush=True)

# (2) every method at N = 1 equals base (prompt 0, seed 2000)
prec = prompts[0]
base = ML.generate(H, "base", prec, 2000, 1, K, alpha)["base"]
same = {}
for run in ("bon", "greedy_chunk", "lsp1", "lsp05", "lsp025", "fk_noise"):
    method, beta = ML.METHOD_RUNS[run]
    res = ML.generate(H, method, prec, 2000, 1, K, alpha, beta)
    for name, r in res.items():
        same[name] = {"latent_bitwise": key(r)[0] == key(base)[0], "reward_equal": key(r)[1] == key(base)[1], "nfe": r[5]["nfe"]}
rep["n1_equals_base"] = same
print(json.dumps(same, indent=1), flush=True)

# (3) beta = 0 == bon_is at N = 4; (4) determinism; (5) NFE
a = ML.generate(H, "bon", prec, 2000, 4, K, alpha)["bon_is"]
b = ML.generate(H, "lattice_smc_prefix", prec, 2000, 4, K, alpha, 0.0)["lattice_smc_prefix_b0"]
b2 = ML.generate(H, "lattice_smc_prefix", prec, 2000, 4, K, alpha, 0.0)["lattice_smc_prefix_b0"]
rep["beta0_equals_bon_is_N4"] = {"latents_bitwise": key(a)[0] == key(b)[0], "rewards_bitwise": key(a)[1] == key(b)[1],
                                 "chosen_equal": key(a)[2] == key(b)[2], "weights_equal": a[4]["weights"] == b[4]["weights"],
                                 "resampled_events": [e["resampled"] for e in b[4]["ess_content"]]}
rep["determinism_lsp0_N4"] = {"latents_bitwise": key(b)[0] == key(b2)[0], "rewards_bitwise": key(b)[1] == key(b2)[1], "chosen_equal": key(b)[2] == key(b2)[2]}
print(json.dumps({k: rep[k] for k in ("beta0_equals_bon_is_N4", "determinism_lsp0_N4")}, indent=1), flush=True)

# (6) wall per sequence at N = 8, every method, 2 prompts x seed 2000; determinism of fk_noise at N = 8 on one prompt
walls = {}
for pid in (0, 4):
    prec = prompts[pid]
    for run in ("bon", "greedy_chunk", "lsp1", "lsp05", "lsp025", "fk_noise"):
        method, beta = ML.METHOD_RUNS[run]
        res = ML.generate(H, method, prec, 2000, 8, K, alpha, beta)
        for name, r in res.items():
            lat, R, emb, audio, log, extra = r
            walls.setdefault(name, []).append({"prompt_id": pid, "wall": extra["wall"], "nfe": extra["nfe"], "decode_calls": extra["decode_calls"],
                                               "clap_calls": extra["clap_calls"], "final_reward": log["final_reward"],
                                               "resampled": [e["resampled"] for e in (log["ess_content"] or log["ess_noise"]) if not e.get("final_draw")],
                                               "telescoping": log.get("telescoping_max_abs_err")})
            if pid == 0 and name == "fk_noise":
                fk1 = key(r)
        print(run, pid, {n: walls[n][-1]["wall"] for n in res}, flush=True)
fk2 = key(ML.generate(H, "fk_noise", prompts[0], 2000, 8, K, alpha)["fk_noise"])
rep["determinism_fk_noise_N8"] = {"latents_bitwise": fk1[0] == fk2[0], "rewards_bitwise": fk1[1] == fk2[1], "chosen_equal": fk1[2] == fk2[2]}
rep["walls_N8"] = walls
rep["wall_N8_mean_per_run"] = {n: float(np.mean([w["wall"] for w in v])) for n, v in walls.items()}
json.dump(rep, open(os.path.join(OUT, "smoke.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("determinism_fk_noise_N8", "wall_N8_mean_per_run")}, indent=1), flush=True)
