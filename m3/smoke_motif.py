"""Amendment V smoke test (reward motif): every method at N = 1 equals base; beta = 0 == bon_is at N = 4; determinism; NFE
and decoder / CLAP call counts; wall at N = 8 on 2 prompts x seed 2000; telescoping. Writes results/m4/smoke/smoke.json."""
import os
import json, os, sys
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m3")
import numpy as np
import music_lattice as ML

OUT = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m4/smoke"; os.makedirs(OUT, exist_ok=True)
prompts = json.load(open(ML.PROMPTS)); alpha = float(json.load(open(os.environ.get("LATTICESMC_ROOT", ".") + "/results/m4/alpha.json"))["alpha"])
H = ML.MusicLattice(); H.reward = "motif"; K = 4
key = lambda r: (r[0].float().cpu().numpy().tobytes(), np.asarray(r[1]).tobytes(), r[4]["chosen"], r[5]["nfe"])  # noqa: E731
rep = {"alpha": alpha, "reward": "motif"}
prec = prompts[0]
base = ML.generate(H, "base", prec, 2000, 1, K, alpha)["base"]
rep["base_N1"] = {"R_motif": float(base[1][0]), "nfe": base[5]["nfe"], "decode_calls": base[5]["decode_calls"], "clap_calls": base[5]["clap_calls"]}
same = {}
for run in ("bon", "greedy_chunk", "lsp1", "lsp05", "lsp025", "fk_noise"):
    m, b = ML.METHOD_RUNS[run]
    for name, r in ML.generate(H, m, prec, 2000, 1, K, alpha, b).items():
        same[name] = {"latent_bitwise": key(r)[0] == key(base)[0], "reward_equal": key(r)[1] == key(base)[1], "nfe": r[5]["nfe"], "decode_calls": r[5]["decode_calls"]}
rep["n1_equals_base"] = same
a = ML.generate(H, "bon", prec, 2000, 4, K, alpha)["bon_is"]
b = ML.generate(H, "lattice_smc_prefix", prec, 2000, 4, K, alpha, 0.0)["lattice_smc_prefix_b0"]
b2 = ML.generate(H, "lattice_smc_prefix", prec, 2000, 4, K, alpha, 0.0)["lattice_smc_prefix_b0"]
rep["beta0_equals_bon_is_N4"] = {"latents_bitwise": key(a)[0] == key(b)[0], "rewards_bitwise": key(a)[1] == key(b)[1], "chosen_equal": key(a)[2] == key(b)[2], "weights_equal": a[4]["weights"] == b[4]["weights"]}
rep["determinism_lsp0_N4"] = {"latents_bitwise": key(b)[0] == key(b2)[0], "chosen_equal": key(b)[2] == key(b2)[2]}
walls = {}
for pid in (0, 4):
    prec = prompts[pid]
    for run in ("bon", "greedy_chunk", "lsp1", "lsp05", "lsp025", "fk_noise"):
        m, b = ML.METHOD_RUNS[run]
        for name, r in ML.generate(H, m, prec, 2000, 8, K, alpha, b).items():
            lat, R, emb, audio, log, extra = r
            walls.setdefault(name, []).append({"prompt_id": pid, "wall": extra["wall"], "nfe": extra["nfe"], "decode_calls": extra["decode_calls"], "clap_calls": extra["clap_calls"],
                                               "final_reward": log["final_reward"], "resampled": [e["resampled"] for e in (log["ess_content"] or log["ess_noise"]) if not e.get("final_draw")],
                                               "telescoping": log.get("telescoping_max_abs_err")})
            if pid == 0 and name == "fk_noise":
                fk1 = key(r)
        print(run, pid, {n: walls[n][-1]["wall"] for n in walls if walls[n][-1]["prompt_id"] == pid}, flush=True)
fk2 = key(ML.generate(H, "fk_noise", prompts[0], 2000, 8, K, alpha)["fk_noise"])
rep["determinism_fk_noise_N8"] = {"latents_bitwise": fk1[0] == fk2[0], "chosen_equal": fk1[2] == fk2[2]}
rep["walls_N8"] = walls
rep["wall_N8_mean_per_run"] = {n: float(np.mean([w["wall"] for w in v])) for n, v in walls.items()}
json.dump(rep, open(os.path.join(OUT, "smoke.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "walls_N8"}, indent=1))
