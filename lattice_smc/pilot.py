"""Alpha pilot (SPEC 3.2): bon_is with N = 8 on the first 10 prompts, 2 seeds, alpha in
{0.02, 0.05, 0.1, 0.2}; take the largest alpha whose mean R_BA (of the drawn sample) improves
over base by at least 0.02. Base R_BA is recomputed from samples_E/base/1 with the Amendment I
pipeline. Writes data/alpha.json and results/alpha_pilot.json; samples under samples_pilot/.

  python -m lattice_smc.pilot --ckpt runs/chunk_dispE/ckpt_adopted.pt
"""
import argparse
import glob
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lattice_smc.data.pairs import PromptSet  # noqa: E402
from lattice_smc.edge_import import ROOT  # noqa: E402
from lattice_smc.generate import ALPHA_PATH, Context, generate_sequence, set_deterministic  # noqa: E402
from lattice_smc.model.small import load_checkpoint  # noqa: E402
from lattice_smc.rewards.counters import Counters  # noqa: E402
from lattice_smc.rewards.steering import RewardEvaluator  # noqa: E402

ALPHAS = [0.02, 0.05, 0.1, 0.2]
N_PROMPTS, SEEDS, N = 10, [2000, 2001], 8
MIN_GAIN = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_root", default="samples_E")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_deterministic()
    dev = torch.device(args.device)
    model, diffusion, ck = load_checkpoint(args.ckpt, dev)
    prompts = PromptSet()
    ids = prompts.prompt_ids[:N_PROMPTS]
    t0 = time.time()
    rows = []
    for pid in ids:
        music = prompts.music[prompts.index(pid)]
        ev = RewardEvaluator(music, diffusion.smpl, *__import__("lattice_smc.data.clips", fromlist=["load_normalizer"]).load_normalizer(), Counters())
        for seed in SEEDS:
            z = np.load(os.path.join(ROOT, args.base_root, "base", "1", f"{pid}_{seed}.npz"))
            base_R = float(ev.full(ev.joints(torch.from_numpy(z["motion"])[None].to(dev)))[0])
            row = {"prompt_id": pid, "seed": seed, "base": base_R}
            for alpha in ALPHAS:
                ctx = Context(model, diffusion, dev, ck["config"], alpha=alpha)
                motion, log = generate_sequence(ctx, music, prompts.K, seed, "bon_is", N)
                out_dir = os.path.join(ROOT, "samples_pilot", f"alpha{alpha}", "bon_is", str(N))
                os.makedirs(out_dir, exist_ok=True)
                np.savez(os.path.join(out_dir, f"{pid}_{seed}.npz"), motion=motion.cpu().numpy(), nfe=log["nfe"],
                         reward_evals=log["reward_evals"], wall=log["wall"], log=json.dumps(log))
                row[f"is_{alpha}"] = log["final_reward"]
                row[f"wmean_{alpha}"] = log["weighted_mean_reward"]
                row["argmax"] = float(max(log["rewards"]))
                row["particle_mean"] = float(np.mean(log["rewards"]))
                row["nfe"] = log["nfe"]
            rows.append(row)
            print(f"{pid} seed={seed} base={base_R:.4f} " + " ".join(f"a{a}={row[f'is_{a}']:.4f}" for a in ALPHAS) +
                  f" argmax={row['argmax']:.4f}", flush=True)
    base = np.mean([r["base"] for r in rows])
    table = {str(a): {"mean_R_drawn": float(np.mean([r[f"is_{a}"] for r in rows])),
                      "mean_weighted_mean": float(np.mean([r[f"wmean_{a}"] for r in rows])),
                      "gain_over_base": float(np.mean([r[f"is_{a}"] for r in rows]) - base)} for a in ALPHAS}
    ok = [a for a in ALPHAS if table[str(a)]["gain_over_base"] >= MIN_GAIN]
    alpha = max(ok) if ok else None
    out = {"rule": "largest alpha in {0.02, 0.05, 0.1, 0.2} with mean R_BA(bon_is draw) - mean R_BA(base) >= 0.02, "
                   "10 prompts x 2 seeds, N = 8", "n_sequences": len(rows), "base_mean": float(base),
           "argmax_mean": float(np.mean([r["argmax"] for r in rows])),
           "particle_mean": float(np.mean([r["particle_mean"] for r in rows])), "table": table,
           "qualifying": ok, "alpha": alpha, "nfe_per_sequence": rows[0]["nfe"], "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "alpha_pilot.json"), "w"), indent=1)
    if alpha is not None:
        json.dump({"alpha": alpha, "source": "results/alpha_pilot.json", "rule": out["rule"]}, open(ALPHA_PATH, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
