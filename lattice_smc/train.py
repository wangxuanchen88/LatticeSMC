# Copied from the earlier harness: <earlier-harness>/train.py (working tree; the
# the earlier harness repo has no git commit). sha256 of the source file:
# 5ef90feeb373d28b37d413643d6224067e2d047726ce82b12a679a3d7199cc71; EDGE pin
# 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications (SPEC 2.1-2.3): pairs instead of clips (300-frame windows); per-batch
# unconditional-start mask with p_uncond; chunk_loss (frames 150..299) instead of the span loss;
# the held-out monitor is the chunk loss on held-out-sequence pairs at fixed (t, noise); no span
# masks; skip-if-done keys on train_summary.json (there is no eval.py in this repo).
"""One deterministic training run from a YAML config.

  python -m lattice_smc.train --config configs/chunk_base.yaml --run_name chunk_s101 --seed 101

Everything random is driven by `seed`: init (torch), data order (numpy Generator), timesteps,
noise, the unconditional-start mask and the music-dropout mask (torch CUDA generator). The whole
pair pool lives on the GPU, so there is no DataLoader and no worker nondeterminism.
"""
import argparse
import csv
import json
import os
import shutil
import time

# Must be set before CUDA initialises for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lattice_smc.data.pairs import CHUNK_LEN, PairSet  # noqa: E402
from lattice_smc.edge_import import ROOT  # noqa: E402
from lattice_smc.model.loss import chunk_loss  # noqa: E402
from lattice_smc.model.small import build_diffusion, build_model, count_params  # noqa: E402
from lattice_smc.model.window import loss_cfg_from  # noqa: E402
from model.adan import Adan  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "runs")
MONITOR_T = [300, 500, 700]


def set_deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Flash / memory-efficient SDPA backward kernels are not bitwise reproducible.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def save_steps_for(cfg):
    total = cfg["total_steps"]
    if cfg.get("save_every"):
        return list(range(int(cfg["save_every"]), total + 1, int(cfg["save_every"])))
    fracs = cfg.get("save_fracs", [0.5, 0.75, 1.0])
    return sorted({int(round(f * total)) for f in fracs})


def monitor_noise(n, t, dtype=torch.float32):
    """Fixed CPU-generated noise per timestep so the monitor does not depend on the device."""
    g = torch.Generator(device="cpu").manual_seed(20_000 + int(t))
    return torch.randn(n, CHUNK_LEN, 151, generator=g, dtype=dtype)


@torch.no_grad()
def heldout_monitor(model, diffusion, heldout, loss_cfg, batch_size=64):
    """Conditional chunk loss on held-out-sequence pairs at fixed t and noise (mean over
    MONITOR_T). Returns (weighted loss, simple term)."""
    dev = heldout.motion.device
    n = len(heldout)
    tot, simple = 0.0, 0.0
    for t_val in MONITOR_T:
        noise_all = monitor_noise(n, t_val).to(dev)
        for i in range(0, n, batch_size):
            x0 = heldout.motion[i:i + batch_size]
            mu = heldout.music[i:i + batch_size]
            b = len(x0)
            t = torch.full((b,), t_val, device=dev, dtype=torch.long)
            unc = torch.zeros(b, dtype=torch.bool, device=dev)
            loss, terms, _ = chunk_loss(diffusion, model, x0, mu, t, noise_all[i:i + b], unc, 0.0, cfg=loss_cfg)
            tot += loss.item() * b
            simple += terms["simple"].item() * b
    return tot / (n * len(MONITOR_T)), simple / (n * len(MONITOR_T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--n_pairs", type=int, default=None, help="use only the first n pairs (smoke tests)")
    ap.add_argument("--eval_every", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, {"seed": args.seed, "total_steps": args.total_steps,
                                    "n_pairs": args.n_pairs, "eval_every": args.eval_every})
    cfg["run_name"] = args.run_name
    run_dir = os.path.join(RUNS_DIR, args.run_name)
    if os.path.exists(os.path.join(run_dir, "train_summary.json")):
        print(f"{run_dir}/train_summary.json exists, skipping")
        return
    os.makedirs(run_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(run_dir, "config.yaml"), "w"), sort_keys=False)

    set_deterministic()
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda:0")

    train = PairSet(cfg["pairs"], device=dev, n=cfg.get("n_pairs"))
    heldout = PairSet(cfg["heldout_pairs"], device=dev, n=cfg.get("n_heldout_monitor"))

    model = build_model(cfg.get("model")).to(dev)
    diffusion = build_diffusion(model, dev)
    n_params = count_params(model)
    optim = Adan(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.02)))

    N = len(train)
    B = min(int(cfg["batch_size"]), N)
    total = int(cfg["total_steps"])
    save_steps = save_steps_for(cfg)
    eval_every = int(cfg.get("eval_every", 0) or 0)
    p_uncond = float(cfg.get("p_uncond", 0.2))
    cond_drop = float(cfg.get("cond_drop_prob", 0.25))
    loss_cfg = loss_cfg_from(cfg)
    print(f"run={args.run_name} seed={seed} pairs={N} heldout_monitor={len(heldout)} params={n_params} "
          f"steps={total} batch={B} p_uncond={p_uncond} cond_drop={cond_drop} loss={loss_cfg} save={save_steps}", flush=True)

    log_f = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "loss", "simple", "vel", "fk", "foot", "seam", "wall"])
    eval_f = open(os.path.join(run_dir, "eval_log.csv"), "w", newline="")
    elog = csv.writer(eval_f)
    elog.writerow(["step", "heldout_loss", "heldout_simple", "wall"])

    order = rng.permutation(N)
    pos = 0
    t0 = time.time()
    model.train()
    for step in range(1, total + 1):
        if pos + B > N:
            order = rng.permutation(N)
            pos = 0
        idx = torch.from_numpy(order[pos: pos + B]).to(dev)
        pos += B
        x0, music = train.motion[idx], train.music[idx]
        t = torch.randint(0, diffusion.n_timestep, (B,), device=dev).long()
        noise = torch.randn(B, CHUNK_LEN, 151, device=dev)
        uncond = torch.rand(B, device=dev) < p_uncond
        loss, terms, _ = chunk_loss(diffusion, model, x0, music, t, noise, uncond, cond_drop, cfg=loss_cfg)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % 10 == 0 or step == 1:
            log.writerow([step, f"{loss.item():.6f}"] + [f"{terms[k].item():.6f}" for k in ["simple", "vel", "fk", "foot"]]
                         + [f"{terms['seam'].item():.6f}" if "seam" in terms else "0", f"{time.time() - t0:.1f}"])
            log_f.flush()
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} wall {time.time() - t0:.0f}s", flush=True)
        if eval_every and (step % eval_every == 0 or step == total):
            model.eval()
            hl, hs = heldout_monitor(model, diffusion, heldout, loss_cfg)
            model.train()
            elog.writerow([step, f"{hl:.6f}", f"{hs:.6f}", f"{time.time() - t0:.1f}"])
            eval_f.flush()
        if step in save_steps:
            ck = {"model_state_dict": model.state_dict(), "step": step, "config": cfg, "n_params": n_params}
            torch.save(ck, os.path.join(run_dir, f"ckpt_step{step}.pt"))
    wall = time.time() - t0
    shutil.copy(os.path.join(run_dir, f"ckpt_step{total}.pt"), os.path.join(run_dir, "ckpt_final.pt"))
    json.dump({"n_params": n_params, "total_steps": total, "batch_size": B, "n_pairs": N,
               "train_wall_seconds": round(wall, 1), "steps_per_second": round(total / wall, 3),
               "gpu": torch.cuda.get_device_name(dev)},
              open(os.path.join(run_dir, "train_summary.json"), "w"), indent=2)
    print(f"done: {wall:.0f}s, {total / wall:.2f} steps/s", flush=True)


if __name__ == "__main__":
    main()
