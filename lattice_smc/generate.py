"""generate_sequence(music, K, seed, method, N): every inference-time method behind one interface,
plus the CLI that fills samples/{method}/{N}/{prompt}_{seed}.npz (SPEC 2.4).

  python -m lattice_smc.generate --ckpt runs/chunk_s101/ckpt_final.pt --method base --N 1 \
      --prompts all --seeds 2000-2003 [--out_root samples] [--shard 0/2]

Determinism: (checkpoint, prompt, seed, method, N) determines the output. Noise comes from CPU
generators seeded per (seed, chunk, particle); the network runs with deterministic algorithms,
TF32 off and the math SDPA kernel only; each (prompt, seed) sequence is generated on its own so
the batch composition is a function of (method, N) alone. Existing outputs are skipped.
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lattice_smc.data.clips import load_normalizer  # noqa: E402
from lattice_smc.data.pairs import PromptSet  # noqa: E402
from lattice_smc.edge_import import ROOT  # noqa: E402
from lattice_smc.methods import METHODS  # noqa: E402
from lattice_smc.model.small import load_checkpoint  # noqa: E402
from lattice_smc.model.window import loss_cfg_from  # noqa: E402
from lattice_smc.rewards.counters import Counters  # noqa: E402
from lattice_smc.twist.model import TWIST_PATH, load_twist  # noqa: E402
import yaml  # noqa: E402

SAMPLES_DIR = os.path.join(ROOT, "samples")


def set_deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


ALPHA_PATH = os.path.join(ROOT, "data", "alpha.json")


def load_alpha(path=ALPHA_PATH):
    return float(json.load(open(path))["alpha"]) if os.path.exists(path) else None


class Context:
    """What every method needs: the model, the diffusion buffers (incl. the SMPL skeleton), the
    counters, the device, the normalizer, the checkpoint's representation, alpha and the twist."""

    def __init__(self, model, diffusion, device, config=None, alpha=None, twist=None, reward="ba"):
        self.model, self.diffusion, self.device = model, diffusion, device
        self.counters = Counters()
        self.loss_cfg = loss_cfg_from(config)  # representation and loss variant of the checkpoint
        self.scale, self.mn = load_normalizer()
        self.alpha, self.twist, self.reward = alpha, twist, reward
        self.twist_mode, self.twist_beta, self.oracle_M = "learned", 1.0, 16  # Amendment N
        self.oracle_counters = Counters()


def generate_sequence(ctx, music, K, seed, method, N, **kw):
    """music [K*150, 35] tensor. Returns (motion [K*150, 151] normalized, log dict with nfe,
    reward_evals, twist_evals, wall and the method's own entries)."""
    ctx.counters.reset()
    t0 = time.time()
    motion, log = METHODS[method](ctx, music.to(ctx.device), K, seed, N, **kw)
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)
    log.update(ctx.counters.as_dict())
    log.update({"wall": round(time.time() - t0, 4), "method": method, "N": N, "seed": seed, "K": K})
    return motion, log


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def run_jobs(ctx, prompts, jobs, out_root, ckpt, ck_step):
    """jobs: list of (method, N, prompt_id, seed); skips existing outputs."""
    K = prompts.K
    t_all = time.time()
    n_done = n_skip = 0
    for method, N, pid, seed in jobs:
        out_dir = os.path.join(out_root, method, str(N))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{pid}_{seed}.npz")
        if os.path.exists(path):
            n_skip += 1
            continue
        music = prompts.music[prompts.index(pid)]
        motion, log = generate_sequence(ctx, music, K, seed, method, N)
        log.update({"prompt_id": pid, "ckpt": os.path.relpath(ckpt, ROOT), "ckpt_step": ck_step, "loss_cfg": ctx.loss_cfg,
                    "alpha": ctx.alpha, "reward": ctx.reward, "device": str(ctx.device),
                    "gpu": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "cpu",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")})
        np.savez(path, motion=motion.cpu().numpy().astype(np.float32), nfe=log["nfe"],
                 reward_evals=log["reward_evals"], wall=log["wall"], log=json.dumps(log))
        n_done += 1
        print(f"{method} N={N} {pid} seed={seed} nfe={log['nfe']} rew={log['reward_evals']} "
              f"R={log.get('final_reward', float('nan')):.4f} wall={log['wall']:.2f}s", flush=True)
    print(f"done: {n_done} generated, {n_skip} skipped, {time.time() - t_all:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--method", default="base", choices=sorted(METHODS))
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--prompts", default="all", help="'all', an integer (first n), or comma-separated ids")
    ap.add_argument("--seeds", default="2000-2003")
    ap.add_argument("--out_root", default=SAMPLES_DIR)
    ap.add_argument("--shard", default="0/1", help="i/n: take prompts with index % n == i")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=None, help="default: data/alpha.json")
    ap.add_argument("--twist", default=None, help="default: data/twist/twist.pt if it exists")
    ap.add_argument("--grid", default=None, help="YAML grid: ckpt, methods, Ns, seeds, out_root [, prompts, alpha, twist, reward]")
    ap.add_argument("--reward", default=None, choices=[None, "ba", "rep"], help="steering reward (default: grid key or ba)")
    args = ap.parse_args()

    set_deterministic()
    dev = torch.device(args.device)
    grid = yaml.safe_load(open(args.grid)) if args.grid else None
    ckpt = args.ckpt or (grid or {}).get("ckpt")
    assert ckpt, "--ckpt or a grid with ckpt is required"
    model, diffusion, ck = load_checkpoint(ckpt, dev)
    alpha = args.alpha if args.alpha is not None else ((grid or {}).get("alpha") if (grid or {}).get("alpha") is not None else load_alpha())
    twist_path = args.twist or (grid or {}).get("twist") or (TWIST_PATH if os.path.exists(TWIST_PATH) else None)
    twist = load_twist(twist_path, dev) if twist_path else None
    reward = args.reward or (grid or {}).get("reward", "ba")
    ctx = Context(model, diffusion, dev, ck["config"], alpha=alpha, twist=twist, reward=reward)
    ctx.twist_mode = (grid or {}).get("twist_mode", "learned")
    ctx.twist_beta = float((grid or {}).get("twist_beta", 1.0))
    ctx.oracle_M = int((grid or {}).get("oracle_M", 16))
    ctx.rollout_M = int((grid or {}).get("rollout_M", 1))
    ctx.return_rule = (grid or {}).get("return_rule", "draw")  # Amendment U: "draw" (default) or "argmax"
    ctx.prune_M = (grid or {}).get("prune_M")  # R1 Phase 2: chunk-pruning M (int or "N/2"); None = max(1, N // 4)
    if (grid or {}).get("prompts_file"):  # R1 Phase 3: K = 6 / 8 prompt lists (also read by PromptSet via LATTICE_PROMPTS)
        os.environ["LATTICE_PROMPTS"] = os.path.join(ROOT, (grid or {})["prompts_file"])
    prompts = PromptSet()
    ids = prompts.prompt_ids
    psel = (grid or {}).get("prompts", args.prompts) if grid else args.prompts
    if psel != "all":
        ids = ids[: int(psel)] if str(psel).isdigit() else str(psel).split(",")
    si, sn = (int(v) for v in args.shard.split("/"))
    ids = [p for j, p in enumerate(ids) if j % sn == si]
    if grid:
        methods, Ns, seeds, out_root = grid["methods"], grid["Ns"], parse_seeds(str(grid["seeds"])), grid.get("out_root", SAMPLES_DIR)
    else:
        methods, Ns, seeds, out_root = [args.method], [args.N], parse_seeds(args.seeds), args.out_root
    jobs = [(m, N, pid, seed) for N in Ns for m in methods for pid in ids for seed in seeds
            if not (m == "base" and N != 1)]
    print(f"ckpt={ckpt} step={ck['step']} methods={methods} Ns={Ns} K={prompts.K} prompts={len(ids)} "
          f"seeds={seeds} alpha={alpha} reward={reward} twist={twist_path} device={dev} out={out_root} jobs={len(jobs)}", flush=True)
    run_jobs(ctx, prompts, jobs, out_root, ckpt, ck["step"])


if __name__ == "__main__":
    main()
