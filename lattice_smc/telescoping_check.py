"""Amendment L: along every particle lineage, the product of a chunk's five fk_noise potentials
equals exp((r_hat_50 - r_prefix) / alpha). Re-runs the Phase 2b alpha = 0.02 fk_noise sequences
with a trace (bitwise identical to the stored samples, verified), then checks
  sum_t logG_t (along the lineage)  ==  (r_hat_50 - r_prefix_start) / alpha
per slot and chunk, with r_prefix_start carried along the lineage at every resampling.

  python -m lattice_smc.telescoping_check --N 8 --prompts all --seeds 2000
"""
import argparse
import json
import os

import numpy as np
import torch

from lattice_smc.data.pairs import PromptSet
from lattice_smc.edge_import import ROOT
from lattice_smc.generate import Context, generate_sequence, set_deterministic
from lattice_smc.model.small import load_checkpoint
from lattice_smc.twist.model import load_twist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--prompts", default="all")
    ap.add_argument("--seeds", default="2000")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--samples", default="samples_p2_a002")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_deterministic()
    dev = torch.device(args.device)
    model, diffusion, ck = load_checkpoint("runs/chunk_dispE/ckpt_adopted.pt", dev)
    ctx = Context(model, diffusion, dev, ck["config"], alpha=args.alpha, twist=None)
    P = PromptSet()
    ids = P.prompt_ids if args.prompts == "all" else P.prompt_ids[: int(args.prompts)]
    seeds = [int(s) for s in args.seeds.split(",")]
    max_rel, max_abs, n_checked, n_bitwise, n_seq = 0.0, 0.0, 0, 0, 0
    worst = None
    for pid in ids:
        for seed in seeds:
            trace = []
            motion, log = generate_sequence(ctx, P.music[P.index(pid)], P.K, seed, "fk_noise", args.N, trace=trace)
            stored = np.load(os.path.join(ROOT, args.samples, "fk_noise", str(args.N), f"{pid}_{seed}.npz"))["motion"]
            n_bitwise += int(np.array_equal(motion.cpu().numpy().astype(np.float32), stored))
            n_seq += 1
            for k in range(1, P.K + 1):
                ev = [e for e in trace if e["chunk"] == k]
                assert len(ev) == 5
                cum = np.zeros(args.N)
                r_start = ev[0]["r_prev"].copy()  # the prefix reward at the chunk start, per slot
                for e in ev:
                    cum = cum + e["logG"]
                    if e["parents"] is not None:
                        cum = cum[e["parents"]]
                        r_start = r_start[e["parents"]]
                        last_r_hat = e["r_hat"][e["parents"]]
                    else:
                        last_r_hat = e["r_hat"]
                rhs = (last_r_hat - r_start) / args.alpha
                err = np.abs(cum - rhs)
                rel = err / np.maximum(np.abs(rhs), 1e-12)
                n_checked += args.N
                if rel.max() > max_rel:
                    max_rel, worst = float(rel.max()), {"prompt": pid, "seed": seed, "chunk": k, "lhs": float(cum[rel.argmax()]), "rhs": float(rhs[rel.argmax()])}
                max_abs = max(max_abs, float(err.max()))
    out = {"alpha": args.alpha, "N": args.N, "n_sequences": n_seq, "n_bitwise_equal_to_stored": n_bitwise,
           "n_particle_chunks_checked": n_checked, "max_abs_error_log": max_abs, "max_rel_error": max_rel, "worst": worst,
           "statement": "sum of the five log-potentials along the lineage == (r_hat_50 - r_prefix_start) / alpha"}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"telescoping_check_N{args.N}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
