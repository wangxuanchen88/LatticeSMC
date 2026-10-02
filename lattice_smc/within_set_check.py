"""Amendment P part 1: within-set standard deviation of log psi at k = 1, 2 for the learned twist
(all 160 sequences at N = 32, deterministic re-run of the Phase 3 alpha = 0.02 lattice_smc runs,
verified bitwise against the stored samples) and, on a subset, the oracle log psi evaluated on
the SAME particles (M = 16), giving the within-set residual sd (learned - oracle, set mean
removed) and the within-set sd of the oracle itself.

  python -m lattice_smc.within_set_check --N 32 --prompts all --oracle_prompts 5
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
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--prompts", default="all")
    ap.add_argument("--seeds", default="2000,2001,2002,2003")
    ap.add_argument("--oracle_prompts", type=int, default=5, help="first n prompts (seed 2000) also get the oracle on the same particles")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_dir", default=None, help="save each traced set (particle chunks, learned and oracle log psi) as npz")
    ap.add_argument("--out", default="phase3c_within_set", help="results/<out>.json")
    args = ap.parse_args()
    set_deterministic()
    dev = torch.device(args.device)
    model, diffusion, ck = load_checkpoint("runs/chunk_dispE/ckpt_adopted.pt", dev)
    tw = load_twist("data/twist_rep/twist_a002.pt", dev)
    ctx = Context(model, diffusion, dev, ck["config"], alpha=0.02, twist=tw, reward="rep")
    ctx.oracle_M = 16
    P = PromptSet()
    ids = P.prompt_ids if args.prompts == "all" else P.prompt_ids[: int(args.prompts)]
    seeds = [int(s) for s in args.seeds.split(",")]
    sd_learned = {1: [], 2: [], 3: []}
    sd_oracle, sd_resid, se_oracle, corr = {1: [], 2: []}, {1: [], 2: []}, {1: [], 2: []}, {1: [], 2: []}
    n_bitwise = n_seq = 0
    for pi, pid in enumerate(ids):
        for seed in seeds:
            ctx.trace_oracle = pi < args.oracle_prompts and seed == seeds[0]
            trace = []
            motion, log = generate_sequence(ctx, P.music[P.index(pid)], P.K, seed, "lattice_smc", args.N, trace=trace)
            stored = np.load(os.path.join(ROOT, "samples_p3_rep_a002", "lattice_smc", str(args.N), f"{pid}_{seed}.npz"))["motion"]
            n_bitwise += int(np.array_equal(motion.cpu().numpy().astype(np.float32), stored))
            n_seq += 1
            for rec in trace:
                k = rec["chunk"]
                if args.save_dir and "oracle" in rec:
                    os.makedirs(args.save_dir, exist_ok=True)
                    np.savez(os.path.join(args.save_dir, f"{pid}_{seed}_k{k}.npz"), prompt_id=pid, seed=seed, k=k,
                             chunks=np.stack([c.cpu().numpy() for c in rec["chunks"]], axis=1), logpsi_learned=rec["logpsi"],
                             logpsi_oracle=rec["oracle"], oracle_se=rec["oracle_se"], r=rec["r"])
                sd_learned[k].append(float(rec["logpsi"].std(ddof=1)))
                if "oracle" in rec:
                    o, l = rec["oracle"], rec["logpsi"]
                    sd_oracle[k].append(float(o.std(ddof=1)))
                    r = l - o
                    sd_resid[k].append(float((r - r.mean()).std(ddof=1)))
                    se_oracle[k].append(float(rec["oracle_se"].mean()))
                    corr[k].append(float(np.corrcoef(l, o)[0, 1]))
            print(f"{pid} seed={seed} bitwise={np.array_equal(motion.cpu().numpy().astype(np.float32), stored)} sd_learned k1/k2/k3 = "
                  f"{trace[0]['logpsi'].std(ddof=1):.3f}/{trace[1]['logpsi'].std(ddof=1):.3f}/{trace[2]['logpsi'].std(ddof=1):.3f}"
                  + (f" | oracle sd {sd_oracle[1][-1]:.3f}/{sd_oracle[2][-1]:.3f} resid sd {sd_resid[1][-1]:.3f}/{sd_resid[2][-1]:.3f} corr {corr[1][-1]:.2f}/{corr[2][-1]:.2f}" if ctx.trace_oracle else ""), flush=True)
    out = {"N": args.N, "n_sequences": n_seq, "n_bitwise_equal_to_stored": n_bitwise,
           "learned_within_set_sd_logpsi_mean": {str(k): float(np.mean(v)) for k, v in sd_learned.items()},
           "learned_within_set_sd_logpsi_median": {str(k): float(np.median(v)) for k, v in sd_learned.items()},
           "oracle_subset_n_sets": {str(k): len(v) for k, v in sd_oracle.items()},
           "oracle_within_set_sd_logpsi_mean": {str(k): float(np.mean(v)) for k, v in sd_oracle.items() if v},
           "within_set_residual_sd_learned_minus_oracle_mean": {str(k): float(np.mean(v)) for k, v in sd_resid.items() if v},
           "oracle_mc_se_nats_mean": {str(k): float(np.mean(v)) for k, v in se_oracle.items() if v},
           "within_set_corr_learned_oracle_mean": {str(k): float(np.mean(v)) for k, v in corr.items() if v},
           "oracle_nfe_total": ctx.oracle_counters.nfe.n}
    json.dump(out, open(os.path.join(ROOT, "results", args.out + ".json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
