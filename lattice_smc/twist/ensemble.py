"""Amendment P: ensemble of trained twists, shrinkage constant c fitted on the validation
prefixes only, fresh-set residuals for the shrunk twist, the ensemble mean and the single twist.

  python -m lattice_smc.twist.ensemble --twist_dir data/twist_rep --alpha 0.02 --members data/twist_rep/ens_a002_s{0..4}.pt --single data/twist_rep/twist_a002.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from lattice_smc.edge_import import ROOT
from lattice_smc.twist.model import TwistEnsemble, load_twist
from lattice_smc.twist.train import features_for, load_rollouts, targets_from


def resid_stats(pred, target):
    r = target - pred
    return {"rmse_nats": float(np.sqrt((r ** 2).mean())), "median_abs": float(np.median(np.abs(r))),
            "p90_abs": float(np.quantile(np.abs(r), 0.9)), "bias": float(r.mean()),
            "r2": float(1 - (r ** 2).mean() / target.var())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twist_dir", default="data/twist_rep")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--out_prefix", default="data/twist_rep/ens_a002")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    K = 4
    members = [load_twist(p, dev) for p in args.members]
    single = load_twist(args.single, dev)
    ens = TwistEnsemble(members, "mean").to(dev).eval()
    # validation prefixes: the split stored in the first member's checkpoint (all members share it)
    ck0 = torch.load(args.members[0], map_location="cpu", weights_only=False)
    val_idx = np.array(ck0["val_idx"])
    train = load_rollouts("train", 4000, K, args.twist_dir)
    t_all, _, _ = targets_from(train["r_future"], args.alpha)
    with torch.no_grad():
        X_val = features_for({**train, "chunk": train["chunk"][val_idx], "segment": train["segment"][val_idx], "k": train["k"][val_idx]}, K, dev)
        m_val, v_val = ens.mean_var(X_val)
    m_val, v_val = m_val.cpu().numpy(), v_val.cpu().numpy()
    y_val, k_val = t_all[val_idx], train["k"][val_idx]
    s2 = np.array([y_val[k_val == k].var() for k in (1, 2, 3)])
    s2_row = s2[k_val - 1]
    grid = np.concatenate([[0.0], np.logspace(-3, 4, 141)])
    rmse = [np.sqrt(((y_val - m_val * s2_row / (s2_row + c * v_val)) ** 2).mean()) for c in grid]
    c = float(grid[int(np.argmin(rmse))])
    fresh = load_rollouts("fresh", 500, K, args.twist_dir)
    t_fresh, _, _ = targets_from(fresh["r_future"], args.alpha)
    with torch.no_grad():
        X_f = features_for(fresh, K, dev)
        m_f, v_f = ens.mean_var(X_f)
        p_single = single(X_f).cpu().numpy()
    m_f, v_f = m_f.cpu().numpy(), v_f.cpu().numpy()
    s2_f = s2[fresh["k"] - 1]
    shrunk_f = m_f * s2_f / (s2_f + c * v_f)
    out = {"alpha": args.alpha, "n_members": len(members), "members": args.members, "c": c, "s2_val_within_k": s2.tolist(),
           "val_rmse_single": float(np.sqrt(((y_val - single(X_val).detach().cpu().numpy()) ** 2).mean())),
           "val_rmse_mean": float(np.sqrt(((y_val - m_val) ** 2).mean())), "val_rmse_shrunk": float(min(rmse)),
           "val_rmse_grid": [(float(g), float(r)) for g, r in zip(grid[::10], rmse[::10])],
           "ensemble_var_val_mean": float(v_val.mean()), "ensemble_var_val_median": float(np.median(v_val)),
           "shrink_factor_fresh_mean": float((s2_f / (s2_f + c * v_f)).mean()), "shrink_factor_fresh_min": float((s2_f / (s2_f + c * v_f)).min()),
           "fresh": {"single": resid_stats(p_single, t_fresh), "ensemble_mean": resid_stats(m_f, t_fresh), "shrunk": resid_stats(shrunk_f, t_fresh)},
           "fresh_ensemble_var_mean": float(v_f.mean()), "fresh_member_spread_sd_mean": float(np.sqrt(v_f).mean())}
    for mode, cc in (("mean", 0.0), ("shrunk", c)):
        torch.save({"type": "ensemble", "K": K, "hidden": 512, "mode": mode, "c": cc, "s2": s2.tolist(), "alpha": args.alpha,
                    "members": [m.state_dict() for m in members]}, f"{args.out_prefix}_{mode}.pt")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "phase3c_ensemble.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "val_rmse_grid"}, indent=1))


if __name__ == "__main__":
    main()
