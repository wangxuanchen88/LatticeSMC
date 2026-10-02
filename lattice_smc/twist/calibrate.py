"""Calibration table (SPEC 5): on 500 fresh prefixes with fresh rollouts, the learned twist's
log psi versus the realized log mean exp(S / alpha), and the plug-in noise-axis potential
r_hat / alpha (r_hat = r_{k+1} of the x0 prediction after DDIM steps 45, 35, 25, 15, 5 of chunk
k + 1 on continuation 0) versus (a) the realized next-chunk reward r_{k+1} / alpha of that same
continuation and (b) the same full-future target as the learned twist. Reports R^2, RMSE,
Pearson r, and the slope / intercept of realized on predicted, plus a binned calibration table
and a plot.

  python -m lattice_smc.twist.calibrate --fresh 500
"""
import argparse
import json
import os

import numpy as np
import torch

from lattice_smc.edge_import import ROOT
from lattice_smc.generate import load_alpha
from lattice_smc.twist.collect import PLUGIN_STEPS, TWIST_DIR
from lattice_smc.twist.model import TWIST_PATH, load_twist
from lattice_smc.twist.train import features_for, load_rollouts, targets_from


def stats(pred, real):
    pred, real = np.asarray(pred, float), np.asarray(real, float)
    ok = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[ok], real[ok]
    resid = real - pred
    var = real.var()
    slope, intercept = np.polyfit(pred, real, 1) if pred.std() > 0 else (float("nan"), float("nan"))
    return {"n": int(ok.sum()), "r2": float(1 - (resid ** 2).mean() / var), "rmse": float(np.sqrt((resid ** 2).mean())),
            "pearson_r": float(np.corrcoef(pred, real)[0, 1]) if pred.std() > 0 else float("nan"),
            "slope": float(slope), "intercept": float(intercept), "mean_pred": float(pred.mean()), "mean_real": float(real.mean()),
            "bias": float(resid.mean())}


def binned(pred, real, n_bins=8):
    pred, real = np.asarray(pred, float), np.asarray(real, float)
    ok = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[ok], real[ok]
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    rows = []
    for b in range(n_bins):
        m = (pred >= edges[b]) & (pred <= edges[b + 1] if b == n_bins - 1 else pred < edges[b + 1])
        if m.sum():
            rows.append({"bin": b, "n": int(m.sum()), "pred_mean": float(pred[m].mean()), "real_mean": float(real[m].mean()),
                         "real_se": float(real[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--twist", default=None)
    ap.add_argument("--out", default="twist_calibration", help="results/<out>.{json,png}")
    ap.add_argument("--twist_dir", default=TWIST_DIR)
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else load_alpha()
    dev = torch.device(args.device)
    K = 4
    data = load_rollouts("fresh", args.fresh, K, args.twist_dir)
    target, se, S = targets_from(data["r_future"], alpha)
    twist = load_twist(args.twist or TWIST_PATH, dev)
    with torch.no_grad():
        pred = twist(features_for(data, K, dev)).cpu().numpy()
    next_r = data["r_future"][:, 0, 0]  # continuation 0, chunk k+1
    out = {"alpha": alpha, "n_fresh": len(target), "mc_se_mean": float(se.mean()),
           "learned_twist_vs_future_target": stats(pred, target),
           "learned_twist_per_k": {str(k): stats(pred[data["k"] == k], target[data["k"] == k]) for k in (1, 2, 3)},
           "plugin": {}, "binned_learned": binned(pred, target), "binned_plugin": {}}
    for si, step in enumerate(PLUGIN_STEPS):
        p = data["plugin"][:, si] / alpha
        out["plugin"][str(step)] = {"vs_next_chunk_reward_same_continuation": stats(p, next_r / alpha),
                                    "vs_future_target": stats(p, target),
                                    "vs_future_target_k3_only": stats(p[data["k"] == 3], target[data["k"] == 3])}
        out["binned_plugin"][str(step)] = binned(p, next_r / alpha)
    # the learned twist against the next-chunk reward, for the same-target column of the plug-in
    out["learned_twist_vs_next_chunk_reward_k3"] = stats(pred[data["k"] == 3], next_r[data["k"] == 3] / alpha)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", args.out + ".json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(pred, target, s=6, alpha=0.5)
        lo, hi = min(pred.min(), target.min()), max(pred.max(), target.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[0].set_xlabel("learned log psi (predicted)")
        axes[0].set_ylabel("realized log mean exp(S / alpha), fresh rollouts")
        axes[0].set_title(f"learned twist, R2 = {out['learned_twist_vs_future_target']['r2']:.3f}")
        for si, step in enumerate(PLUGIN_STEPS):
            axes[1].scatter(data["plugin"][:, si] / alpha, next_r / alpha, s=5, alpha=0.4, label=f"step {step}")
        lo, hi = np.nanmin(data["plugin"] / alpha), np.nanmax(next_r / alpha)
        axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[1].set_xlabel("plug-in r_hat / alpha (x0 prediction)")
        axes[1].set_ylabel("realized r_{k+1} / alpha (same continuation)")
        axes[1].set_title("plug-in noise-axis potential")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results", args.out + ".png"), dpi=130)
    except Exception as e:  # the table is the deliverable; the figure is a convenience
        out["plot_error"] = str(e)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("binned")}, indent=2))


if __name__ == "__main__":
    main()
