"""Twist training (SPEC 5): MLP on (root-relative last 30 prefix frames, pooled music of the
remaining chunks, one-hot chunk index) -> log psi_k = log mean_m exp(S_m / alpha), S_m the
future reward of continuation m. 500 held-out prefixes for validation; MSE loss; the checkpoint
with the lowest validation MSE over a fixed number of epochs is kept (rule recorded). Reports the
Monte Carlo standard error of the targets and the validation R^2.

  python -m lattice_smc.twist.train --alpha <from data/alpha.json>
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from lattice_smc.data.pairs import DATA_DIR
from lattice_smc.generate import load_alpha
from lattice_smc.twist.collect import TWIST_DIR
from lattice_smc.twist.features import twist_features
from lattice_smc.twist.model import TWIST_PATH, TwistMLP

CHUNK_LEN = 150


def load_rollouts(set_name, n_prefixes, K=4, twist_dir=TWIST_DIR):
    """Prefixes in (sequence, k) order, first n_prefixes. Returns dict of arrays."""
    segs_music = np.load(os.path.join(twist_dir, "segments_music.npy"))
    files = sorted(glob.glob(os.path.join(twist_dir, "rollouts", set_name, "*.npz")))
    chunk, seg, ks, r_future, plugin, r_base = [], [], [], [], [], []
    for f in files:
        z = np.load(f)
        for a, k in enumerate(z["prefix_k"].tolist()):
            chunk.append(z["base_motion"][(k - 1) * CHUNK_LEN: k * CHUNK_LEN])
            seg.append(int(z["segment"]))
            ks.append(k)
            r_future.append(z["r_future"][a])
            plugin.append(z["plugin"][a])
            r_base.append(z["r_base"])
    n = min(n_prefixes, len(chunk))
    assert n == n_prefixes, f"only {len(chunk)} prefixes collected, {n_prefixes} requested"
    return {"chunk": np.stack(chunk[:n]), "segment": np.array(seg[:n]), "k": np.array(ks[:n]),
            "r_future": np.stack(r_future[:n]), "plugin": np.stack(plugin[:n]), "r_base": np.stack(r_base[:n]),
            "music": segs_music, "n_files": len(files)}


def targets_from(r_future, alpha):
    """S_m = sum over future chunks (nan-padded) of r_j; target = log mean exp(S_m / alpha);
    Monte Carlo standard error by the delta method: sd(exp(S/alpha)) / (sqrt(M) mean(exp(S/alpha)))."""
    S = np.nansum(r_future, axis=-1)  # [n, M]
    z = S / alpha
    zmax = z.max(1, keepdims=True)
    e = np.exp(z - zmax)
    target = np.log(e.mean(1)) + zmax[:, 0]
    se = e.std(1, ddof=1) / (np.sqrt(e.shape[1]) * e.mean(1))
    return target, se, S


@torch.no_grad()
def features_for(data, K, device, batch=256):
    music = torch.from_numpy(data["music"])
    out = []
    for i in range(0, len(data["k"]), batch):
        ch = torch.from_numpy(data["chunk"][i:i + batch]).to(device)
        feats = []
        for b in range(ch.shape[0]):
            k = int(data["k"][i + b])
            feats.append(twist_features(ch[b:b + 1], music[data["segment"][i + b]].to(device), k, K)[0])
        out.append(torch.stack(feats))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n_prefixes", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=None, help="validation split seed (default: --seed); Amendment P ensembles keep it at 0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="twist checkpoint path (default data/twist/twist.pt)")
    ap.add_argument("--info", default=None, help="train-info json path (default <twist_dir>/train_info.json)")
    ap.add_argument("--twist_dir", default=TWIST_DIR)
    ap.add_argument("--decomp_only", action="store_true", help="write the target statistics and variance decomposition, train nothing")
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else load_alpha()
    assert alpha is not None
    out_path = args.out or os.path.join(args.twist_dir, "twist.pt")
    info_path = args.info or os.path.join(args.twist_dir, "train_info.json")
    dev = torch.device(args.device)
    K = 4
    t0 = time.time()
    data = load_rollouts("train", args.n_prefixes, K, args.twist_dir)
    target, se, S = targets_from(data["r_future"], alpha)
    if args.decomp_only:
        E = np.exp(S / alpha)
        decomp = {}
        for k in (1, 2, 3):
            m = data["k"] == k
            across = float(E[m].mean(1).var(ddof=1)); within = float(E[m].var(1, ddof=1).mean())
            decomp[str(k)] = {"n_prefixes": int(m.sum()), "M": int(S.shape[1]), "target_mean": float(target[m].mean()), "target_var": float(target[m].var()),
                              "mc_se_mean": float(se[m].mean()), "mc_se_median": float(np.median(se[m])), "mc_se2_mean": float((se[m] ** 2).mean()),
                              "noise_ceiling_r2": float(1 - (se[m] ** 2).mean() / target[m].var()),
                              "S_mean": float(S[m].mean()), "S_sd_across_prefixes_of_mean": float(S[m].mean(1).std(ddof=1)), "S_sd_within_prefix": float(S[m].std(1, ddof=1).mean()),
                              "exp_mean": float(E[m].mean()), "var_across_prefixes_of_prefix_mean": across, "var_within_prefix_mean": within,
                              "across_over_within": across / within if within > 0 else float("inf"), "across_over_total": across / (across + within)}
        info = {"alpha": alpha, "reward": str(np.load(sorted(glob.glob(os.path.join(args.twist_dir, "rollouts", "train", "*.npz")))[0]).get("reward", "ba")),
                "n_prefixes": len(target), "variance_decomposition_exp_S_over_alpha": decomp, "seconds": round(time.time() - t0, 1)}
        json.dump(info, open(info_path, "w"), indent=1)
        print(json.dumps(info, indent=1))
        return
    X = features_for(data, K, dev)
    y = torch.from_numpy(target).float().to(dev)
    rng = np.random.default_rng(args.seed if args.split_seed is None else args.split_seed)
    perm = rng.permutation(len(y))
    val_idx, tr_idx = perm[: args.n_val], perm[args.n_val:]
    Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    torch.manual_seed(args.seed)
    model = TwistMLP(K).to(dev)
    model.set_standardization(Xtr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"val_mse": float("inf"), "epoch": -1, "state": None}
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), args.batch):
            idx = order[i:i + args.batch]
            loss = torch.nn.functional.mse_loss(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_mse = torch.nn.functional.mse_loss(model(Xtr), ytr).item()
            va_mse = torch.nn.functional.mse_loss(model(Xva), yva).item()
        hist.append((ep, tr_mse, va_mse))
        if va_mse < best["val_mse"]:
            best = {"val_mse": va_mse, "epoch": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva)
        pred_tr = model(Xtr)
    var_va = yva.var(unbiased=False).item()
    r2_va = 1 - best["val_mse"] / var_va
    r2_tr = 1 - torch.nn.functional.mse_loss(pred_tr, ytr).item() / ytr.var(unbiased=False).item()
    # baselines: constant, and the per-k mean of the training targets
    k_tr, k_va = data["k"][tr_idx], data["k"][val_idx]
    per_k = {k: float(ytr[torch.from_numpy(k_tr == k).to(dev)].mean()) for k in (1, 2, 3)}
    pred_k = torch.tensor([per_k[int(k)] for k in k_va], device=dev)
    r2_perk = 1 - torch.nn.functional.mse_loss(pred_k, yva).item() / var_va
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "K": K, "hidden": 512, "alpha": alpha, "best_epoch": best["epoch"],
                "val_mse": best["val_mse"], "val_r2": r2_va, "n_train": len(tr_idx), "n_val": len(val_idx),
                "val_idx": val_idx.tolist()}, out_path)
    # Amendment K: variance decomposition of exp(S / alpha) per k: across-prefix variance of the
    # per-prefix mean vs the mean within-prefix variance (over the 8 continuations)
    E = np.exp(S / alpha)
    decomp = {}
    for k in (1, 2, 3):
        m = data["k"] == k
        across = float(E[m].mean(1).var(ddof=1)); within = float(E[m].var(1, ddof=1).mean())
        decomp[str(k)] = {"mean_exp": float(E[m].mean()), "var_across_prefixes_of_prefix_mean": across,
                          "var_within_prefix_mean": within, "across_over_within": across / within,
                          "across_over_total": across / (across + within),
                          "target_var": float(target[m].var()), "target_mc_se2_mean": float((se[m] ** 2).mean()),
                          "noise_ceiling_r2": float(1 - (se[m] ** 2).mean() / target[m].var())}
    info = {"alpha": alpha, "n_prefixes": len(y), "n_train": len(tr_idx), "n_val": len(val_idx), "n_rollout_files": data["n_files"],
            "prefixes_per_k": {str(k): int((data["k"] == k).sum()) for k in (1, 2, 3)},
            "target_mean": float(target.mean()), "target_std": float(target.std()),
            "target_mean_per_k": {str(k): float(target[data["k"] == k].mean()) for k in (1, 2, 3)},
            "mc_se_mean": float(se.mean()), "mc_se_median": float(np.median(se)), "mc_se_90pct": float(np.quantile(se, 0.9)),
            "future_reward_S_mean": float(S.mean()), "future_reward_S_std_within_prefix_mean": float(S.std(1, ddof=1).mean()),
            "epochs": args.epochs, "best_epoch": best["epoch"], "rule": "checkpoint with the lowest validation MSE over the fixed epochs",
            "train_mse": float(torch.nn.functional.mse_loss(pred_tr, ytr).item()), "val_mse": best["val_mse"],
            "train_r2": r2_tr, "val_r2": r2_va, "val_r2_per_k_mean_baseline": r2_perk, "val_var": var_va,
            "val_r2_per_k": {str(k): float(1 - ((pred_va - yva)[torch.from_numpy(k_va == k).to(dev)] ** 2).mean().item()
                                           / max(yva[torch.from_numpy(k_va == k).to(dev)].var(unbiased=False).item(), 1e-12)) for k in (1, 2, 3)},
            "history_every_50": [h for h in hist if h[0] % 50 == 0 or h[0] == 1], "seconds": round(time.time() - t0, 1),
            "variance_decomposition_exp_S_over_alpha": decomp, "twist_path": out_path}
    json.dump(info, open(info_path, "w"), indent=1)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
