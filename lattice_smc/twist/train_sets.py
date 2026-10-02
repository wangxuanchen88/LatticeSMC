"""Amendment R part 2: within-set twist. Sets of 16 particles at k = 1 and 16 at k = 2 on the same
music (data/twist_rep/sets). Loss per set = cross-entropy(softmax(targets), softmax(pred)) +
MSE of the set-centred prediction against the set-centred target. Same MLP and features as the
learned twist; checkpoint by the held-out within-set loss (20 held-out segments x 2 seeds).
Reports the held-out within-set Spearman correlation per k and, on the saved Phase 3c oracle
sets (data/twist_rep/oracle_sets), the within-set Pearson / Spearman correlation with the oracle
log psi, before any grid.

  python -m lattice_smc.twist.train_sets --alpha 0.02 --out data/twist_rep/wstwist_a002.pt
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from lattice_smc.edge_import import ROOT
from lattice_smc.twist.collect import load_segments
from lattice_smc.twist.collect_sets import SETS_DIR
from lattice_smc.twist.features import twist_features
from lattice_smc.twist.model import TwistMLP

K = 4


def load_sets(device):
    segs, music = load_segments(K, os.path.join(ROOT, "data", "twist_rep"))
    sets = []
    for f in sorted(glob.glob(os.path.join(SETS_DIR, "*.npz"))):
        z = np.load(f)
        m = music[int(z["segment"])].to(device)
        for k in (1, 2):
            chunk = torch.from_numpy(z[f"chunk{k}"]).to(device)
            feats = twist_features(chunk, m, k, K)
            sets.append({"feats": feats, "target": torch.from_numpy(z[f"target_k{k}"]).float().to(device), "k": k,
                         "heldout": bool(z["heldout"]), "se": z[f"se_k{k}"], "file": os.path.basename(f)})
    return sets


def set_loss(pred, target):
    ce = -(torch.softmax(target, 0) * torch.log_softmax(pred, 0)).sum()
    reg = ((pred - pred.mean()) - (target - target.mean())).pow(2).mean()
    return ce + reg, ce, reg


def within_set_stats(model, sets):
    out = {1: {"spearman": [], "pearson": [], "loss": []}, 2: {"spearman": [], "pearson": [], "loss": []}}
    with torch.no_grad():
        for s in sets:
            p = model(s["feats"])
            l, _, _ = set_loss(p, s["target"])
            pn, tn = p.cpu().numpy(), s["target"].cpu().numpy()
            out[s["k"]]["spearman"].append(spearmanr(pn, tn).correlation)
            out[s["k"]]["pearson"].append(float(np.corrcoef(pn, tn)[0, 1]))
            out[s["k"]]["loss"].append(l.item())
    return {str(k): {n: float(np.mean(v)) for n, v in d.items()} for k, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "twist_rep", "wstwist_a002.pt"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()
    sets = load_sets(dev)
    train, held = [s for s in sets if not s["heldout"]], [s for s in sets if s["heldout"]]
    X_train = torch.cat([s["feats"] for s in train])
    torch.manual_seed(args.seed)
    model = TwistMLP(K).to(dev)
    model.set_standardization(X_train)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    hist = []
    g = torch.Generator().manual_seed(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        for i in torch.randperm(len(train), generator=g).tolist():
            s = train[i]
            loss, _, _ = set_loss(model(s["feats"]), s["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        st = within_set_stats(model, held)
        hl = np.mean([st["1"]["loss"], st["2"]["loss"]])
        hist.append((ep, float(hl), st["1"]["spearman"], st["2"]["spearman"]))
        if hl < best["loss"]:
            best = {"loss": float(hl), "epoch": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    model.eval()
    held_stats, train_stats = within_set_stats(model, held), within_set_stats(model, train)
    # oracle sets (Phase 3c, 5 prompts x N = 32, k = 1 and 2)
    oracle = {}
    segs, music = load_segments(K, os.path.join(ROOT, "data", "twist_rep"))
    from lattice_smc.data.pairs import PromptSet
    P = PromptSet()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "twist_rep", "oracle_sets", "*.npz"))):
        z = np.load(f)
        k = int(z["k"])
        chunks = torch.from_numpy(z["chunks"]).to(dev)  # [N, k, 150, 151]
        feats = twist_features(chunks[:, k - 1], P.music[P.index(str(z["prompt_id"]))].to(dev), k, K)
        with torch.no_grad():
            p = model(feats).cpu().numpy()
        o, l = z["logpsi_oracle"], z["logpsi_learned"]
        oracle.setdefault(str(k), []).append({"pearson_new_vs_oracle": float(np.corrcoef(p, o)[0, 1]), "spearman_new_vs_oracle": float(spearmanr(p, o).correlation),
                                              "pearson_old_vs_oracle": float(np.corrcoef(l, o)[0, 1]), "sd_new": float(p.std(ddof=1)), "sd_oracle": float(o.std(ddof=1))})
    oracle_summary = {k: {n: float(np.mean([r[n] for r in v])) for n in v[0]} | {"n_sets": len(v)} for k, v in oracle.items()}
    torch.save({"state_dict": model.state_dict(), "K": K, "hidden": 512, "alpha": args.alpha, "best_epoch": best["epoch"],
                "objective": "within-set cross-entropy + set-centred MSE", "held_out_loss": best["loss"]}, args.out)
    info = {"alpha": args.alpha, "n_sets_train": len(train), "n_sets_heldout": len(held), "n_particles": 16, "epochs": args.epochs, "best_epoch": best["epoch"],
            "heldout_within_set": held_stats, "train_within_set": train_stats,
            "target_within_set_sd_mean": {str(k): float(np.mean([s["target"].std(unbiased=True).item() for s in sets if s["k"] == k])) for k in (1, 2)},
            "mc_se_mean": {str(k): float(np.mean([s["se"].mean() for s in sets if s["k"] == k])) for k in (1, 2)},
            "oracle_sets": oracle_summary, "history_every_25": [h for h in hist if h[0] % 25 == 0 or h[0] == 1], "seconds": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(ROOT, "results", "phase3d_wstwist_train.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "history_every_25"}, indent=1))


if __name__ == "__main__":
    main()
