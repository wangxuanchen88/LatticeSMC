"""Amendment G: the adopted checkpoint of a run is the saved checkpoint with the lowest held-out
monitor loss (ties -> earliest). Writes runs/<run>/adopted.json and copies it to ckpt_adopted.pt.

  python scripts/pick_ckpt.py --run runs/chunk_dispE
"""
import argparse
import csv
import json
import os
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
saved = sorted(int(m.group(1)) for f in os.listdir(a.run) for m in [re.match(r"ckpt_step(\d+)\.pt$", f)] if m)
ev = {int(r["step"]): float(r["heldout_loss"]) for r in csv.DictReader(open(os.path.join(a.run, "eval_log.csv")))}
cands = [(ev[s], s) for s in saved if s in ev]
best_loss, best_step = min(cands)  # tuple order: lowest loss, then earliest step
shutil.copy(os.path.join(a.run, f"ckpt_step{best_step}.pt"), os.path.join(a.run, "ckpt_adopted.pt"))
info = {"run": a.run, "rule": "lowest held-out monitor loss among saved checkpoints, ties -> earliest",
        "saved_steps": saved, "heldout_loss_at_saved": {str(s): ev[s] for s in saved if s in ev},
        "adopted_step": best_step, "adopted_heldout_loss": best_loss}
json.dump(info, open(os.path.join(a.run, "adopted.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
