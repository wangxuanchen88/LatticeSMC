"""M3 held-out audiobox-aesthetics on the returned clips (isolated venv m2/.venv-aes), N in --Ns plus base. Resumable.
  m2/.venv-aes/bin/python m3/score_aes.py --samples results/m3/samples --out results/m3/aesthetics.json --Ns 8,32
"""
import os
import argparse, json, os, sys
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m3")
from audiobox_aesthetics.infer import initialize_predictor



def returned_wav(sd):
    """The returned clip: seq_chunk03.wav in the K = 4 grids, seq_chunk{K-1:02d}.wav in the R1 K = 8 runs."""
    p = os.path.join(sd, "seq_chunk03.wav")
    if os.path.exists(p):
        return p
    c = sorted(x for x in os.listdir(sd) if x.startswith("seq_chunk") and x.endswith(".wav") and "argmax" not in x)
    return os.path.join(sd, c[-1])

def seq_dirs(root):
    out = []
    for method in sorted(os.listdir(root)):
        mdir = os.path.join(root, method)
        if not os.path.isdir(mdir):
            continue
        for N in sorted(os.listdir(mdir), key=int):
            for d in sorted(os.listdir(os.path.join(mdir, N))):
                sd = os.path.join(mdir, N, d)
                if os.path.exists(os.path.join(sd, "log.json")):
                    out.append((method, int(N), d, sd))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--Ns", default="8,32")
a = ap.parse_args()
Ns = {int(x) for x in a.Ns.split(",")}
rows = {r["key"]: r for r in json.load(open(a.out))["rows"]} if os.path.exists(a.out) else {}
p = initialize_predictor()
todo = [x for x in seq_dirs(a.samples) if (x[1] in Ns or x[0] == "base") and f"{x[0]}/{x[1]}/{x[2]}" not in rows]
print(f"{len(rows)} scored, {len(todo)} to do", flush=True)
for i, (method, N, d, sd) in enumerate(todo):
    log = json.load(open(os.path.join(sd, "log.json")))
    r = p.forward([{"path": returned_wav(sd)}])[0]
    rows[f"{method}/{N}/{d}"] = {"key": f"{method}/{N}/{d}", "method": method, "N": N, "prompt_id": log["prompt_id"], "base_seed": log["base_seed"], **r}
    if i % 50 == 0:
        print(i, f"{method}/{N}/{d}", r, flush=True)
        json.dump({"rows": list(rows.values())}, open(a.out, "w"))
json.dump({"rows": list(rows.values()), "package": "audiobox_aesthetics==0.0.4", "held_out": True}, open(a.out, "w"), indent=1)
print("wrote", a.out, len(rows), flush=True)
