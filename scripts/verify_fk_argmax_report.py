"""Dagger-cell report: full R_BA of the regenerated fk_noise argmax sequences at N = 32 (4M and 38M), bootstrap over prompts,
against the logged additive argmax values. Writes results/verify_fkargmax_summary.json."""
import os
import glob, json, os
import numpy as np
ROOT = os.environ.get("LATTICESMC_ROOT", ".")
rng = np.random.default_rng(0)
def boot(by_prompt):
    v = np.array(list(by_prompt.values())); idx = rng.integers(0, len(v), size=(1000, len(v))); b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
out = {}
for model, root, tag, stored_root in [("4M", "samples_p2_a002_fkargmax", "verify_fkargmax_p2_a002", "samples_p2_a002"), ("38M", "samples_L_ba_a002_fkargmax", "verify_fkargmax_L_ba_a002", "samples_L_ba_a002")]:
    m = json.load(open(f"{ROOT}/results/{tag}_metrics.json"))
    rows = {os.path.basename(k): r for k, r in m.items() if r["method"] == "fk_noise" and r["N"] == 32}
    full, add, logged, gap, ok = {}, {}, {}, {}, 0
    for b, r in rows.items():
        z = np.load(f"{ROOT}/{stored_root}/fk_noise/32/{b}"); l = json.loads(str(z["log"]))
        zr = np.load(f"{ROOT}/{root}/fk_noise/32/{b}"); lr = json.loads(str(zr["log"]))
        ok += int(np.array_equal(np.asarray(l["rewards"]), np.asarray(lr["rewards"])) and np.array_equal(np.asarray(l["weights"]), np.asarray(lr["weights"])) and lr["chosen"] == int(np.argmax(l["rewards"])) and int(zr["nfe"]) == 12800)
        p = r["prompt_id"]; full.setdefault(p, []).append(r["ba_full"]); add.setdefault(p, []).append(r["ba_additive"]); logged.setdefault(p, []).append(max(l["rewards"])); gap.setdefault(p, []).append(r["ba_full"] - max(l["rewards"]))
    pm = lambda d: {p: float(np.mean(v)) for p, v in d.items()}
    mf, lo, hi = boot(pm(full)); ma = float(np.mean([x for v in add.values() for x in v])); ml = float(np.mean([x for v in logged.values() for x in v])); g, glo, ghi = boot(pm(gap))
    out[model] = {"n_sequences": len(rows), "n_prompts": len(full), "replay_bitwise_ok": ok, "full_mean": mf, "full_ci": [lo, hi], "additive_mean": ma, "logged_argmax_mean": ml,
                  "full_minus_logged_mean": g, "full_minus_logged_ci": [glo, ghi], "max_abs_additive_minus_logged": float(max(abs(a - b) for v1, v2 in zip(add.values(), logged.values()) for a, b in zip(v1, v2))),
                  "frac_sequences_within_0.01": float(np.mean([abs(x) <= 0.01 for v in gap.values() for x in v])), "pfc_mean": float(np.mean([r["pfc"] for r in rows.values()])), "w1_mean": float(np.mean([r["realism_w1"] for r in rows.values()]))}
    print(model, json.dumps(out[model]))
json.dump(out, open(f"{ROOT}/results/verify_fkargmax_summary.json", "w"), indent=1)
