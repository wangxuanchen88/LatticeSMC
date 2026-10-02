"""M2 base grid: SA3 medium + replacement inpainting, RAW decode, pinned decoder seed.

Resumable: a sequence directory that already has log.json is skipped.
"""
import os
import argparse, json, os, sys, time
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m2")
import numpy as np, torch
from sa3_harness_m2 import SA3Harness, derive_decoder_seed

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default=os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/b2/samples")
ap.add_argument("--prompt_ids", type=int, nargs="*", default=None,
                help="explicit prompt ids; default = all 40")
ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002, 2003])
ap.add_argument("--K", type=int, default=4)
a = ap.parse_args()

prompts = json.load(open(os.environ.get("LATTICESMC_ROOT", ".") + "/m1/prompts.json"))
if a.prompt_ids:
    keep = set(a.prompt_ids)
    prompts = [p for p in prompts if p["id"] in keep]
os.makedirs(a.outdir, exist_ok=True)
h = SA3Harness()
json.dump({"load_s": h.load_s, "model": h.model_name, "steps": h.steps,
           "cfg_scale": h.cfg_scale, "sampler": h.sampler_type, "sr": h.sr,
           "downsampling_ratio": h.ds, "objective": h.objective,
           "duration_padding_sec": h.duration_padding_sec,
           "decode": "raw (no waveform splice); decoder seed pinned per (prompt, seed)"},
          open(os.path.join(a.outdir, "_harness.json"), "w"), indent=1)
t_all = time.time()
n = 0
for p in prompts:
    for s in a.seeds:
        d = os.path.join(a.outdir, f"p{p['id']:02d}_s{s}")
        if os.path.exists(os.path.join(d, "log.json")):
            print("skip", d, flush=True); continue
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        dseed = derive_decoder_seed(p["id"], s)
        st = h.start(p["text"], p["id"], s, d, "seq")
        np.save(os.path.join(d, "seq_chunk0_latents.npy"), st.latent.cpu().numpy())
        for k in range(1, a.K):
            st = h.extend(st, p["text"], p["id"], s, d, "seq")
            np.save(os.path.join(d, f"seq_chunk{k}_latents.npy"), st.latent.cpu().numpy())
        # latent prefix check (the particle state; Amendment S item 1)
        lat_ok = []
        for k in range(1, a.K):
            L0 = np.load(os.path.join(d, f"seq_chunk{k-1}_latents.npy"))
            L1 = np.load(os.path.join(d, f"seq_chunk{k}_latents.npy"))
            P = h.n_prefix_frames(k * h.chunk_sec)
            lat_ok.append({"k": k, "P": int(P),
                           "latent_prefix_bitexact": bool(np.array_equal(L0[..., :P], L1[..., :P])),
                           "latent_prefix_max_abs_diff": float(np.abs(
                               L0[..., :P].astype(np.float64) - L1[..., :P].astype(np.float64)).max())})
        log = {"prompt_id": p["id"], "genre": p["genre"], "prompt": p["text"],
               "bpm": p["tempo_bpm"], "base_seed": s, "K": a.K,
               "decoder_seed": int(dseed),
               "wall_s": time.time() - t0,
               "nfe_total": sum(c["nfe"] for c in st.per_chunk),
               "dit_calls_total": sum(c["dit_calls"] for c in st.per_chunk),
               "decode_calls_total": sum(c["decode_calls"] for c in st.per_chunk),
               "per_chunk": st.per_chunk, "latent_prefix": lat_ok}
        json.dump(log, open(os.path.join(d, "log.json"), "w"), indent=1)
        n += 1
        print(f"done {d} {log['wall_s']:.2f}s nfe={log['nfe_total']} "
              f"dec={log['decode_calls_total']} latok={all(x['latent_prefix_bitexact'] for x in lat_ok)}",
              flush=True)
print(f"TOTAL {time.time() - t_all:.1f}s new={n}", flush=True)
