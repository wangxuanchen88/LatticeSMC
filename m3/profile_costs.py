"""Profile: decode vs CLAP time per particle, DiT time and peak memory at N = 32 (bon), for the budget projection."""
import os
import json, sys, time
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m3")
import numpy as np, torch
import music_lattice as ML
prompts = json.load(open(ML.PROMPTS)); alpha = float(json.load(open(ML.ALPHA_FILE))["alpha"])
H = ML.MusicLattice(); prec = prompts[0]
torch.cuda.reset_peak_memory_stats()
t0 = time.time(); lat = None; prefix = None
for k in range(4):
    lat = H.sample_chunk(prec["text"], prec["id"], 2000, k, 32, prefix); prefix = lat[..., :H.n_prefix_frames((k + 1) * 10.0)].clone()
torch.cuda.synchronize(); print("DiT N=32, 4 chunks:", round(time.time() - t0, 1), "s; peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for p in range(8): a = H.decode(lat[p:p + 1], prec["id"], 2000, 40.0)
torch.cuda.synchronize(); print("decode x8 (40 s):", round((time.time() - t0) / 8, 3), "s each; peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
t0 = time.time()
for p in range(8): s, e = H.prefix_clap(a, prec["text"], 4)
print("clap x8 (4 windows incl. resample):", round((time.time() - t0) / 8, 3), "s each", flush=True)
import librosa
t0 = time.time()
for p in range(8): m48 = librosa.resample(a.mean(0), orig_sr=H.sr, target_sr=48000)
print("librosa resample x8:", round((time.time() - t0) / 8, 3), "s each", flush=True)
t0 = time.time()
for p in range(8): E = H.clap.window_embeds(m48, 4)
print("clap embed x8:", round((time.time() - t0) / 8, 3), "s each", flush=True)
t0 = time.time(); lat1 = H.sample_chunk(prec["text"], prec["id"], 2000, 3, 1, prefix[:1]); torch.cuda.synchronize(); print("DiT N=1 chunk 4:", round(time.time() - t0, 2), flush=True)
t0 = time.time(); lat8 = H.sample_chunk(prec["text"], prec["id"], 2000, 3, 8, prefix[:8]); torch.cuda.synchronize(); print("DiT N=8 chunk 4:", round(time.time() - t0, 2), flush=True)
t0 = time.time(); lat32 = H.sample_chunk(prec["text"], prec["id"], 2000, 3, 32, prefix); torch.cuda.synchronize(); print("DiT N=32 chunk 4:", round(time.time() - t0, 2), flush=True)
print("particle 0 same in N=1/8/32 batches:", torch.equal(lat1[0], lat8[0]), torch.equal(lat8[0], lat32[0]), float((lat1[0].float() - lat32[0].float()).abs().max()))
