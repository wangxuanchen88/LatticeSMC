"""Is the CLAP score of a fixed wav deterministic within and across processes? (root-of-trust diagnostic)"""
import os
import sys, json, numpy as np, torch
sys.path.insert(0, os.environ.get("LATTICESMC_ROOT", ".") + "/m1"); import rewards as R
det = len(sys.argv) > 1 and sys.argv[1] == "det"
if det:  # the flags Stable Audio 3 sets at load time (model.py:25-28), under which the in-run CLAP scores were computed
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
clap = R.ClapScorer(device="cuda")
wav = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m3/samples/base/1/p00_s2000/seq_chunk03.wav"
text = json.load(open(os.environ.get("LATTICESMC_ROOT", ".") + "/m1/prompts.json"))[0]["text"]
m48 = R.load_mono48k(wav); t = clap.text_embed([text])[0]
vals = []
for i in range(3):
    E = clap.window_embeds(m48, 4); mv = E.mean(0); vals.append(float((mv / np.linalg.norm(mv)) @ t))
print("cudnn_det", det, "values", vals, "logged", json.load(open(wav.replace("seq_chunk03.wav", "log.json")))["final_reward"], flush=True)
