"""Amendment S item 2: pick 10 genre-diverse prompts with numpy default_rng(0).

Rule as written: "choose with numpy default_rng(0) one or two prompts per genre
import os
from m1/prompts.json so that all 8 genres appear and the total is 10".
Procedure (fixed here, before it is run):
  1. genres in first-appearance order in prompts.json;
  2. rng.choice one prompt id per genre  -> 8 prompts;
  3. rng.choice 2 distinct genres (no replacement) -> each gets one more prompt,
     drawn with rng.choice from that genre's remaining ids -> 10 total.
"""
import json
import numpy as np

P = os.environ.get("LATTICESMC_ROOT", ".") + "/m1/prompts.json"
OUT = os.environ.get("LATTICESMC_ROOT", ".") + "/results/m2/selected_prompts.json"

prompts = json.load(open(P))
genres = []
for p in prompts:
    if p["genre"] not in genres:
        genres.append(p["genre"])

rng = np.random.default_rng(0)
by_genre = {g: [p["id"] for p in prompts if p["genre"] == g] for g in genres}
chosen = {}
for g in genres:
    chosen[g] = [int(rng.choice(by_genre[g]))]
extra = rng.choice(len(genres), size=2, replace=False)
for gi in extra:
    g = genres[int(gi)]
    rest = [i for i in by_genre[g] if i not in chosen[g]]
    chosen[g].append(int(rng.choice(rest)))

ids = sorted(i for v in chosen.values() for i in v)
assert len(ids) == 10 and len(set(ids)) == 10
assert set(chosen) == set(genres)

out = {
    "rng": "numpy.random.default_rng(0)",
    "genre_order": genres,
    "per_genre": chosen,
    "second_prompt_genres": [genres[int(i)] for i in extra],
    "prompt_ids": ids,
    "prompts": [p for p in prompts if p["id"] in ids],
}
json.dump(out, open(OUT, "w"), indent=1)
for p in out["prompts"]:
    print(p["id"], p["genre"], p["tempo_bpm"], p["text"][:60])
print("ids:", ids)
