
<p align="center">
<h1 align="center">  LatticeSMC: Where to Spend Inference-Time Compute in Chunked Sequence Generators</h1>


<div align="center">

[**Xuanchen Wang**](https://scholar.google.com/citations?user=H356FF8AAAAJ&hl=en)<sup></sup> · [**Heng Wang**](https://scholar.google.com.au/citations?user=jPj4ViQAAAAJ&hl=en&oi=ao)<sup></sup> ·   [**Weidong Cai**](https://scholar.google.com.au/citations?user=N8qTc2AAAAAJ&hl=en&oi=ao)<sup></sup>

School of Computer Science, The University of Sydney

</div>

Code for the paper *LatticeSMC: Where to Spend Inference-Time Compute in Chunked Sequence Generators* ([paper](https://arxiv.org/abs/2610.02774)). If you use it, please cite the paper (BibTeX at the end of this file, `CITATION.cff`).

This repository contains the sampling harness for both testbeds, the reward and held-out metric implementations, the data-preparation and analysis scripts that produce every table and figure of the paper (including the pruning-strength sweep, the horizon study at K = 6 and K = 8, the per-boundary-tilt runs at alpha = 0.01 and the K = 8 music qualification), the configs of every run and the stored analysis JSONs from which the paper's tables are reproduced without any sample. Checkpoints and generated samples are not included; the checkpoints are released at camera-ready (`MODELS.md`). License: Apache 2.0 (`LICENSE`).

## Layout

| path | content |
|---|---|
| `lattice_smc/` | dance testbed: chunked training (`train.py`), the methods behind one interface (`generate.py`, `methods/`: best-of-N, FK steering = dense schedule, chunk pruning with the `prune_M` grid key for the M sweep, LatticeSMC boundary schedule with `lattice_smc_notwist` = exact chunk potentials and `lattice_smc` = learned twist), rewards and held-out metrics (`rewards/`), twist rollouts / training / calibration (`twist/`), analyses (`analyze*.py`, `telescoping_check.py`, `within_set_check.py`), data preparation (`data/`: held-out split and prompt selection `make_prompts.py`, chunk pairs `build_pairs.py`, cache `build_cache.py`, prompt loader `pairs.py`, which reads `LATTICE_PROMPTS` or the grid key `prompts_file` for the K = 6 / 8 prompt lists) |
| `m3/` | music testbed on Stable Audio 3: the K = 4 lattice harness (`music_lattice.py`: best-of-N, chunk pruning with `--prune_M`, LatticeSMC with the prefix-score potential, FK steering), the K = 8 rolling-context harness (`music_lattice_k8.py`, used by the qualification gate), alpha rule (`alpha_motif.py`), argmax replay (`replay_argmax.py`), held-out scoring (`score_samples.py`, `score_tempo.py`, `score_aes.py`), analysis (`analyze_music.py`, `analyze_m3.py`) |
| `m1/`, `m2/` | CLAP / seam / tempo reward implementations (`m1/rewards.py`), the 40 music prompts (`m1/prompts.json`), the Stable Audio 3 wrapper with deterministic per-slot noise and pinned decoder seeds (`m2/sa3_harness_m2.py`), base-sample generation and the temperature rule (`m2/run_base.py`, `m2/alpha.py`), qualification (`m2/qual_report.py`, `m2/tempo_chunks.py`, `m2/select_prompts.py`) |
| `scripts/` | grid launchers and analyses: `analyze_u.py` (return rule, cost axes), `analyze_w.py` (consolidated tables, crossover figures), `analyze_x.py` (38M replicate), `paper_section4_figs.py` (Figures 4-7), `gpu_hours.py`; the extension study: `r1_phase1.py` (paired schedule intervals), `r1_phase2*.{sh,py}` (pruning-strength sweep and its analysis), `r1_make_prompts_k.py` (K = 6 / 8 prompt lists), `r1_phase3_dance.sh`, `r1_phase3b.sh`, `r1_phase3_dance_analysis.py` (horizon study, alpha = 0.01 rows), `r1_music_qual.sh`, `r1_music_qual_compare.py` (K = 8 music qualification gate), `r1_phase3_music.sh`, `r1_phase3_music_analysis.py` (the K = 8 music grid, gated on the qualification and therefore not run) |
| `configs/` | every training and grid config (YAML), including `r1_prune_*` (M sweep), `r1_k{6,8}_*` (horizon, both return rules) and `r1_k{6,8}_ba_a001*` (alpha = 0.01) |
| `data/` | the 40 dance prompts (`prompts.json`, 20 s segments of held-out AIST++ sequences, 4 seeds each), the K = 6 (40 prompts) and K = 8 (16 prompts) lists built from the same source sequences (`prompts_K6.json`, `prompts_K8.json`), and the held-out sequence lists (`heldout_sequences.json`, `heldout_sequences_extended.json`) |
| `edge/` | the vendored EDGE code (MIT, upstream pin 17c3428) that the dance model and rewards import; its `data/` symlinks are not included |
| `tests/` | blocking checks: loss mask, determinism, boundary continuity, NFE accounting, method identities |
| `analysis_json/` | the stored analysis outputs the paper's tables are read from (`analysis_json/INDEX.md`); the extension study is under `analysis_json/r1/` (`schedule_pairs`, `pruning_sweep`, `dance_k6`, `dance_k8`, `qual_k8/`, held-out scores of the music pruning runs) |
| `reproduce_tables.py`, `paper_tables.tex` | Tables 1, 3, 7, 8, 9 and 12 from `analysis_json/`, as LaTeX bodies and plain text, checked value by value against the manuscript excerpts in `paper_tables.tex` |

## Environments

- Dance: `requirements.txt` (Python 3.10, torch 2.14.0, pytorch3d 0.7.9 built from source, CUDA). Set `LATTICESMC_ROOT` to this directory. AIST++ in EDGE's format is needed under `edge/data/` (EDGE's `download` instructions); `python -m lattice_smc.data.make_prompts --K 4 --min_prompts 40` then `python -m lattice_smc.data.build_pairs --heldout data/heldout_sequences_extended.json` rebuild the held-out split, the prompts and the chunk pairs; `python scripts/r1_make_prompts_k.py` builds the K = 6 / 8 prompt caches from the same source sequences.
- Music: the harness runs inside the Stable Audio 3 repository's own environment (`requirements-music.txt`, Python 3.10, torch 2.7.1+cu126) with `laion_clap==1.1.7` and the LAION checkpoint `music_audioset_epoch_15_esc_90.14.pt`; set `STABLE_AUDIO_ROOT` to that repository (default `third_party/stable-audio-3`). The tempo tracker (`requirements-tempo.txt`, beat_this) and audiobox-aesthetics (`requirements-aesthetics.txt`) each run in their own environment.

## (1) Reproduce the tables from the stored analysis JSONs (seconds, no GPU)

```
python reproduce_tables.py                  # Tables 1, 3, 7, 8, 9, 12; exit 0 only if every value matches the manuscript within rounding
python reproduce_tables.py --tables 7,8,9   # a subset
```
Every number of these tables is read from `analysis_json/`; no sample is needed. The remaining appendix tables and the figures come from the same files: `scripts/analyze_w.py` (consolidated tables, crossover figure), `scripts/paper_section4_figs.py` (Figures 4-7), `scripts/analyze_x.py` (scale table), `scripts/r1_phase2_analysis.py` and `scripts/r1_phase3_dance_analysis.py` (the extension tables, which also need the per-sequence metrics caches produced by `lattice_smc.analyze_phase2 --metrics_only` from the samples); with `LATTICESMC_ROOT` set and `results/` pointing at `analysis_json/` (`ln -s analysis_json results`).

## (2) One steered sequence on each testbed

Dance, K = 4 (LatticeSMC boundary schedule, draw return, N = 32, alpha = 0.02, first prompt, seed 2000; about 9 s on an A6000 for the 4M model, 30 s for the 38M):

```
export LATTICESMC_ROOT=$PWD
python -m lattice_smc.generate --ckpt runs/chunk_dispE/ckpt_adopted.pt --method lattice_smc_notwist --N 32 --prompts 1 --seeds 2000 --alpha 0.02 --reward ba --out_root samples_demo
```
(`--reward rep` for repetition; `--method fk_noise` is the dense schedule, `greedy_chunk` chunk pruning, `bon_argmax` / `bon_is` best-of-N. The search rule is a grid key: run a YAML with `return_rule: argmax`, as `configs/phase2_grid_a002_argmax.yaml`; a grid's `prompts` / `seeds` keys override the command line, so copy the config and set `prompts: 1`, `seeds: "2000"` for a single sequence.)

Dance, K = 6 (one 30 s sequence of the first K = 6 prompt; `LATTICE_PROMPTS` selects the prompt list, whose caches `scripts/r1_make_prompts_k.py` builds):

```
LATTICE_PROMPTS=data/prompts_K6.json python -m lattice_smc.generate --ckpt runs/chunk_dispE/ckpt_adopted.pt --method lattice_smc_notwist --N 32 --prompts 1 --seeds 2000 --alpha 0.02 --reward ba --out_root samples_demo_k6
```
Dance, K = 8 (one 40 s sequence, the 16-prompt list; `--alpha 0.01` is the per-boundary tilt of Table 3):

```
LATTICE_PROMPTS=data/prompts_K8.json python -m lattice_smc.generate --ckpt runs/chunk_dispE/ckpt_adopted.pt --method lattice_smc_notwist --N 32 --prompts 1 --seeds 2000 --alpha 0.01 --reward ba --out_root samples_demo_k8
```
(The full K = 6 / 8 grids with argmax return are the configs `configs/r1_k{6,8}_{ba,rep}{,_argmax}.yaml` and `configs/r1_k{6,8}_ba_a001{,_argmax}.yaml`, launched by `scripts/r1_phase3_dance.sh` and `scripts/r1_phase3b.sh`.)

Music, K = 4 (LatticeSMC with the prefix-score potential, beta = 1, N = 32, prompt 0, seed 2000; about 11 minutes, of which 128 decoder and CLAP calls):

```
cd $STABLE_AUDIO_ROOT && HF_HOME=$STABLE_AUDIO_ROOT/.hf-cache python $LATTICESMC_ROOT/m3/music_lattice.py --runs lsp1 --Ns 32 --seeds 2000 --prompts 0 --alpha_file $LATTICESMC_ROOT/analysis_json/m2/alpha.json --out_root $LATTICESMC_ROOT/samples_demo_music
```
(`--reward motif --alpha_file analysis_json/m4/alpha.json` for motif recurrence; `--runs bon,greedy_chunk,lsp1,fk_noise` for the other methods.)

## (3) Pruning-strength sweep at one M (Table 7)

Chunk pruning keeps the top M particles at every boundary; the paper's grids use M = N/4 and Table 7 sweeps M in {1, N/4, N/2} at N = 32 on all four rewards with the same conditions, seeds and temperatures. One M on each testbed:

```
python -m lattice_smc.generate --grid configs/r1_prune_ba_M1.yaml            # dance, beat alignment, M = 1 (r1_prune_{ba,rep}_M{1,16}.yaml; M = 16 is N/2)
python -m lattice_smc.analyze_phase2 --samples samples_r1_prune_ba_M1 --tag r1_prune_ba_M1 --metrics_only
cd $STABLE_AUDIO_ROOT && HF_HOME=$STABLE_AUDIO_ROOT/.hf-cache python $LATTICESMC_ROOT/m3/music_lattice.py --reward clap --alpha_file $LATTICESMC_ROOT/analysis_json/m2/alpha.json \
    --runs greedy_chunk --Ns 32 --seeds 2000,2001 --prune_M 1 --out_root $LATTICESMC_ROOT/results/r1/samples_prune_clap_M1     # --prune_M N/2 for M = 16
```
then `m3/score_samples.py`, `m3/score_tempo.py`, `m3/score_aes.py` on the music root and `python scripts/r1_phase2_analysis.py` (the full sweep is `scripts/r1_phase2.sh` + `scripts/r1_phase2_post.sh`; the music runs as launched by `scripts/r1_phase2_music_rerun.sh`).

## (4) K = 8 music qualification check (Appendix C.6)

The music generator's maximum clip length allows K = 8 (80 s) only with a rolling 40 s context (`m3/music_lattice_k8.py`). Before any steered run, 20 base clips at K = 8 (prompts 4 8 9 12 16 21 25 30 33 35, seeds 2000 2001) are compared with the paper's K = 4 base clips of the same prompts under a gate fixed in advance (seam spectral-flux ratio and whole-clip tempo adherence within 10 percent of K = 4):

```
scripts/r1_music_qual.sh <gpu>          # generates results/r1/qual_k8/, scores seams and tempo, then:
python scripts/r1_music_qual_compare.py  # -> results/r1/qual_k8/qualification.json, prints QUAL_PASS / QUAL_FAIL
```
The stored outcome (`analysis_json/r1/qual_k8/qualification.json`): seams pass (1.237 against 1.261), tempo adherence fails (0.85 against 0.95, -10.5 percent), so the K = 8 music grid in `scripts/r1_phase3_music.sh` was not run; the horizon results rest on dance.

## (5) Re-run a full N = 32 grid

Dance, beat alignment at alpha = 0.02 (40 prompts x 4 seeds x N in {1, 2, 4, 8, 16, 32} x 7 methods = 5,920 sequences; about 5 GPU-hours on one A6000, sharded over two GPUs by `--shard i/2`):

```
python -m lattice_smc.generate --grid configs/phase2_grid_a002.yaml --shard 0/2   # and 1/2 on the other GPU
python -m lattice_smc.analyze_phase2 --samples samples_p2_a002 --tag phase2b_a002 --reward ba
```
Repetition: `configs/phase3_rep_grid_a002.yaml` (about 7 GPU-hours, twist rollouts and training excluded); the 38M replicate: `configs/phaseX_{ba,rep}_grid_a002.yaml` (15 + 16 GPU-hours); argmax-return regenerations: the `*_argmax.yaml` configs (about 1 GPU-hour per reward for the 4M model).

Music, prompt adherence (40 prompts x 2 seeds x 6 budgets x 6 runs = 3,440 clips; about 77 GPU-hours exclusive, run as 3 shards per GPU with `scripts/m3_grid.sh <gpu> <i/6>`):

```
scripts/m3_grid.sh 0 0/6   # ... 5/6; then
python m3/score_samples.py --samples results/m3/samples --out results/m3/scores.json
python m3/analyze_music.py --samples results/m3/samples --out results/m3 --reward clap --tag m3
```
Motif recurrence: `scripts/m4_grid.sh` and `--reward motif` (about 55 GPU-hours).

Extension study (GPU-hours as summed job walls on shared A6000s): pruning-strength sweep, dance 3.0 and music 32.7; horizon study K = 6 / 8 on dance 49.4; alpha = 0.01 runs 7.4; the K = 8 music qualification (20 base clips and their scoring) under 0.5. The per-phase accounting of every run in the paper is in `analysis_json/GPU_HOURS.md` (about 250 GPU-hours for the main grids on two 48 GB GPUs, plus the extension above).

## Determinism and audits

Every sequence is determined by (checkpoint, prompt, seed, method, N); `tests/` (`python -m pytest tests -q`) checks the loss mask, bitwise determinism across runs and GPUs, boundary continuity, the NFE count against N x K x S x (guidance multiplicity), and that every method at N = 1 reproduces the base sampler bitwise. The telescoping identities are checked numerically on stored lineages by `lattice_smc/telescoping_check.py` and inside `m3/music_lattice.py` (logged per sequence). The K = 8 rolling-context harness reproduces the K = 4 base latent bitwise over the first 40 s (checked in its smoke test).

## Citation

```bibtex
@article{wang2026latticesmc,
  title   = {LatticeSMC: Where to Spend Inference-Time Compute in Chunked Sequence Generators},
  author  = {Wang, Xuanchen and Wang, Heng and Cai, Weidong},
  journal = {https://arxiv.org/abs/2610.02774},
  year    = {2026}
}
```

## License

Apache License 2.0 (`LICENSE`). The vendored `edge/` directory keeps its upstream MIT license (`edge/LICENSE`).
