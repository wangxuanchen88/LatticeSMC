# GPU-hours per phase (from logs; `scripts/gpu_hours.py`)

Each job ran on one GPU; the seconds are the job's own logged wall (`done: ... s`), summed per phase. Jobs that shared a GPU (Phase 3d, Phases M3 and M4 with 3 shards per GPU, the Phase U and W music replays with 2 shards) are counted by their own walls, so those phases are upper bounds on exclusive GPU time; the exclusive Phase M3 and M4 figures are derived below from the shard timestamps. Phases U, M4, W and X were added by Amendment W (2026-09-16). Delegated music phases are taken from the agents' reports.

| phase | GPU-hours (sum of job walls) | source |
|---|---|---|
| Phase 1 (chunk_s101 training, base generation x3 roots) | 0.70 | chunk_s101.log, gen_samples_rerun_s0.log, gen_samples_rerun_s1.log, gen_samples_s0.log, gen_samples_s1.log, gen_samples_swap_s0.log, gen_samples_swap_s1.log |
| Phase 1b (12k and 30k training, 3 roots each) | 2.24 | chunk_s101_12k.log, chunk_s101_30k.log, gen_samples_12k_gpu0.log, gen_samples_12k_rerun_gpu0.log, gen_samples_12k_swap_gpu1.log, gen_samples_30k_rerun_s0.log, gen_samples_30k_rerun_s1.log, gen_samples_30k_s0.log, gen_samples_30k_s1.log, gen_samples_30k_swap_s0.log, gen_samples_30k_swap_s1.log |
| Phase 1c (candidates E, F training, 3 roots each) | 1.97 | chunk_dispE.log, chunk_seamF.log, gen_samples_E_rerun_s0.log, gen_samples_E_rerun_s1.log, gen_samples_E_s0.log, gen_samples_E_s1.log, gen_samples_E_swap_s0.log, gen_samples_E_swap_s1.log, gen_samples_F_rerun_s0.log, gen_samples_F_rerun_s1.log, gen_samples_F_s0.log, gen_samples_F_s1.log, gen_samples_F_swap_s0.log, gen_samples_F_swap_s1.log |
| Phase 2a (alpha pilot, R_BA twist rollouts) | 1.41 | alpha_pilot.log, twist_collect_fresh_s0.log, twist_collect_fresh_s1.log, twist_collect_train_s0.log |
| Phase 2 (grid alpha 0.2, both GPUs) | 5.08 | grid_s0.log, grid_s1.log |
| Phase 2 background (candidate H training + 3 roots) | 0.97 | chunk_dispH.log, gen_samples_H_rerun_s0.log, gen_samples_H_rerun_s1.log, gen_samples_H_s0.log, gen_samples_H_s1.log, gen_samples_H_swap_s0.log, gen_samples_H_swap_s1.log |
| Phase 2b (grids alpha 0.02 and 0.05) | 10.08 | grid_a002.log, grid_a005.log |
| Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05) | 13.37 | grid_rep_a002.log, grid_rep_a005.log, rep_collect_fresh.log, rep_collect_train.log, rep_collect_train_rev.log |
| Phase 3b (tempered twist grids, oracle grid) | 5.13 | grid_rep_oracle.log, grid_rep_tw010.log, grid_rep_tw025.log, grid_rep_tw050.log |
| Phase 3c (ensemble grids, within-set trace) | 3.26 | grid_rep_ens_mean.log, grid_rep_ens_shrunk.log, phase3c_within_set.log |
| Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids) | 7.51 | collect_sets.log, phase3d_oracle_sets.log, phase3d_rollout_grids.log, phase3d_wstwist_grid.log |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | 222.64 | m3_grid_0.log, m3_grid_1.log, m3_grid_2.log, m3_grid_3.log, m3_grid_4.log, m3_grid_5.log |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | 5.13 | argmax_regen_gpu0.log, argmax_regen_gpu1.log, u_argmax_scores.log, u_argmax_tempo.log, u_replay_0.log, u_replay_1.log |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | 158.55 | m4_grid_0.log, m4_grid_1.log, m4_grid_2.log, m4_grid_3.log, m4_grid_4.log, m4_grid_5.log, m4_scores.log, m4_tempo.log |
| Phase W (M4 argmax replays, scoring) | 3.96 | w_argmax_scores.log, w_argmax_tempo.log, w_replay_0.log, w_replay_1.log |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | 44.35 | L_rep_collect_fresh_s0.log, L_rep_collect_fresh_s1.log, L_rep_collect_train_s0.log, L_rep_collect_train_s1.log, chunk_dispE_L.log, gen_samples_L_rerun_s0.log, gen_samples_L_rerun_s1.log, gen_samples_L_s0.log, gen_samples_L_s1.log, gen_samples_L_swap_s0.log, gen_samples_L_swap_s1.log, grid_L_ba_a002_argmax.log, grid_L_ba_a002_s0.log, grid_L_ba_a002_s1.log, grid_L_rep_a002_argmax.log, grid_L_rep_a002_s0.log, grid_L_rep_a002_s1.log, probe_L.log |
| Phase M0 (music feasibility, GPU 1) | 0.83 | RESULTS.md Phase 3 / M0: 'about 50 min wall' |
| Phase M1 (ACE-Step harness, GPU 1) | 3.00 | RESULTS.md Phase 3b / M1: 'Wall about 3 h (of which the variance study 2.41 h)' |
| Phase M1b (B1 / B2 harnesses, GPU 1) | 2.60 | RESULTS.md Phase 3c / M1b: 'about 2.6 h wall' |
| Phase M2 (B2 qualification, base samples, GPU 1) | 1.25 | RESULTS.md Phase 3d / M2: 'Wall about 1 h 15 min' |

**Total: 494.0 GPU-hours as the sum of job walls.** The Phase M3 shards ran 3 per GPU, so their summed walls (222.6 h) overstate exclusive GPU time: the grid occupied both GPUs from the relaunch to the last shard's end, 38.6 h of wall, i.e. **77.1 GPU-hours exclusive**, plus about 2 GPU-hours for the first (prompt-major) attempt whose 200 finished sequences were reused. The Phase M4 shards likewise ran 3 per GPU (summed walls 158.5 h); the grid occupied both GPUs for 27.4 h of wall, i.e. **54.7 GPU-hours exclusive**. **Exclusive total: about 247 GPU-hours.** Not included: CPU-only scoring (beat_this, aesthetics), analysis scripts, unit tests, smoke tests and the interactive diagnostics, each minutes.

## Per-job walls

| phase | log | seconds | note |
|---|---|---|---|
| Phase 1 (chunk_s101 training, base generation x3 roots) | `chunk_s101.log` | 2023 |  |
| Phase 1 (chunk_s101 training, base generation x3 roots) | `gen_samples_s0.log` | 81 |  |
| Phase 1 (chunk_s101 training, base generation x3 roots) | `gen_samples_s1.log` | 81 |  |
| Phase 1 (chunk_s101 training, base generation x3 roots) | `gen_samples_rerun_s0.log` | 82 |  |
| Phase 1 (chunk_s101 training, base generation x3 roots) | `gen_samples_rerun_s1.log` | 81 |  |
| Phase 1 (chunk_s101 training, base generation x3 roots) | `gen_samples_swap_s0.log` | 81 |  |
| Phase 1 (chunk_s101 training, base generation x3 roots) | `gen_samples_swap_s1.log` | 81 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `chunk_s101_12k.log` | 2051 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `chunk_s101_30k.log` | 5023 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_12k_gpu0.log` | 165 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_12k_rerun_gpu0.log` | 164 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_12k_swap_gpu1.log` | 162 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_30k_rerun_s0.log` | 83 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_30k_rerun_s1.log` | 83 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_30k_s0.log` | 84 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_30k_s1.log` | 83 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_30k_swap_s0.log` | 84 |  |
| Phase 1b (12k and 30k training, 3 roots each) | `gen_samples_30k_swap_s1.log` | 83 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `chunk_dispE.log` | 3086 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `chunk_seamF.log` | 3023 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_E_rerun_s0.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_E_rerun_s1.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_E_s0.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_E_s1.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_E_swap_s0.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_E_swap_s1.log` | 80 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_F_rerun_s0.log` | 80 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_F_rerun_s1.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_F_s0.log` | 80 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_F_s1.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_F_swap_s0.log` | 81 |  |
| Phase 1c (candidates E, F training, 3 roots each) | `gen_samples_F_swap_s1.log` | 80 |  |
| Phase 2a (alpha pilot, R_BA twist rollouts) | `twist_collect_fresh_s0.log` | 764 |  |
| Phase 2a (alpha pilot, R_BA twist rollouts) | `twist_collect_fresh_s1.log` | 749 |  |
| Phase 2a (alpha pilot, R_BA twist rollouts) | `twist_collect_train_s0.log` | 3355 |  |
| Phase 2a (alpha pilot, R_BA twist rollouts) | `alpha_pilot.log` | 204 | pilot wall from RESULTS.md Phase 2a (204 s) |
| Phase 2 (grid alpha 0.2, both GPUs) | `grid_s0.log` | 9179 |  |
| Phase 2 (grid alpha 0.2, both GPUs) | `grid_s1.log` | 9099 |  |
| Phase 2 background (candidate H training + 3 roots) | `chunk_dispH.log` | 3013 |  |
| Phase 2 background (candidate H training + 3 roots) | `gen_samples_H_rerun_s0.log` | 81 |  |
| Phase 2 background (candidate H training + 3 roots) | `gen_samples_H_rerun_s1.log` | 80 |  |
| Phase 2 background (candidate H training + 3 roots) | `gen_samples_H_s0.log` | 81 |  |
| Phase 2 background (candidate H training + 3 roots) | `gen_samples_H_s1.log` | 81 |  |
| Phase 2 background (candidate H training + 3 roots) | `gen_samples_H_swap_s0.log` | 81 |  |
| Phase 2 background (candidate H training + 3 roots) | `gen_samples_H_swap_s1.log` | 81 |  |
| Phase 2b (grids alpha 0.02 and 0.05) | `grid_a002.log` | 18193 |  |
| Phase 2b (grids alpha 0.02 and 0.05) | `grid_a005.log` | 18110 |  |
| Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05) | `rep_collect_fresh.log` | 1306 |  |
| Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05) | `rep_collect_train.log` | 5560 |  |
| Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05) | `rep_collect_train_rev.log` | 4638 |  |
| Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05) | `grid_rep_a002.log` | 18450 |  |
| Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05) | `grid_rep_a005.log` | 18165 |  |
| Phase 3b (tempered twist grids, oracle grid) | `grid_rep_tw010.log` | 3001 |  |
| Phase 3b (tempered twist grids, oracle grid) | `grid_rep_tw025.log` | 3017 |  |
| Phase 3b (tempered twist grids, oracle grid) | `grid_rep_tw050.log` | 3036 |  |
| Phase 3b (tempered twist grids, oracle grid) | `grid_rep_oracle.log` | 9431 |  |
| Phase 3c (ensemble grids, within-set trace) | `grid_rep_ens_mean.log` | 5085 |  |
| Phase 3c (ensemble grids, within-set trace) | `grid_rep_ens_shrunk.log` | 3046 |  |
| Phase 3c (ensemble grids, within-set trace) | `phase3c_within_set.log` | 3600 | trace re-run of 160 sequences at N = 32 with 5 oracle sets; wall not logged, estimated 1 h from log timestamps |
| Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids) | `collect_sets.log` | 9828 |  |
| Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids) | `phase3d_wstwist_grid.log` | 3009 |  |
| Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids) | `phase3d_rollout_grids.log` | 11792 |  |
| Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids) | `phase3d_oracle_sets.log` | 2400 | oracle-set save (5 prompts, N = 32, M = 16), about 40 min from log timestamps |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | `m3_grid_0.log` | 138743 | two passes per shard (seed 2000, then seed 2001), 3 shards per GPU |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | `m3_grid_1.log` | 137898 | two passes per shard (seed 2000, then seed 2001), 3 shards per GPU |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | `m3_grid_2.log` | 138740 | two passes per shard (seed 2000, then seed 2001), 3 shards per GPU |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | `m3_grid_3.log` | 137903 | two passes per shard (seed 2000, then seed 2001), 3 shards per GPU |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | `m3_grid_4.log` | 124492 | two passes per shard (seed 2000, then seed 2001), 3 shards per GPU |
| Phase M3 (music grid, 6 shards on 2 GPUs; scoring) | `m3_grid_5.log` | 123726 | two passes per shard (seed 2000, then seed 2001), 3 shards per GPU |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | `argmax_regen_gpu0.log` | 2827 | 5 dance roots, both GPUs |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | `argmax_regen_gpu1.log` | 1661 | 5 dance roots, both GPUs |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | `u_replay_0.log` | 6362 | sum of the per-sequence replay walls (2 shards, one per GPU) |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | `u_replay_1.log` | 7473 | sum of the per-sequence replay walls (2 shards, one per GPU) |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | `u_argmax_scores.log` | 86 | GPU scoring, wall from the log timestamps |
| Phase U (argmax regenerations: dance roots, music replays, scoring) | `u_argmax_tempo.log` | 52 | GPU scoring, wall from the log timestamps |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_grid_0.log` | 98366 | two passes per shard, 3 shards per GPU |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_grid_1.log` | 98118 | two passes per shard, 3 shards per GPU |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_grid_2.log` | 98381 | two passes per shard, 3 shards per GPU |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_grid_3.log` | 98110 | two passes per shard, 3 shards per GPU |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_grid_4.log` | 88570 | two passes per shard, 3 shards per GPU |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_grid_5.log` | 88261 | two passes per shard, 3 shards per GPU |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_scores.log` | 762 | GPU scoring, wall from the log timestamps |
| Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring) | `m4_tempo.log` | 199 | GPU scoring, wall from the log timestamps |
| Phase W (M4 argmax replays, scoring) | `w_replay_0.log` | 7065 | sum of the per-sequence replay walls (2 shards on GPU 1) |
| Phase W (M4 argmax replays, scoring) | `w_replay_1.log` | 7067 | sum of the per-sequence replay walls (2 shards on GPU 1) |
| Phase W (M4 argmax replays, scoring) | `w_argmax_scores.log` | 70 | GPU scoring, wall from the log timestamps |
| Phase W (M4 argmax replays, scoring) | `w_argmax_tempo.log` | 52 | GPU scoring, wall from the log timestamps |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `probe_L.log` | 165 | throughput probe, 300 steps, wall from its step lines (crashed at the final save, run deleted) |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `chunk_dispE_L.log` | 18019 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `gen_samples_L_rerun_s0.log` | 131 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `gen_samples_L_rerun_s1.log` | 129 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `gen_samples_L_s0.log` | 126 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `gen_samples_L_s1.log` | 126 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `gen_samples_L_swap_s0.log` | 127 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `gen_samples_L_swap_s1.log` | 126 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `L_rep_collect_fresh_s0.log` | 1170 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `L_rep_collect_fresh_s1.log` | 1140 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `L_rep_collect_train_s0.log` | 9187 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `L_rep_collect_train_s1.log` | 9075 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `grid_L_ba_a002_argmax.log` | 6043 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `grid_L_ba_a002_s0.log` | 24959 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `grid_L_ba_a002_s1.log` | 24506 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `grid_L_rep_a002_argmax.log` | 5940 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `grid_L_rep_a002_s0.log` | 29422 |  |
| Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids) | `grid_L_rep_a002_s1.log` | 29269 |  |


## Extension study (pruning-strength sweep, horizon, per-boundary tilt, K = 8 music qualification)

Summed job walls on A6000 GPUs shared with other jobs (upper bounds on exclusive GPU time); from the run logs.

| phase | GPU-hours | outputs |
|---|---|---|
| pruning-strength sweep, dance (M = 1 and N/2 on beat alignment and repetition, N = 32, 4 x 160 sequences) | 3.0 | `analysis_json/r1/pruning_sweep.json` |
| pruning-strength sweep, music (M = 1 and N/2 on prompt adherence and motif recurrence, N = 32, 4 x 80 clips, scoring) | 32.7 | `analysis_json/r1/pruning_sweep.json`, `analysis_json/r1/prune_*` |
| horizon study, dance K = 6 (40 conditions) and K = 8 (16 conditions), both rewards, both return rules, N in {8, 32} | 49.4 | `analysis_json/r1/dance_k6.json`, `dance_k8.json` |
| per-boundary tilt alpha = 0.01, dance K = 6 / 8, beat alignment, both schedules and return rules, N = 32 | 7.4 | `analysis_json/r1/dance_k6.json`, `dance_k8.json` |
| K = 8 music qualification (20 base clips with the rolling 40 s context, seam and tempo scoring) | < 0.5 | `analysis_json/r1/qual_k8/` |
