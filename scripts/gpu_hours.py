"""GPU-hours per phase from the logs (results/GPU_HOURS.md). Every entry names the log it was read from; delegated
music phases (M0, M1, M1b, M2) have no per-job timing lines in their agents' logs, so their wall is taken from the
agents' reports as quoted in RESULTS.md and marked as such. GPU-hours = wall x number of GPUs the job occupied
(each job ran on one GPU; concurrent jobs on one GPU are counted by their own walls, so the total is an upper bound
on exclusive GPU time for the phases that shared a GPU).

  .venv/bin/python scripts/gpu_hours.py
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# (phase, log glob, how to read the seconds, note)
JOBS = [
    ("Phase 1 (chunk_s101 training, base generation x3 roots)", "chunk_s101.log", "train", ""),
    ("Phase 1 (chunk_s101 training, base generation x3 roots)", "gen_samples_s?.log,gen_samples_rerun_s?.log,gen_samples_swap_s?.log", "gen", ""),
    ("Phase 1b (12k and 30k training, 3 roots each)", "chunk_s101_12k.log,chunk_s101_30k.log", "train", ""),
    ("Phase 1b (12k and 30k training, 3 roots each)", "gen_samples_12k*.log,gen_samples_30k*.log", "gen", ""),
    ("Phase 1c (candidates E, F training, 3 roots each)", "chunk_dispE.log,chunk_seamF.log", "train", ""),
    ("Phase 1c (candidates E, F training, 3 roots each)", "gen_samples_E*.log,gen_samples_F*.log", "gen", ""),
    ("Phase 2a (alpha pilot, R_BA twist rollouts)", "twist_collect_*.log", "collect", ""),
    ("Phase 2a (alpha pilot, R_BA twist rollouts)", "alpha_pilot.log", "fixed:204", "pilot wall from RESULTS.md Phase 2a (204 s)"),
    ("Phase 2 (grid alpha 0.2, both GPUs)", "grid_s?.log", "gen", ""),
    ("Phase 2 background (candidate H training + 3 roots)", "chunk_dispH.log", "train", ""),
    ("Phase 2 background (candidate H training + 3 roots)", "gen_samples_H*.log", "gen", ""),
    ("Phase 2b (grids alpha 0.02 and 0.05)", "grid_a002.log,grid_a005.log", "gen", ""),
    ("Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05)", "rep_collect_*.log", "collect", ""),
    ("Phase 3 (R_rep rollouts, grids alpha 0.02 and 0.05)", "grid_rep_a002.log,grid_rep_a005.log", "gen", ""),
    ("Phase 3b (tempered twist grids, oracle grid)", "grid_rep_tw*.log,grid_rep_oracle.log", "gen", ""),
    ("Phase 3c (ensemble grids, within-set trace)", "grid_rep_ens_*.log", "gen", ""),
    ("Phase 3c (ensemble grids, within-set trace)", "phase3c_within_set.log", "fixed:3600", "trace re-run of 160 sequences at N = 32 with 5 oracle sets; wall not logged, estimated 1 h from log timestamps"),
    ("Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids)", "collect_sets.log", "collect_sets", ""),
    ("Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids)", "phase3d_wstwist_grid.log,phase3d_rollout_grids.log", "gen_all", ""),
    ("Phase 3d (within-set rollouts, oracle-set save, wstwist grid, rollout + matched grids)", "phase3d_oracle_sets.log", "fixed:2400", "oracle-set save (5 prompts, N = 32, M = 16), about 40 min from log timestamps"),
    ("Phase M3 (music grid, 6 shards on 2 GPUs; scoring)", "m3_grid_?.log", "gen_all", "two passes per shard (seed 2000, then seed 2001), 3 shards per GPU"),
    # Amendment W: the phases added since the tagged commit
    ("Phase U (argmax regenerations: dance roots, music replays, scoring)", "argmax_regen_gpu?.log", "gen", "5 dance roots, both GPUs"),
    ("Phase U (argmax regenerations: dance roots, music replays, scoring)", "u_replay_?.log", "wall_sum", "sum of the per-sequence replay walls (2 shards, one per GPU)"),
    ("Phase U (argmax regenerations: dance roots, music replays, scoring)", "u_argmax_scores.log,u_argmax_tempo.log", "mtime", "GPU scoring, wall from the log timestamps"),
    ("Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring)", "m4_grid_?.log", "gen_all", "two passes per shard, 3 shards per GPU"),
    ("Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring)", "m4_scores.log,m4_tempo.log", "mtime", "GPU scoring, wall from the log timestamps"),
    ("Phase W (M4 argmax replays, scoring)", "w_replay_?.log", "wall_sum", "sum of the per-sequence replay walls (2 shards on GPU 1)"),
    ("Phase W (M4 argmax replays, scoring)", "w_argmax_scores.log,w_argmax_tempo.log", "mtime", "GPU scoring, wall from the log timestamps"),
    ("Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids)", "probe_L.log", "fixed:165", "throughput probe, 300 steps, wall from its step lines (crashed at the final save, run deleted)"),
    ("Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids)", "chunk_dispE_L.log", "train", ""),
    ("Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids)", "gen_samples_L*.log", "gen", ""),
    ("Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids)", "L_rep_collect_*.log", "collect", ""),
    ("Phase X (chunk_dispE_L training, 3 roots, R_rep rollouts, grids)", "grid_L_*.log", "gen", ""),
]
DELEGATED = [  # from the agents' reports as quoted in RESULTS.md
    ("Phase M0 (music feasibility, GPU 1)", 50 / 60, "RESULTS.md Phase 3 / M0: 'about 50 min wall'"),
    ("Phase M1 (ACE-Step harness, GPU 1)", 3.0, "RESULTS.md Phase 3b / M1: 'Wall about 3 h (of which the variance study 2.41 h)'"),
    ("Phase M1b (B1 / B2 harnesses, GPU 1)", 2.6, "RESULTS.md Phase 3c / M1b: 'about 2.6 h wall'"),
    ("Phase M2 (B2 qualification, base samples, GPU 1)", 1.25, "RESULTS.md Phase 3d / M2: 'Wall about 1 h 15 min'"),
]


def seconds_from(path, how):
    txt = open(path, errors="ignore").read()
    if how == "train":
        m = re.findall(r"done: (\d+)s", txt)
    elif how in ("gen", "gen_all"):
        m = re.findall(r"done: \d+ generated, \d+ skipped, (\d+)s", txt) or re.findall(r"done: \d+ runs in (\d+)s", txt)
    elif how == "collect":
        m = re.findall(r"done: \d+ sequences in (\d+)s", txt)
    elif how == "collect_sets":
        m = re.findall(r"done: \d+ new sets in (\d+)s", txt)
    elif how == "wall_sum":
        m = re.findall(r"wall=([\d.]+)s", txt)
        return int(round(sum(float(x) for x in m))) if m else None
    elif how == "mtime":  # first line's dynamo timestamp is not reliable for short logs: use file birth -> last modification
        st = os.stat(path)
        birth = getattr(st, "st_birthtime", None)
        if birth is None:
            import subprocess
            out = subprocess.run(["stat", "-c", "%W", path], capture_output=True, text=True).stdout.strip()
            birth = float(out) if out and out != "0" else None
        return int(round(st.st_mtime - birth)) if birth else None
    else:
        return None
    if not m:
        return None
    return sum(int(x) for x in m) if how == "gen_all" else int(m[-1])


def main():
    rows, per_phase = [], {}
    for phase, pat, how, note in JOBS:
        if how.startswith("fixed:"):
            secs = int(how.split(":")[1])
            rows.append((phase, pat, secs, note or "fixed from RESULTS.md"))
            per_phase[phase] = per_phase.get(phase, 0) + secs
            continue
        for g in pat.split(","):
            for f in sorted(glob.glob(os.path.join(LOGS, g))):
                secs = seconds_from(f, how)
                if secs is None:
                    rows.append((phase, os.path.basename(f), None, "no timing line"))
                    continue
                rows.append((phase, os.path.basename(f), secs, note))
                per_phase[phase] = per_phase.get(phase, 0) + secs
    md = ["# GPU-hours per phase (from logs; `scripts/gpu_hours.py`)", "",
          "Each job ran on one GPU; the seconds are the job's own logged wall (`done: ... s`), summed per phase. Jobs that shared a GPU "
          "(Phase 3d, Phases M3 and M4 with 3 shards per GPU, the Phase U and W music replays with 2 shards) are counted by their own walls, so those phases are upper bounds on exclusive GPU time; "
          "the exclusive Phase M3 and M4 figures are derived below from the shard timestamps. Phases U, M4, W and X were added by Amendment W (2026-09-16). Delegated music phases are taken from the agents' reports.", "",
          "| phase | GPU-hours (sum of job walls) | source |", "|---|---|---|"]
    total = 0.0
    for phase in dict.fromkeys(p for p, *_ in JOBS):
        h = per_phase.get(phase, 0) / 3600
        total += h
        src = ", ".join(sorted({r[1] for r in rows if r[0] == phase and r[2] is not None}))
        md.append(f"| {phase} | {h:.2f} | {src} |")
    for phase, h, src in DELEGATED:
        total += h
        md.append(f"| {phase} | {h:.2f} | {src} |")
    m3 = per_phase.get("Phase M3 (music grid, 6 shards on 2 GPUs; scoring)", 0) / 3600
    m4 = per_phase.get("Phase M4 (music R_motif grid, 6 shards on 2 GPUs; scoring)", 0) / 3600
    # exclusive M3 figure from the shard logs' first timestamp (relaunch, 2026-09-13 09:43) to the last shard's end (2026-09-15 00:16): 2 GPUs x wall
    import datetime as dt
    starts, ends = [], []
    for f in glob.glob(os.path.join(LOGS, "m3_grid_?.log")):
        m = re.search(r"W(\d{4}) (\d\d:\d\d:\d\d)", open(f, errors="ignore").read())
        if m:
            starts.append(dt.datetime.strptime("2026" + m.group(1) + m.group(2), "%Y%m%d%H:%M:%S"))
        ends.append(dt.datetime.fromtimestamp(os.path.getmtime(f)))
    wall_h = (max(ends) - min(starts)).total_seconds() / 3600 if starts else float("nan")
    m3_excl = 2 * wall_h
    # M4: shard logs start at the launch (file birth) and end at the last shard's end; 2 GPUs x wall
    m4_files = glob.glob(os.path.join(LOGS, "m4_grid_?.log"))
    import subprocess
    m4_starts = [float(subprocess.run(["stat", "-c", "%W", f], capture_output=True, text=True).stdout.strip() or 0) for f in m4_files]
    m4_wall_h = (max(os.path.getmtime(f) for f in m4_files) - min(m4_starts)) / 3600 if m4_files and min(m4_starts) > 0 else float("nan")
    m4_excl = 2 * m4_wall_h
    md += ["", f"**Total: {total:.1f} GPU-hours as the sum of job walls.** The Phase M3 shards ran 3 per GPU, so their summed walls ({m3:.1f} h) overstate exclusive GPU time: "
           f"the grid occupied both GPUs from the relaunch to the last shard's end, {wall_h:.1f} h of wall, i.e. **{m3_excl:.1f} GPU-hours exclusive**, plus about 2 GPU-hours for the "
           f"first (prompt-major) attempt whose 200 finished sequences were reused. The Phase M4 shards likewise ran 3 per GPU (summed walls {m4:.1f} h); "
           f"the grid occupied both GPUs for {m4_wall_h:.1f} h of wall, i.e. **{m4_excl:.1f} GPU-hours exclusive**. "
           f"**Exclusive total: about {total - m3 + m3_excl + 2 - m4 + m4_excl:.0f} GPU-hours.** "
           "Not included: CPU-only scoring (beat_this, aesthetics), analysis scripts, unit tests, smoke tests and the interactive diagnostics, each minutes.",
           "", "## Per-job walls", "", "| phase | log | seconds | note |", "|---|---|---|---|"]
    for phase, f, secs, note in rows:
        md.append(f"| {phase} | `{f}` | {secs if secs is not None else '-'} | {note} |")
    open(os.path.join(ROOT, "results", "GPU_HOURS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:len(JOBS) + 12]))


if __name__ == "__main__":
    main()
