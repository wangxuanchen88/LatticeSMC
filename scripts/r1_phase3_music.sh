#!/bin/bash
# R1 Phase 3 (music): after the dance K runs, the 10-prompt K = 8 qualification on GPU 1; if it passes, the K = 8 grid
# (40 prompts x 2 seeds x N in {8, 32} x {bon, greedy_chunk, lsp1}) as two prompt shards on the two GPUs, then scoring.
cd ${LATTICESMC_ROOT:-.}
PY=.venv/bin/python; SA3=${STABLE_AUDIO_ROOT:-third_party/stable-audio-3}; OUT=${LATTICESMC_ROOT:-.}/results/r1/samples_music_k8
until grep -q "phase 3 dance generation done" logs/r1_phase3_dance.log; do sleep 120; done
echo "music qualification start $(date)"
scripts/r1_music_qual.sh 1 > logs/r1_music_qual.log 2>&1
if ! grep -q "QUAL_PASS" logs/r1_qual_k8_compare.log; then echo "QUALIFICATION FAILED: stop $(date)"; exit 0; fi
echo "music K=8 grid start $(date)"
for s in 0 1; do
  (cd $SA3 && CUDA_VISIBLE_DEVICES=$s HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/music_lattice_k8.py --runs bon,greedy_chunk,lsp1 --Ns 8,32 --seeds 2000,2001 --shard $s/2 --out_root $OUT) > logs/r1_music_k8_s$s.log 2>&1 &
done
wait
echo "music K=8 generation done $(date)"
(cd $SA3 && CUDA_VISIBLE_DEVICES=1 HF_HOME=$SA3/.hf-cache $SA3/.venv/bin/python ${LATTICESMC_ROOT:-.}/m3/score_samples.py --samples $OUT --out $OUT/scores.json) > logs/r1_music_k8_scores.log 2>&1
CUDA_VISIBLE_DEVICES=1 m0/.venv/bin/python m3/score_tempo.py --samples $OUT --out $OUT/tempo.json --Ns 8,32 --device cuda > logs/r1_music_k8_tempo.log 2>&1
CUDA_VISIBLE_DEVICES="" m2/.venv-aes/bin/python m3/score_aes.py --samples $OUT --out $OUT/aesthetics.json --Ns 8,32 > logs/r1_music_k8_aes.log 2>&1
echo "music K=8 scoring done $(date)"
