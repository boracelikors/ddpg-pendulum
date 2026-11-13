#!/usr/bin/env bash
# 3 reward configs x 3 seeds = 9 runs in parallel (2 CPU threads each), same noise (0.05) for all.
#   A: original reward      B: +1 upright bonus      C: +1 bonus and -1 per non-upright step
set -e
cd "$(dirname "$0")"
for seed in 0 1 2; do
  OMP_NUM_THREADS=2 python3 -u main.py --bonus 0 --penalty 0 --seed $seed --out runs/A_original_s$seed      > /dev/null 2>&1 &
  OMP_NUM_THREADS=2 python3 -u main.py --bonus 1 --penalty 0 --seed $seed --out runs/B_bonus_s$seed         > /dev/null 2>&1 &
  OMP_NUM_THREADS=2 python3 -u main.py --bonus 1 --penalty 1 --seed $seed --out runs/C_bonus_penalty_s$seed > /dev/null 2>&1 &
done
wait
echo "all 9 runs finished"
