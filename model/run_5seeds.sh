#!/usr/bin/env bash
set -e
conda activate sigma

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

for seed in 2020 2021 2022 2023 2024; do
  echo "===== seed=${seed} ====="
  python run_noflops.py --model=SIGMA --dataset=ml-1m --seed=${seed}
done
