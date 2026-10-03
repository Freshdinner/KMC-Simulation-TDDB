#!/bin/sh
# ADR-045 数值抽查队列：每条轨迹跑完（final_s{seed}.npz 出现）后，单进程依次抽查它的 100 / 200 / 400 个空位快照。
# 与 run_all.sh 的 2 条轨迹并行时合计最多 3 个进程
cd "$(dirname "$0")/../../.."
export PYTHONPATH="src" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for seed in 1 2 3 4; do
  while [ ! -f "results/10.03/rerun/final_s$seed.npz" ]; do sleep 60; done
  .venv/Scripts/python.exe results/10.03/rerun/check_states.py $seed 0 100 200 400 \
    > "results/10.03/rerun/logs/check_s$seed.log" 2>&1
done
