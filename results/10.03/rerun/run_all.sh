#!/bin/sh
# ADR-045：seed 1–4 开热反馈、空位上限 400、限流 1e8 A/cm²。每条一份逐步日志 logs/run_s{seed}.log。
# 并发 2 条（xargs -P 2）：单进程峰值工作集约 1.9 GB，开跑时系统空闲内存只有约 1.5 GB；第 4 条本来就要排队，
# 2 + 2 与 3 + 1 的总墙钟相当，空出的名额留给 check_states.py
cd "$(dirname "$0")/../../.."
export PYTHONPATH="src" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf "1\n2\n3\n4\n" | xargs -P 2 -I{} sh -c \
  '.venv/Scripts/python.exe results/10.03/rerun/drive.py {} 400 1e8 > results/10.03/rerun/logs/run_s{}.log 2>&1'
