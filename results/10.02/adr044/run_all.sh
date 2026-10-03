#!/bin/sh
# ADR-044 验收：9 个状态分两路并行（单线程 BLAS；列间距减半那一项单进程约 1 GB），每个状态一份日志
cd "$(dirname "$0")/../../.."
export PYTHONPATH="src" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { for s in "$@"; do .venv/Scripts/python results/10.02/adr044/acceptance.py "$s" > "results/10.02/adr044/acceptance_$s.log" 2>&1; done; }
run rerun_s2 p4_s2 p4_s4 t0_s2 t0_s4 &
run p4_s1 p4_s3 t0_s1 t0_s3 &
wait
