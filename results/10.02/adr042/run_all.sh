#!/bin/sh
# 网格无关性验证：9 个状态分三路并行（单线程 BLAS），每个状态一份日志
cd "$(dirname "$0")/../../.."
export PYTHONPATH="src;results/10.02/adr042" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { for s in "$@"; do .venv/Scripts/python results/10.02/adr042/grid_independence.py "$s" 2000 > "results/10.02/adr042/grid_$s.log" 2>&1; done; }
run rerun_s2 t0_s1 &
run p4_s1 p4_s2 t0_s3 &
run p4_s3 p4_s4 t0_s4 t0_s2 &
wait
