#!/bin/sh
# ADR-063/064（E_A 指数尾、离域判据）之后的 3.6 V 验收（examples/zro2_tddb.yaml 原样，seed 1、2）。一个通道依次跑给定的种子，每条机时上限 MAX_WALL_H，
# 到点存续跑状态 state_s{seed}.npz 后接下一个种子；两个通道并发（每条峰值工作集约 1.5 GB）。
# 已有 state_s{seed}.npz 时续跑：先把它复制成 state_s{seed}_n{NNNN}.npz（NNNN 是上一段停下时的生成空位数，取自日志最后一条 "done wall"）
# 作为续跑起点，逐步日志写到 logs/run_s{seed}_from{NNNN}.log；没有时从 t = 0 起跑，日志写到 logs/run_s{seed}.log。
# 启动方式同 ../../10.14/floating/run.sh：经 WMI（Win32_Process.Create）启动，不在当前 shell 的进程树里；停跑按打印的 PID 用 taskkill /T。
# usage: run.sh MAX_WALL_H SEED [SEED ...]
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cygpath -w "$(cd "$HERE/../../.." && pwd)")
mkdir -p "$HERE/logs"
wall=$1
shift
chain=""
for seed in "$@"; do
  [ -n "$chain" ] && chain="$chain& "
  if [ -f "$HERE/state_s$seed.npz" ]; then
    n=$(grep -h '^done wall' "$HERE"/logs/run_s$seed*.log | tail -n 1 | sed 's/.*n_generated=\([0-9]*\).*/\1/')
    tag=$(printf '%04d' "$n")
    cp "$HERE/state_s$seed.npz" "$HERE/state_s${seed}_n$tag.npz"
    args="$seed $wall results\10.19\accept\state_s${seed}_n$tag.npz"
    suffix="_from$tag"
  else
    args="$seed $wall"
    suffix=""
  fi
  chain="$chain.venv\Scripts\python.exe results\10.19\accept\drive.py $args > results\10.19\accept\logs\\run_s$seed$suffix.log 2>&1"
done
cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& $chain"
powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seeds $*: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
