#!/bin/sh
# ADR-060/061 之后的 3.6 V 验收（examples/zro2_tddb.yaml 原样，seed 1–4）。一个通道依次跑给定的种子，每条机时上限 MAX_WALL_H，
# 到点存续跑状态 state_s{seed}.npz 后接下一个种子；两个通道并发（每条峰值工作集约 1.5 GB）。
# 启动方式同 ../../10.14/floating/run.sh：经 WMI（Win32_Process.Create）启动，不在当前 shell 的进程树里；停跑按打印的 PID 用 taskkill /T。
# 每条一份逐步日志 logs/run_s{seed}.log。usage: run.sh MAX_WALL_H SEED [SEED ...]
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
wall=$1
shift
chain=""
for seed in "$@"; do
  [ -n "$chain" ] && chain="$chain& "
  chain="$chain.venv\\Scripts\\python.exe results\\10.15\\accept\\drive.py $seed $wall > results\\10.15\\accept\\logs\\run_s$seed.log 2>&1"
done
cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& $chain"
powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seeds $*: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
