#!/bin/sh
# ADR-053 验收：E_A 全部前驱相同，examples/zro2_tddb.yaml 默认工况，seed 1–4 并发跑到击穿。应力时间上限取 10³⁰ s（不按时间停，
# E_A 只改时间标尺），停在击穿或 3000 个空位。启动方式同 ../../10.11/discrete/run_all.sh：进程经 WMI（Win32_Process.Create）启动，
# 不在当前 shell 的进程树里；停跑按打印的 PID 用 taskkill。每条一份逐步日志 logs/run_s{seed}.log。
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in 1 2 3 4; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.12\\fixed_ea\\drive.py $seed 1e30 \
> results\\10.12\\fixed_ea\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
