#!/bin/sh
# ADR-052 验证续：与 run_all.sh 同一工况与种子，应力时间上限改为 10⁷ s，跑到击穿，结果写到 long/。同一种子在 10⁴ s 之前的
# 记录与 run_all.sh 的逐位相同。启动方式同 run_all.sh（WMI，不在当前 shell 的进程树里）。
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/long/logs"
for seed in 1 2 3 4; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.11\\discrete\\drive.py $seed 1e7 long \
> results\\10.11\\discrete\\long\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
