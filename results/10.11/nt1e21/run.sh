#!/bin/sh
# N_T = 10²¹ cm⁻³ 试算，seed 1–2 并发。启动方式同 ../discrete/run_all.sh：进程经 WMI（Win32_Process.Create）启动，不在当前
# shell 的进程树里；停跑按打印的 PID 用 taskkill。每条一份逐步日志 logs/run_s{seed}.log。
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in 1 2; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.11\\nt1e21\\drive.py $seed \
> results\\10.11\\nt1e21\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
