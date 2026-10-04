#!/bin/sh
# ADR-046：seed 1–4 开热反馈、空位上限 1000、限流 1e8 A/cm²，4 条并发。每条一份逐步日志 logs/run_s{seed}.log。
# 驱动断开了事件表的引用环（drive.py 的 generation_rates），单进程常驻工作集约 0.2 GB、求事件表时瞬时峰值约 0.8 GB，
# 所以能 4 条并发。进程经 WMI（Win32_Process.Create）启动，父进程是 WMI 服务、不在当前 shell 的进程树里，
# 当前 shell 被回收时不会连带停掉；停跑按打印的 PID 用 taskkill。
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in 1 2 3 4; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.03\\ext1000\\drive.py $seed 1000 1e8 \
> results\\10.03\\ext1000\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
