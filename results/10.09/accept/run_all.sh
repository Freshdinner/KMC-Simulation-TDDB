#!/bin/sh
# ADR-048 验收：examples/zro2_tddb.yaml 默认工况（3.6 V / 25 °C，击穿判据 J > 10 A/cm²），seed 1–4 并发，跑到击穿。
# 启动方式同 ../../10.03/ext1000/run_all.sh：进程经 WMI（Win32_Process.Create）启动，父进程是 WMI 服务、不在当前 shell
# 的进程树里，当前 shell 被回收时不会连带停掉；停跑按打印的 PID 用 taskkill。每条一份逐步日志 logs/run_s{seed}.log。
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in 1 2 3 4; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.09\\accept\\drive.py $seed \
> results\\10.09\\accept\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
