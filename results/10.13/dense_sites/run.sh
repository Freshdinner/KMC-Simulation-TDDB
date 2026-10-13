#!/bin/sh
# ADR-055 验收：前驱位点按最小间隔 0.26 nm 铺满（不设浓度）、r_c = 0.3 nm。examples/zro2_tddb.yaml 工况，seed 并发跑到击穿；
# 应力时间不设上限（10³⁰ s），停在击穿或 3000 个空位。每条约 80 万个前驱位点，内存比 ADR-054 的运行大，默认只并发 2 条。
# 启动方式同 ../../10.12/percolation/run.sh：进程经 WMI（Win32_Process.Create）启动，不在当前 shell 的进程树里；停跑按打印的 PID
# 用 taskkill。每条一份逐步日志 logs/run_s{seed}.log。usage: run.sh [SEED ...]（缺省 1 2）
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in ${@:-1 2}; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.13\\dense_sites\\drive.py $seed 1e30 \
> results\\10.13\\dense_sites\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
