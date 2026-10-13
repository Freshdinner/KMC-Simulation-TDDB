#!/bin/sh
# ADR-054 验收：贯穿通路判据 + 细丝电流 + 接电极的导电簇与电极等电位。examples/zro2_tddb.yaml 工况，N_T = 10²¹ cm⁻³、r_c = 1.0 nm
# （N_T = 10²⁰ 时 r_c ≤ 1.38 nm 不可能贯穿），seed 1–4 并发跑到击穿；应力时间不设上限（10³⁰ s），停在击穿或 3000 个空位。
# 启动方式同 ../fixed_ea/run.sh：进程经 WMI（Win32_Process.Create）启动，不在当前 shell 的进程树里；停跑按打印的 PID 用 taskkill。
# 每条一份逐步日志 logs/run_s{seed}.log。usage: run.sh [SEED ...]（缺省 1 2 3 4）
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in ${@:-1 2 3 4}; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.12\\percolation\\drive.py $seed 1e21 1.0 1e30 \
> results\\10.12\\percolation\\logs\\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
