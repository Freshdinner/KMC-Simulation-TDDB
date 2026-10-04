#!/bin/sh
# what-if（O_i⁻ 与空位重合，生成按电荷守恒记账）：seed 1、2，空位上限 400、限流 1e8 A/cm²，与 ext1000 的 4 条并行。
# 启动方式同 ../ext1000/run_all.sh（WMI 起进程，不在当前 shell 的进程树里）。
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
for seed in 1 2; do
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\Scripts\python.exe results\10.03\oi_colocated\drive_oi.py $seed 400 1e8 \
> results\10.03\oi_colocated\logs\run_s$seed.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; 'seed $seed: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
