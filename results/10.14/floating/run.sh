#!/bin/sh
# ADR-056 验收：悬空导电簇也等电位，同一个种子下 r_c 取 0.3 / 0.35 / 0.4 nm 各跑一条。examples/zro2_tddb.yaml 工况，跑到击穿；
# 应力时间不设上限（10³⁰ s），停在击穿或 3000 个空位。每条约 80 万个前驱位点、峰值工作集约 1.5 GB，三条并发。
# 启动方式同 ../../10.13/dense_sites/run.sh：进程经 WMI（Win32_Process.Create）启动，不在当前 shell 的进程树里；停跑按打印的 PID
# 用 taskkill。每条一份逐步日志 logs/run_s{seed}_rc{100·r_c}.log。usage: run.sh SEED [R_C_NM ...]（R_C_NM 缺省 0.3 0.35 0.4）
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
mkdir -p "$(dirname "$0")/logs"
seed=$1
shift
for rc in ${@:-0.3 0.35 0.4}; do
  tag=$(printf "s%s_rc%03d" "$seed" "$(echo "$rc" | awk '{printf "%d", $1 * 100 + 0.5}')")
  cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.14\\floating\\drive.py $seed $rc 1e30 \
> results\\10.14\\floating\\logs\\run_$tag.log 2>&1"
  powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; '$tag: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
done
