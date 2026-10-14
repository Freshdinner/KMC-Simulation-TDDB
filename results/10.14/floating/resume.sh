#!/bin/sh
# ADR-056 验收续跑（ADR-057）：从续跑状态接着跑一条。启动方式同 run.sh（WMI，停跑按打印的 PID 用 taskkill）。
# 续跑状态来自 make_state.py（停跑前的快照）或 drive.py 每 25 个空位存的 state_{tag}.npz。
# 逐步日志 logs/run_{tag}_from{N}.log。usage: resume.sh SEED R_C_NM STATE_NPZ（STATE_NPZ 是相对本目录的文件名）
ROOT=$(cygpath -w "$(cd "$(dirname "$0")/../../.." && pwd)")
seed=$1
rc=$2
state=$3
tag=$(printf "s%s_rc%03d" "$seed" "$(echo "$rc" | awk '{printf "%d", $1 * 100 + 0.5}')")
n=$(cd "$(dirname "$0")" && PYTHONIOENCODING=utf-8 "$(dirname "$0")/../../../.venv/Scripts/python.exe" -c \
"import numpy as np; d = np.load('$state'); print('{:04d}'.format(len(d['defect_positions']) - int(d['n_initial'])))")
cmd="cmd.exe /c cd /d $ROOT && set PYTHONPATH=src&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=1&& \
set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& .venv\\Scripts\\python.exe results\\10.14\\floating\\drive.py $seed $rc 1e30 \
results\\10.14\\floating\\$state > results\\10.14\\floating\\logs\\run_${tag}_from$n.log 2>&1"
powershell.exe -NoProfile -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create \
-Arguments @{CommandLine='$cmd'}; '${tag} from $n: cmd.exe PID ' + \$r.ProcessId + ' (return ' + \$r.ReturnValue + ')'"
