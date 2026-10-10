#!/bin/sh
# 每 60 s 记一次本机空闲内存与 4 条日志的最新一步（logs/watch.log）。4 条都写出 done / Traceback，或空闲内存低于
# 500 MB 时退出，退出前打印原因。只看不停跑：内存不足时由人决定是否 taskkill。
HERE=$(cd "$(dirname "$0")" && pwd)
while true; do
  free_kb=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" | tr -d '\r')
  line="$(date '+%m-%d %H:%M:%S') free $((free_kb / 1024)) MB"
  finished=0
  for seed in 1 2 3 4; do
    log="$HERE/logs/run_s$seed.log"
    last=$(grep -E '^ *[0-9]+ t=' "$log" 2>/dev/null | tail -1 | awk '{print $1, $3, $(NF-2), $(NF-1)}')
    line="$line | s$seed $last"
    if grep -qE '^done|Traceback|MemoryError' "$log" 2>/dev/null; then finished=$((finished + 1)); fi
  done
  echo "$line" >> "$HERE/logs/watch.log"
  if [ "$free_kb" -lt 512000 ]; then echo "LOW MEMORY: $line"; exit 0; fi
  if [ "$finished" -eq 4 ]; then echo "ALL FINISHED: $line"; exit 0; fi
  sleep 60
done
