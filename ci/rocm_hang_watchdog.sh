#!/bin/bash
# Debug only. Once the suite has run WATCHDOG_START_MIN minutes, log per-thread
# CPU use and a native stack of every Python process every WATCHDOG_EVERY_MIN
# minutes. Threads pinned near 100% CPU across dumps mean a busy spin.
set +x -u
sleep $(( ${WATCHDOG_START_MIN:-40} * 60 ))
while true; do
  echo "===== hang watchdog $(date -u +%H:%M:%S) ====="
  ps -eLo pid,tid,pcpu,stat,wchan:28,comm --sort=-pcpu | head -60
  for pid in $(pgrep python || true); do
    echo "----- py-spy dump --native --pid $pid"
    timeout 180 "${PYSPY:-py-spy}" dump --native --pid "$pid" 2>&1 | head -250
  done
  sleep $(( ${WATCHDOG_EVERY_MIN:-15} * 60 ))
done
