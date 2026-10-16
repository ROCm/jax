#!/bin/bash
# Debug only. Once the suite has run WATCHDOG_START_MIN minutes, every
# WATCHDOG_EVERY_MIN minutes find the Python processes burning CPU over a 10 s
# window and log, for each: per-thread CPU and names (native threads included),
# the load addresses of the ROCm libraries, gdb backtraces of every thread, and
# a py-spy dump.
set +x -u

ticks() {  # utime+stime of a /proc stat file; comm may contain spaces
  local s
  s=$(<"$1") 2>/dev/null || { echo 0; return; }
  s=${s##*) }
  set -- $s
  echo $(( ${12} + ${13} ))
}

sleep $(( ${WATCHDOG_START_MIN:-35} * 60 ))
while true; do
  echo "===== hang watchdog $(date -u +%H:%M:%S) ====="
  declare -A before=() tbefore=()
  pids=$(pgrep python || true)
  for p in $pids; do
    before[$p]=$(ticks /proc/$p/stat)
    for t in /proc/$p/task/*; do tbefore[$p.${t##*/}]=$(ticks $t/stat); done
  done
  sleep 10
  busy=()
  for p in $pids; do
    d=$(( $(ticks /proc/$p/stat) - ${before[$p]:-0} ))
    echo "pid $p cpu_ticks_10s=$d $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-80)"
    (( d > 300 )) && busy+=("$p")
  done
  for p in "${busy[@]}"; do
    echo "----- busy pid $p: threads (ticks in window, tid, name)"
    hot=()
    for t in /proc/$p/task/*; do
      tid=${t##*/}
      [[ -n ${tbefore[$p.$tid]:-} ]] || continue
      d=$(( $(ticks $t/stat) - ${tbefore[$p.$tid]} ))
      (( d > 0 )) && echo "  $d $tid $(<$t/comm)"
    done | sort -rn | head -20 | tee /tmp/hot.$p
    while read -r d tid _; do (( d > 300 )) && hot+=("$tid"); done < /tmp/hot.$p
    echo "----- busy pid $p: ROCm library mappings"
    grep -E 'amdhip64|hsa-runtime64|rocprofiler|roctracer|librccl|amd_comgr' /proc/$p/maps | awk '{print $1, $3, $6}' | head -30
    # A full "thread apply all" over ~900 threads hides the hot ones; dump only those.
    if (( ${#hot[@]} )); then
      echo "----- busy pid $p: gdb bt of hot threads ${hot[*]}"
      timeout 300 gdb -p "$p" -batch -ex 'set pagination off' \
        -ex "python [(t.switch(), print('=== LWP %d' % t.ptid[1]), gdb.execute('bt 40')) for t in gdb.selected_inferior().threads() if t.ptid[1] in ($(IFS=,; echo "${hot[*]}"),)]" \
        2>&1 | grep -vE '^(\[New LWP|warning: )' | head -400
    fi
    echo "----- busy pid $p: py-spy dump --native"
    timeout 180 "${PYSPY:-py-spy}" dump --native --pid "$p" 2>&1 | head -120
  done
  unset before tbefore
  sleep $(( ${WATCHDOG_EVERY_MIN:-20} * 60 ))
done
