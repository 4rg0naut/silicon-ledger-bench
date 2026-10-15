#!/bin/sh
# quiet_probe.sh — reproducible snapshot of WHAT ELSE is running, before trusting a measurement.
#
# Motivation (2026-10-10): past runs showed unexplained slowdowns and we could not say what
# was contending. oMLX can be quit, but Apple's own background ML/media/telemetry daemons cannot,
# so every benchmark session must start with a snapshot it can be compared against later.
#
# usage: sh quiet_probe.sh [label]        # label defaults to "unlabelled"
# writes: <repo>/results/quiet-<stamp>-<label>.txt  (and prints a QUIET-KEY block for diffing)
set -u
export LC_ALL=C   # Apple CLIs emit localised decimals (ps "9,1", sysctl "3072,00M")
label=${1:-unlabelled}
here=$(cd "$(dirname "$0")" && pwd)       # this script's directory (tools/probes)
root=$(cd "$here/../.." && pwd)           # repo root
outdir="$root/results"
out="$outdir/quiet-$(date -u +%Y%m%dT%H%M%SZ)-$label.txt"
mkdir -p "$outdir"

# Apple background ML / media / telemetry / sync suspects. Presence is not guilt; activity is.
WATCH="oMLX omlx-server distilld mediaanalysisd photolibraryd photoanalysisd cloudd bird backupd
mds mds_stores mdworker spotlightknowledged corespotlightd corespeechd assistantd siri
intelligenceflowd modelcatalogd knowledgeconstructiond aned ANECompilerService powerlogd
powerd thermald ScreenTimeAgent VTEncoderXPCService avconferenced"

{
  echo "# quiet_probe — $(date '+%F %T %z')  label=$label"
  echo "## identity"
  sw_vers | sed 's/^/  /'
  echo "  hw.model=$(sysctl -n hw.model)  gpu_cores=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
  echo "  uptime=$(uptime | sed 's/.*up //; s/,.*load/  load/')"

  echo "## power / thermal / wake"
  pmset -g therm 2>/dev/null | sed 's/^/  /'
  pmset -g batt 2>/dev/null | tail -1 | sed 's/^/  /'
  pmset -g 2>/dev/null | grep -iE "^ *(sleep|displaysleep|powernap|womp|networkoversleep)" | sed 's/^/  /'
  echo "  assertions_holding_awake: $(pmset -g 2>/dev/null | grep -o 'sleep prevented by.*')"

  echo "## memory"
  sysctl -n vm.swapusage | sed 's/^/  swap: /'
  memory_pressure 2>/dev/null | tail -2 | sed 's/^/  /'
  vm_stat | awk '/Pages free|Pages active|Pages wired|Pageins|Pageouts/{print "  " $0}'

  echo "## top cpu (cumulative since process start)"
  ps -Ao pid,pcpu,pmem,nlwp,etime,comm -r 2>/dev/null | head -16 | sed 's/^/  /'

  echo "## watchlist (pid cpu% mem% threads elapsed name) — presence AND activity"
  for p in $WATCH; do
    pids=$(pgrep -x "$p" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    [ -n "$pids" ] || continue
    ps -o pid,pcpu,pmem,nlwp,etime,comm -p "$pids" 2>/dev/null | tail -n +2 | sed 's/^/  /'
  done

  echo "## engine baseline (enginemon, unprivileged, 5s) — observation only, no load generated"
  if [ -x "$here/enginemon" ]; then
    "$here/enginemon" --interval 500 --duration 5 2>&1 | sed 's/^/  /'
  else
    echo "  (enginemon binary missing — build: clang -O2 -o enginemon enginemon.c -framework CoreFoundation)"
  fi

  echo "## canary (fixed work; compare across sessions)"
  if [ -x "$here/canary" ]; then
    i=1; while [ $i -le 3 ]; do "$here/canary" 2>&1 | sed 's/^/  run'"$i"' /'; i=$((i+1)); done
  else
    echo "  (canary missing — build: clang -O0 -o canary canary.c)"
  fi

  echo "## verdict"
  n=$(for p in $WATCH; do pgrep -qx "$p" 2>/dev/null && echo x; done | wc -l | tr -d ' ')
  echo "  suspect_daemons_up=$n"
  pgrep -qx oMLX 2>/dev/null && echo "  oMLX=UP  (measurement host is LOADED; energy arms invalid)" || echo "  oMLX=down"
  echo "  QUIET-KEY: omlx=$(pgrep -qx oMLX && echo up || echo down) loadavg=$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $2}') suspects=$n"
} > "$out" 2>&1

echo "wrote $out"
grep -E "QUIET-KEY|oMLX=|suspect_daemons_up" "$out"
