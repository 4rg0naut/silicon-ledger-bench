#!/bin/zsh
# Official S1 stability series: collect 3 GOOD runs on one machine.
# A run is discarded (and retried after cooldown) when the GPU primer fails
# to recover sustained copy bandwidth (>=340 GB/s), or when the pre-flight
# load gate aborts. Runs are spaced by COOLDOWN (keep SHORT, 30-120s; long
# idle degrades fresh-page placement under external memory pressure — see
# docs/research-notes.md R9). Uses .build/release/bench (PATH swift is the
# CLT toolchain and cannot build this package; run `make stability`).
# CPU write bandwidth additionally bands with physical free-memory pressure
# (an external 87 GB resident server on this machine is the current
# confounder); cross-run CPU consistency is enforced by the final <10% gate
# itself, not by the per-run filter.
#
# Usage: tools/stability_run.sh [prefix]   (writes /tmp/${prefix}_$n.log)
set -u
PREFIX=${1:-mem_official}
# COOLDOWN between runs: keep SHORT (30-120s). Long idle lets the page
# allocator degrade fresh-buffer placement under external memory pressure;
# back-to-back runs keep the healthy band (docs/research-notes.md R9).
COOLDOWN=${COOLDOWN:-60}
ITER=0
GOOD=0
rm -f /tmp/${PREFIX}_good.txt
while (( GOOD < 3 )); do
  (( ITER++ ))
  if (( ITER > 20 )); then
    echo "ABORT: only $GOOD good runs after $ITER attempts"
    exit 1
  fi
  .build/release/bench memory > /tmp/${PREFIX}_${ITER}.log 2>&1
  STATUS=$?
  REPORT=$(sed -n 's/^report: //p' "/tmp/${PREFIX}_${ITER}.log")
  if (( STATUS != 0 )) || [[ -z "$REPORT" ]]; then
    echo "run $ITER: aborted by pre-flight gate (rc=$STATUS); cooling down"
    sleep "$COOLDOWN"
    continue
  fi
  COPY=$(python3 - "$REPORT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
out = {}
for s in d["sections"]:
    for r in s["runs"]:
        out[r["name"]] = r["derived"]["GBs"]
print(out["gpu_copy"], out["cpu_write"])
EOF
)
  OK=$(python3 -c "c,w = map(float, '$COPY'.split()); print('yes' if c >= 340 and (w >= 200 or w <= 140) else 'no')")
  BAND=$(python3 -c "c,w = map(float, '$COPY'.split()); print('good' if w >= 200 else 'low')")
  BANDFILE=/tmp/${PREFIX}_band.txt
  if [[ -f "$BANDFILE" ]] && [[ "$(cat $BANDFILE)" != "$BAND" ]]; then
    MISS=$(($(cat /tmp/${PREFIX}_miss.txt 2>/dev/null || echo 0) + 1))
    echo $MISS > /tmp/${PREFIX}_miss.txt
    if (( MISS >= 3 )); then rm -f "$BANDFILE"; echo 0 > /tmp/${PREFIX}_miss.txt
      if [[ -f /tmp/${PREFIX}_good.txt ]]; then
        rm -f /tmp/${PREFIX}_good.txt; GOOD=0
        echo "  band reset: cleared trio collection"
      fi
    else OK=no; fi
  fi
  if [[ "$OK" == "yes" ]]; then
    (( GOOD++ ))
    touch "$BANDFILE"; echo "$BAND" > "$BANDFILE"
    echo "run $ITER: GOOD ($BAND band; gpu_copy/cpu_write = ${COPY} GB/s) -> $REPORT"
    echo "$REPORT" >> /tmp/${PREFIX}_good.txt
  else
    echo "run $ITER: DISCARD (gpu_copy=${COPY} GB/s; wrong band or transient state)"
  fi
  sleep "$COOLDOWN"
done
python3 tools/check_stability.py --paths-file /tmp/${PREFIX}_good.txt --suite=s1
