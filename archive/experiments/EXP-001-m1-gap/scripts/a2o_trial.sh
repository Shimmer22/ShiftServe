#!/system/bin/sh
# Device-side single trial (runs ON the phone; no root). Measures the A2O gap with consecutive screencaps.
# usage: sh a2o_trial.sh "<inject command>" [N_STABLE=3] [MAX_MS=8000] [NOCHANGE_MS=1500]
# Output lines (all timestamps = CLOCK_REALTIME ns from toybox `date +%s%N`, one clock for everything):
#   BASE <md5 of raw screencap before injection>
#   INJ  <t_inject_start> <t_inject_end>
#   F    <idx> <capture_start> <capture_end> <md5>
INJ="$1"; N="${2:-3}"; MAX="${3:-8000}"; NOCHG="${4:-1500}"
now() { date +%s%N; }
hsh() { screencap | md5sum | cut -d' ' -f1; }
b=$(hsh); echo "BASE $b"
t0=$(now); eval "$INJ" >/dev/null 2>&1; t1=$(now)
echo "INJ $t0 $t1"
i=0; last=""; run=0; changed=0
while :; do
  s=$(now); h=$(hsh); e=$(now)
  echo "F $i $s $e $h"
  [ "$h" != "$b" ] && changed=1
  if [ "$h" = "$last" ]; then run=$((run+1)); else run=1; fi
  last="$h"; i=$((i+1))
  el=$(( (e - t1) / 1000000 ))
  [ "$run" -ge "$N" ] && [ "$changed" = 1 ] && break
  [ "$run" -ge "$N" ] && [ "$el" -gt "$NOCHG" ] && break
  [ "$el" -gt "$MAX" ] && break
done
