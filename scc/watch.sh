#!/bin/bash
# Watch experiments on SCC (by name, as scc/gpus.sh shows them) until every one's run.log has a line matching
# PATTERN, and stop at the first sign of trouble instead of waiting out the budget. Trouble is:
# - still waiting for a holder START minutes after it was queued;
# - no run.log LOG minutes after its holder started it, or a run.log that says it runs on the CPU;
# - a run.log that has not grown for STALE minutes;
# - a traceback in its stderr, or an exit file (it ended) before PATTERN;
# - BUDGET minutes gone.
# Prints one status line per experiment each check, and the reason it stopped; exits 0 once all match, else 1.
#   scc/watch.sh [-u PATTERN] [-s START] [-l LOG] [-q STALE] [-b BUDGET] [-i SECONDS] NAME...
PATTERN="round 0 done" START=30 LOG=8 STALE=20 BUDGET=240 INTERVAL=60
while getopts "u:s:l:q:b:i:" o; do
  case $o in
    u) PATTERN=$OPTARG ;; s) START=$OPTARG ;; l) LOG=$OPTARG ;; q) STALE=$OPTARG ;; b) BUDGET=$OPTARG ;;
    i) INTERVAL=$OPTARG ;; *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || { echo "usage: scc/watch.sh [-u PATTERN] [-s START] [-l LOG] [-q STALE] [-b BUDGET] NAME..."; exit 2; }
began=$(date +%s)
while true; do
  # per experiment: name|state|minutes in that state|run.log's age in minutes (or -)|device|exit|traceback|matched
  status=$(ssh scc1.bu.edu bash -s -- "$(printf '%q ' "$PATTERN" "$@")" <<'EOF'
pattern=$1; shift
cd /projectnb/cs599bg/kawgit56 || exit 1
now=$(date +%s)
age() { echo $(( (now - $(stat -c %Y "$1")) / 60 )); }
for name; do
  a=Decoupled-VAE/artifacts/$name
  queued=$(ls queue/pending/"$name" queue/pools/*/"$name" 2>/dev/null | head -1)
  if [ -n "$queued" ]; then echo "$name|waiting|$(age "$queued")|-|-|-|-|-"; continue; fi
  [ -f "$a/command.sh" ] || { echo "$name|missing|0|-|-|-|-|-"; continue; }
  log=- device=- code=- tb=- matched=-
  [ -f "$a/run.log" ] && log=$(age "$a/run.log") && device=$(grep -o "'device': device(type='[a-z]*'" "$a/run.log" | head -1 | grep -o "'[a-z]*'$" | tr -d "'")
  [ -f "$a/exit" ] && code=$(cat "$a/exit")
  grep -q Traceback "$a/stderr.log" 2>/dev/null && tb=yes
  grep -q -- "$pattern" "$a/run.log" 2>/dev/null && matched=yes
  echo "$name|started|$(age "$a/command.sh")|$log|${device:--}|$code|$tb|$matched"
done
EOF
)
  echo "$(date +%H:%M:%S)"
  problems="" done=0
  while IFS="|" read -r name state mins log device code tb matched; do
    [ -n "$name" ] || continue
    echo "  $name: $state ${mins}m, run.log ${log}m old, device $device, exit $code, traceback $tb, matched $matched"
    [ "$matched" = yes ] && { done=$((done + 1)); continue; }
    if [ "$state" = missing ]; then problems+="$name is neither queued nor started; "
    elif [ "$state" = waiting ]; then
      [ "$mins" -ge "$START" ] && problems+="$name has waited ${mins}m for a holder; "
    else
      [ "$log" = - ] && [ "$mins" -ge "$LOG" ] && problems+="$name has no run.log ${mins}m after it started; "
      [ "$device" = cpu ] && problems+="$name runs on the CPU; "
      [ "$log" != - ] && [ "$log" -ge "$STALE" ] && problems+="$name's run.log has not grown for ${log}m; "
      [ "$tb" = yes ] && problems+="$name has a traceback in stderr.log; "
      [ "$code" != - ] && problems+="$name ended (exit $code) before matching '$PATTERN'; "
    fi
  done <<< "$status"
  [ -z "$status" ] && problems="could not reach SCC; "
  [ -n "$problems" ] && { echo "STOP: $problems"; exit 1; }
  [ "$done" -eq $# ] && { echo "DONE: every run.log matched '$PATTERN'"; exit 0; }
  (( $(date +%s) - began >= BUDGET * 60 )) && { echo "STOP: the ${BUDGET}-minute budget is gone"; exit 1; }
  sleep "$INTERVAL"
done
