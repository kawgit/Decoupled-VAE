#!/bin/bash
# What's running on SCC: each GPU job (holders and one-off jobs), its GPU, and the experiment each holder is
# running (or how long it has sat idle); then the experiments waiting in the queue and its pools.
#   scc/gpus.sh
ssh scc1.bu.edu bash -s <<'EOF'
cd /projectnb/cs599bg/kawgit56/queue || exit 1
now=$(date +%s)
ago() { local s=$(( now - $(date -d "$1" +%s) )); printf "%dh%02dm" $((s / 3600)) $((s % 3600 / 60)); }
printf "%-8s %-11s %-5s %-24s %-12s %s\n" JOB NAME STATE "QUEUE@HOST" GPU DOING
qstat -u "$USER" | tail -n +3 | while read -r id prior name user state day time queue rest; do
  [ "$state" = "qw" ] && queue="(waiting since $time)"
  log=$(ls holders/*.o"$id" 2>/dev/null | head -1)
  gpu="" doing=""
  if [ -n "$log" ]; then
    gpu=$(head -1 "$log" | sed -n 's/.*: \(NVIDIA \|Tesla \)\{0,1\}\([^,]*\),.*/\2/p' | sed 's/-SXM2-16GB\|-PCIE-16GB//')
    last=$(grep -E "^[A-Z][a-z]{2} .* (start|end) " "$log" | tail -1)
    stamp=$(echo "$last" | sed 's/ \(start\|end\) .*//')
    if echo "$last" | grep -q " start "; then
      doing="running $(echo "$last" | sed 's/.* start \([^:]*\):.*/\1/') for $(ago "$stamp")"
    elif [ -n "$last" ]; then
      doing="idle for $(ago "$stamp")"
    else
      doing="idle since it started"
    fi
  elif [ "$state" = "r" ]; then
    doing="one-off job"
  fi
  printf "%-8s %-11s %-5s %-24s %-12s %s\n" "$id" "$name" "$state" "${queue%%.scc*}" "$gpu" "$doing"
done
echo
echo "waiting: $(ls pending | grep -v / | tr '\n' ' ')"
for pool in pools/*/; do
  [ -d "$pool" ] && echo "waiting for $(basename "$pool"): $(ls "$pool" | tr '\n' ' ')"
done
EOF
