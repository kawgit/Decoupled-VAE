#!/bin/bash
# Stop experiments on SCC by name (as scc/gpus.sh shows them): a running one ends within seconds and its holder
# keeps the GPU for the next one; a waiting one is taken out of the queue. Its artifacts stay.
#   scc/cancel.sh NAME...
[ $# -gt 0 ] || { echo "usage: scc/cancel.sh NAME..."; exit 1; }
ssh scc1.bu.edu bash -s -- "$@" <<'EOF'
cd /projectnb/cs599bg/kawgit56/queue || exit 1
for name; do
  if [ -e "running/$name" ]; then
    mkdir -p cancel && touch "cancel/$name"
    echo "$name: stopping (its holder checks every few seconds)"
  elif waiting=$(ls pending/"$name" pools/*/"$name" 2>/dev/null | head -1) && [ -n "$waiting" ]; then
    mkdir -p done && mv "$waiting" done/ && echo "$name: taken out of the queue"
  else
    echo "$name: not running or waiting"
  fi
done
EOF
