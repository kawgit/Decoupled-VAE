#!/bin/bash -l
# Hold one GPU on BU's SCC and run experiments from a queue folder on it as they arrive. From the repo there:
#   qsub [-l gpu_type=L40S] [-l h_rt=12:00:00] [-v POOL=l40s,IDLE=7200,MIN_LEFT=14400] scc/hold.sh
# An experiment is a file in ../queue/pending holding shell commands, run from the repo with $PY the venv's
# python and $DVAE_JOB its file's name (e.g. "$PY train_walk.py --depths 32"), so its files go to
# artifacts/<name>/ (dvae/artifacts.py), beside a copy of the commands, their stderr and exit code. Holders take
# files in name order; each moves to ../queue/running while it runs, then to ../queue/done. A file
# ../queue/cancel/<name> (scc/cancel.sh) stops that experiment within seconds, and its holder takes the next. A
# holder takes nothing new with under MIN_LEFT seconds of its time left, gives its GPU back after IDLE seconds
# with nothing to do, and stops after its current experiment once ../queue/stop exists. With POOL set, it
# takes experiments from ../queue/pools/$POOL instead (so timings compare within one kind of GPU). The class's
# academic-gpu queue holds V100s for up to 48 hours; other groups lend V100s, P100s and a few A100s for 12, and
# their A6000s, A40s, L40Ss and H200s for 1 minute (so never).
#$ -P cs599bg
#$ -N hold
#$ -cwd
#$ -j y
#$ -o ../queue/holders/
#$ -l gpus=1
#$ -l gpu_type=V100
#$ -pe omp 4
#$ -l h_rt=48:00:00

module load python3/3.13.8
export PY=$(cd .. && pwd)/venv/bin/python
Q=../queue
IDLE=${IDLE:-7200}
MIN_LEFT=${MIN_LEFT:-14400}
PENDING=$Q/pending
[ -n "$POOL" ] && PENDING=$Q/pools/$POOL
mkdir -p $PENDING $Q/running $Q/done $Q/cancel

limit=$(qstat -j "$JOB_ID" | grep -o 'h_rt=[0-9]*' | head -1 | cut -d= -f2)
end=$(( $(date +%s) + ${limit:-0} ))
echo "$(date) holder $JOB_ID on $HOSTNAME: $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total \
  --format=csv,noheader); ${limit}s, idle $IDLE, min left $MIN_LEFT"

idle_since=$(date +%s)
while true; do
  now=$(date +%s)
  [ -e $Q/stop ] && { echo "$(date) stop file found"; break; }
  (( end - now < MIN_LEFT )) && { echo "$(date) under ${MIN_LEFT}s left"; break; }
  (( now - idle_since > IDLE )) && { echo "$(date) idle for ${IDLE}s"; break; }
  name=$(find $PENDING -maxdepth 1 -type f -printf "%f\n" | sort | head -1)
  if [ -z "$name" ] || ! mv "$PENDING/$name" "$Q/running/$name" 2>/dev/null; then
    sleep 10
    continue
  fi
  echo "$(date) start $name: $(cat "$Q/running/$name")"
  mkdir -p "artifacts/$name"
  cp "$Q/running/$name" "artifacts/$name/command.sh"
  # its own process group (setsid), so a cancel stops everything it started
  DVAE_JOB=$name setsid bash "$Q/running/$name" > /dev/null 2> "artifacts/$name/stderr.log" &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -e "$Q/cancel/$name" ]; then
      echo "$(date) cancel $name"
      rm -f "$Q/cancel/$name"
      kill -TERM -- -$pid 2>/dev/null
      for _ in $(seq 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done
      kill -KILL -- -$pid 2>/dev/null
    fi
    sleep 5
  done
  wait $pid
  code=$?
  echo $code > "artifacts/$name/exit"
  mv "$Q/running/$name" $Q/done/
  echo "$(date) end $name: exit $code"
  idle_since=$(date +%s)
done
