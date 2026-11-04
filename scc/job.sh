#!/bin/bash -l
# One GPU job on BU's SCC. From the repo there (the venv sits beside it):
#   qsub [-N name] [-v DVAE_JOB=folder] [-l h_rt=HH:MM:SS] [-l gpu_type=L40S] scc/job.sh train_walk.py ...
# The run's files go to artifacts/$DVAE_JOB/ (default: the job's name and ID), beside its stderr and the GPU used.
#$ -P cs599bg
#$ -N dvae
#$ -cwd
#$ -j y
#$ -o artifacts/
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -l h_rt=12:00:00

module load python3/3.13.8
export DVAE_JOB=${DVAE_JOB:-$JOB_NAME-$JOB_ID}
mkdir -p "artifacts/$DVAE_JOB"
{
  echo "$(date) job $JOB_ID on $HOSTNAME: $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=name,memory.total --format=csv,noheader)"
  echo "python $*"
} > "artifacts/$DVAE_JOB/job.txt"
../venv/bin/python "$@" > /dev/null 2> "artifacts/$DVAE_JOB/stderr.log"
echo $? > "artifacts/$DVAE_JOB/exit"
