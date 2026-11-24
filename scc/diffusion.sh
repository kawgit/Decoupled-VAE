#!/bin/bash -l
# Submit from the SCC repository root:
# mkdir -p artifacts
# qsub -v DVAE_JOB=anand-cifar10-ddpm-s0 scc/diffusion.sh
# Resume (adds up to another 23 training hours, total step cap unchanged):
# qsub -v DVAE_JOB=anand-cifar10-ddpm-s0,RESUME=artifacts/anand-cifar10-ddpm-s0/latest.pt scc/diffusion.sh
#$ -P cs599bg
#$ -N cifar-ddpm
#$ -cwd
#$ -j y
#$ -o artifacts/
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -l h_rt=24:00:00

set -euo pipefail
module load python3/3.13.8
export DVAE_JOB=${DVAE_JOB:-cifar10-ddpm-$JOB_ID}
export DVAE_REQUIRE_CUDA=1
export OMP_NUM_THREADS=${NSLOTS:-4}
RUN_DIR="artifacts/$DVAE_JOB"
mkdir -p "$RUN_DIR"
PYTHON_BIN="$(cd .. && pwd)/venv/bin/python"
trap 'code=$?; echo "$code" > "$RUN_DIR/exit"' EXIT
nvidia-smi > "$RUN_DIR/gpu.txt"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1, "baseline requires one visible GPU"'
cp scc/diffusion.sh "$RUN_DIR/submission.sh"
flags=(--device cuda --precision bf16 --out "$RUN_DIR" --max-hours 23 --max-steps 200000)
if [ -n "${RESUME:-}" ]; then
    flags+=(--resume "$RESUME")
fi
"$PYTHON_BIN" -u train_diffusion.py "${flags[@]}" >> "$RUN_DIR/stdout.log" 2>> "$RUN_DIR/stderr.log"
