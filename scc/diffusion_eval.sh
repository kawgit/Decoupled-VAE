#!/bin/bash -l
# A separate evaluation allocation; generation and FID do not run on the login node.
# Metric dependencies and Inception weights must be prepared before submission.
# qsub scc/diffusion_eval.sh
#$ -P cs599bg
#$ -N ddpm-fid10k
#$ -cwd
#$ -j y
#$ -o artifacts/
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -l h_rt=02:00:00

set -euo pipefail
module load python3/3.13.8
export OMP_NUM_THREADS=${NSLOTS:-4}
export DVAE_REQUIRE_CUDA=1
BASELINE_RUN=${BASELINE_RUN:-artifacts/anand-cifar10-ddpm-s0}
export PYTHONPATH="$PWD/$BASELINE_RUN/metric-deps${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$PWD/$BASELINE_RUN/metric-cache"
export MPLCONFIGDIR="$PWD/$BASELINE_RUN/matplotlib-cache"
PYTHON_BIN="$(cd .. && pwd)/venv/bin/python"
trap 'code=$?; echo "$code" > "$BASELINE_RUN/eval-exit"' EXIT
if [ "${SCORE_ONLY:-0}" = 1 ]; then
    # Recover metrics without repeating expensive generation if an optional metric library failed.
    "$PYTHON_BIN" -u evaluate_diffusion.py \
      --images "$BASELINE_RUN/eval-ddpm10k/generated" \
      --sample-metadata "$BASELINE_RUN/eval-ddpm10k/provenance.json" \
      --out "$BASELINE_RUN/eval-fid10k" \
      --count 10000 --batch 64 --reference test --device cuda --fid \
      > "$BASELINE_RUN/score.stdout.log" 2> "$BASELINE_RUN/score.stderr.log"
else
    "$PYTHON_BIN" -u evaluate_diffusion.py \
      --ckpt "$BASELINE_RUN/final.pt" \
      --out "$BASELINE_RUN/eval-ddpm10k" \
      --count 10000 --batch 64 --sampler ddpm --precision bf16 \
      --reference test --seed 2026 --device cuda --fid \
      > "$BASELINE_RUN/eval.stdout.log" 2> "$BASELINE_RUN/eval.stderr.log"
fi
