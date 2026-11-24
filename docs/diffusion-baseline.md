# CIFAR-10 diffusion baseline: one L40S, up to 24 hours

This is an unconditional pixel-space DDPM, trained from scratch, separate from the tree models. It is a
local implementation inspired by [DDPM](https://github.com/hojonathanho/diffusion) and uses the simple
epsilon-prediction MSE objective. It is **not** a reproduction of a published CIFAR-10 FID. DDIM previews
follow the deterministic update described in [DDIM](https://github.com/ermongroup/ddim).

## Fixed baseline configuration

| Setting | Value |
|---|---|
| Data | Project's seed-0 CIFAR-10 split: 45,000 train / 5,000 validation / 10,000 test |
| Input | RGB 32x32 pixels, scaled to [-1,1]; random horizontal flips during training |
| Conditioning | None; no labels, pretrained generator, outer autoencoder or tree |
| U-Net | Base width 128, multipliers 1/2/2/2, 2 residual blocks per level, attention at 16x16 and bottleneck |
| Parameters | 35,746,307 |
| Diffusion | 1,000 steps, linear beta from 0.0001 to 0.02, fixed posterior variance |
| Objective | Uniformly sampled timestep, predict injected Gaussian noise with MSE |
| Optimization | Adam, batch 128, LR 0.0002, 5,000-step warmup, gradient clipping 1 |
| Regularization | Dropout 0.1; EMA decay 0.9999 |
| Precision | bf16 on one L40S, FP32 parameters and optimizer |
| Budget | Stop at 200,000 optimizer steps or 23 invocation hours, whichever comes first; scheduler allocation 24 hours |
| Validation | Fixed 1,024 images from validation, fixed sampled timesteps/noise; EMA denoising MSE |
| Preview | Fixed seed 2026, 64 images, 100-step deterministic DDIM every 10,000 steps |
| Final grid | EMA, full 1,000-step ancestral DDPM; no quality guarantee before measuring |

The first run uses seed 0. The denoising validation loss is not pixel reconstruction MSE, FID, or a likelihood
bound. It selects `best.pt`, but does not establish best perceptual quality. Retain `final.pt` too. The model
is trained on pixels; it is a generation baseline, not an equivalent tree compressor or rate-distortion model.

## First completed run (reviewed November 24, 2026)

`artifacts/anand-cifar10-ddpm-s0`, SCC job 8892731, finished with exit 0 and an empty stderr log. It reached
200,000 steps in 24,047.5 seconds (6.68 L40S hours), stopping at the step cap rather than the 23-hour time cap.
The final/best fixed-noise EMA validation epsilon MSE was 0.0267463. The manifest confirms 45,000 training
images, 1,024 validation examples, one NVIDIA L40S, bf16, and 35,746,307 parameters.

The final 64-image ancestral DDPM grid contains recognizable and varied CIFAR-like animals and vehicles,
with some blur and malformed details. This is one training seed, not a claim of state-of-the-art quality or
absence of memorization.

The completed evaluation reports **FID 11.7073062**: 10,000 EMA samples, full 1,000-step ancestral DDPM,
seed 2026, versus the 10,000 CIFAR-10 test images, Clean-FID 0.1.35 clean mode (SciPy 1.15.3, NumPy 2.2.6).
The original evaluation generated every image but failed in the FID library; the scoring-only retry reused
those exact PNGs and completed with exit 0. Results and the checkpoint hash are in
`artifacts/anand-cifar10-ddpm-s0/eval-fid10k/results.json`. The scoring retry's timing excludes the earlier
image-generation phase. Compare tree samples using this same sample count, reference split and preprocessing;
this is not the published 50k-training-statistics FID protocol.

## Submit on SCC

From the remote repository root, after copying the new files there:

```bash
mkdir -p artifacts
qsub -v DVAE_JOB=anand-cifar10-ddpm-s0 scc/diffusion.sh
qstat -u "$USER"
tail -F artifacts/anand-cifar10-ddpm-s0/run.log
```

This is a standalone single-GPU job, not an item for an existing holder. It requests an L40S and 24 hours.
Do not submit it twice. A queued state is `qw`; the directory and log appear when the job starts. The script
uses the same `../venv/bin/python` and Python module as the other SCC scripts. No additional dependencies
are needed for training beyond the project's torch/torchvision environment. It does not assume SCC has Git
metadata. It captures the source files and manifest even in the sync-only SCC copy.

Checkpoints (`latest.pt`, `best.pt`, `final.pt`) include model, EMA, Adam state, mixed-precision scaler, step,
RNG states and validation history. They are written atomically. `latest.pt` is saved every 5,000 steps, on
SIGINT/SIGTERM after the current step, and on reaching the wall-clock cap. SIGKILL cannot be handled; the last
completed checkpoint still survives. Disk requirements are several checkpoint files of roughly 0.6 GB each.

Useful artifacts:

- `run.log`, `stdout.log`, `stderr.log`: progress and errors.
- `records.json`: validation curve, training step and run manifest; `run.png`: curve and latest sample grid.
- `manifest_step*.json`, `source_step*/`: environment, data split, settings and source provenance.
- `samples/step*_ddim.png`: fast previews; `samples/step*_ddpm.png`: full ancestral samples.
- `gpu.txt`, `submission.sh`, `exit`: allocation details, submitted script and process status.

## Resume

Unlike the existing experimental prior trainers, this trainer supports full resume. Locally or inside a
scheduled allocation:

```bash
python train_diffusion.py --resume artifacts/anand-cifar10-ddpm-s0/latest.pt --max-hours 23
```

For an additional SCC allocation, after the original job has ended:

```bash
qsub -v DVAE_JOB=anand-cifar10-ddpm-s0,RESUME=artifacts/anand-cifar10-ddpm-s0/latest.pt scc/diffusion.sh
```

This requests additional GPU time; it is not part of the first 24-hour budget. `--max-steps` is the total
target, not additional steps. Immutable model/training settings cannot be changed on resume. On CUDA,
bit-for-bit replay is not guaranteed across hardware or nondeterministic kernels. Do not read a checkpoint
while another process replaces it if you need a fixed evaluation target; use a copied, stable checkpoint.

## Sampling and fair quantitative comparison

First inspect an EMA sample grid with no extra metric dependency:

```bash
python evaluate_diffusion.py --ckpt artifacts/anand-cifar10-ddpm-s0/final.pt \
  --out artifacts/anand-cifar10-ddpm-s0/eval-ddpm64 --count 64 --device cuda
```

For quick previews only, add `--sampler ddim --sampling-steps 100`. Always record the sampler and step count;
a DDIM grid is not the full DDPM generation protocol.

For FID, install the optional [Clean-FID](https://github.com/GaParmar/clean-fid) dependency in the environment
used for evaluation (its Inception weights download on first use; compute nodes may not have internet access):

```bash
python -m pip install clean-fid 'scipy<1.18'
python evaluate_diffusion.py --ckpt artifacts/anand-cifar10-ddpm-s0/final.pt \
  --out artifacts/anand-cifar10-ddpm-s0/eval-ddpm10k --count 10000 --fid --device cuda
```

Run sampling/FID inside a GPU allocation, not on the SCC login node. Sampling and FID take additional GPU
time; they are not automatically submitted by the training job. The first 24-hour job includes training,
periodic previews and a final 64-image grid only.

`scc/diffusion_eval.sh` submits a separate two-hour single-L40S allocation for the 10,000-sample ancestral
DDPM evaluation. It expects Clean-FID and its missing dependencies in the baseline run's `metric-deps/`
and cached Inception weights in `metric-cache/`; those can be prepared without changing the shared venv.
Its progress is `eval.stdout.log`, errors are `eval.stderr.log`, completion is `eval-exit`, and the final
metrics are `eval-ddpm10k/results.json`. Refusing an existing evaluation directory prevents accidentally
combining different sample sets.

Clean-FID 0.1.35 uses `scipy.linalg.sqrtm(..., disp=False)`, removed in SciPy 1.18; use SciPy 1.15.3
(Python 3.13 on SCC, paired with NumPy 2.2.6) or another compatible version below 1.18 for this metric protocol. Keep the metric
dependencies in the run's isolated directory. The evaluator saves provenance before invoking FID and saves
the exception if scoring fails, so generated images do not need to be regenerated. To score existing samples,
use `--images` with a new `--out`; optionally pass `--sample-metadata` to retain checkpoint/sampler provenance.

Evaluation writes individual generated PNGs, `grid.png`, and `results.json` with checkpoint hash, seed,
training step, sample count, sampler, reference count, metric-library version and timings. FID uses all
10,000 CIFAR-10 test images explicitly, Clean-FID's clean preprocessing, and EMA-generated samples. It is
**not** directly comparable to published 50k-sample FID against training-set statistics. For tuning use
`--reference val`, reserving the test comparison for final selected models. Keep sample count, resolution,
reference split, preprocessing and sampling settings fixed across compared models.

The same evaluator can score a tree's individually exported RGB 32x32 PNG samples, not its montage:

```bash
python evaluate_diffusion.py --images /path/to/tree_individual_pngs \
  --count 10000 --out artifacts/tree-fid10k --fid --device cuda
```

FID is a generation-quality metric, not code bits or reconstruction error. Report FID, sample grids,
training GPU hours, parameter count and sampling time separately from the tree's code cost. This baseline
does not implement diffusion likelihood estimation or image compression; do not compare its epsilon MSE
numerically with the tree's bits/image. Strong conclusions require repeated training seeds and fixed budgets.
