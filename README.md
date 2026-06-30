# WOOD

WOOD is a compact InstructPix2Pix-only white-box geometry experiment.

The optimized scalar is called `Z`, and every optimization run uses:

```text
loss = -Z
```

There is no visual-MSE counter-loss, no original/blank visual-reference variant matrix, and no FLUX backend in this repository.

## Objective

For an original image `x`, a differentiable geometric perturbation `T_theta`, and a blank image `b` with the same spatial size:

```text
x_p = T_theta(x)
Z = distance(objective(x_p), objective(b))
loss = -Z
```

The blank image is the objective-space reference. The original image is used as the source image and for diagnostics such as PSNR/SSIM, but it is not used as the reference in `Z`.

The two supported InstructPix2Pix objectives are:

- `vae_conditioning`
- `unet_prediction`

For each run, the blank objective representation is computed once with fixed prompt/settings/noise/timestep, then the perturbed representation is compared to that blank reference at every iteration.

## White-box constraints

WOOD freezes all InstructPix2Pix model weights and optimizes only differentiable geometry parameters with Adam. It does not optimize pixels, add pixel noise, add adversarial patches, train/fine-tune InstructPix2Pix, use LoRA, or run black-box search.

Gradients flow through:

```text
Instruct internal objective
→ perturbed image
→ differentiable geometry
→ geometry parameters
```

## Geometry

All required geometry components are active together in one combined perturbation module:

- TPS / Thin Plate Spline
- Delaunay / piecewise affine
- FFT phase
- Rolling shutter
- DCT low-frequency warp

TPS, Delaunay/piecewise affine, rolling shutter, and DCT produce spatial displacement fields that are summed and applied through `grid_sample`. FFT phase is a differentiable frequency-domain stage applied after the spatial warp.

Elastic was not included in the first WOOD implementation because the current differentiable GLASS-style module set does not include a ready PyTorch elastic component.

The loss is only `loss = -Z`. Geometry is constrained by hard projection/clamping after optimizer steps. Displacement, smoothness, foldover, visual distances, and boundary saturation are logged as diagnostics only.

For fp16 numerical stability, the `unet_prediction` backward pass uses fixed gradient scaling and then unscales geometry gradients before Adam. The logged scalar remains `Z`, and the logged loss remains exactly `loss = -Z`.

Default hard projection ranges are mapped from the old geometric-v1 strength-style ranges:

| Component | Parameter clamp |
| --- | --- |
| TPS | `[-0.007, +0.007]` normalized coordinate equivalent |
| Delaunay | `[-0.010, +0.010]` normalized coordinate equivalent |
| Rolling shutter | `[-0.009, +0.009]` normalized coordinate equivalent |
| DCT | `[-0.008, +0.008]` normalized coordinate equivalent |
| FFT phase | `[-pi, +pi]` direct phase delta |

For 512×512 inputs these correspond roughly to 3.6 px, 5.1 px, 4.6 px, and 4.1 px for the spatial components. FFT logs `legacy_fft_strength_equivalent = mean_abs_phase_delta / pi * 1_000_000`.

## Cases

WOOD uses the same four image/prompt cases as GLASS:

- `face_002` + `add black sunglasses`
- `face_002` + `add headphones`
- `face_005` + `add black sunglasses`
- `face_005` + `add headphones`

Images are read from MAT via `--mat-root`, normally `/home/interns/Desktop/mat`. The expected InstructPix2Pix input is `data/<face_id>/instruct_512.png`; if needed, the resolver falls back to another sensible image inside the face folder and prints what was chosen.

## Smoke timing

Quick smoke:

```bash
cd /home/interns/Desktop/wood
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m wood.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --iters 2 \
  --quick
```

All-case smoke:

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m wood.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --iters 2 \
  --all-cases
```

## Full 150-iteration run

Run this only after smoke succeeds:

```bash
cd /home/interns/Desktop/wood
mkdir -p logs

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m wood.scripts.run_matrix \
  --mat-root /home/interns/Desktop/mat \
  --iters 150 \
  --output-root outputs/blank_objective_ref \
  2>&1 | tee logs/wood_blank_objective_ref_150.log
```

## Summaries

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m wood.scripts.summarize_runs \
  --results-root outputs/blank_objective_ref \
  --output-root outputs/reports/blank_objective_ref
```

Summary outputs include aggregate CSVs, per-run final values, metric curves, an image index, and a lightweight HTML/Markdown report.

## Output layout

```text
outputs/
  smoke_timing/
  blank_objective_ref/
  reports/
```

Each run saves `history.csv`, `history.jsonl`, `config_resolved.json`, `summary.json`, `DONE.json` or `FAILED.json`, final images, flow visualizations, final/best parameter tensors, and a comparison sheet.

The `theta_final.pt` and `theta_best.pt` files are local replay/debug artifacts only. They are ignored by Git and should not be pushed. These files contain only compact trainable theta tensors plus metadata, not large deterministic interpolation buffers.
