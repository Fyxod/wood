"""Run orchestration for WOOD smoke and 150-iteration jobs."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .cases import RunSpec, build_matrix, print_resolved_cases, resolve_image_path
from .logging import append_jsonl, nvidia_smi_memory_gb, read_json, write_csv, write_json
from .metrics import (
    delta_to_pil,
    flow_to_pil,
    image_metrics,
    make_blank_like,
    pil_to_tensor,
    save_sheet,
    tensor_pair_metrics,
    tensor_to_pil,
)
from .runtime import torch_device, torch_peak_gb


@dataclass
class RunConfig:
    mat_root: str
    output_root: str
    iters: int
    blank_value: float = 0.0
    lr: float = 0.05
    seed: int = 1234
    init: str | None = None
    unet_backward_scale: float = 8192.0
    geometry_config_path: str | None = None
    quick: bool = False
    all_cases: bool = False
    mode: str = "smoke_timing"
    skip_final_edits: bool = False


REQUIRED_HISTORY_FIELDS = {
    "iter",
    "Z",
    "scaled_Z",
    "loss",
    "objective_name",
    "model_name",
    "face_id",
    "prompt",
    "seed",
    "blank_value",
    "backward_scale",
    "Z_to_blank_objective",
    "psnr_to_original",
    "ssim_to_original",
    "mse_to_original",
    "l2_to_original",
    "psnr_to_blank",
    "ssim_to_blank",
    "mse_to_blank",
    "combined_max_disp_px",
    "combined_mean_disp_px",
    "combined_p95_disp_px",
    "jacobian_det_min",
    "foldover_fraction",
    "smoothness_tv",
    "tps_mean_disp",
    "tps_max_disp",
    "delaunay_mean_disp",
    "delaunay_max_disp",
    "rolling_mean_disp",
    "rolling_max_disp",
    "dct_mean_disp",
    "dct_max_disp",
    "fft_phase_norm",
    "fft_phase_mean_abs",
    "fft_phase_max_abs",
    "legacy_fft_strength_equivalent",
    "fft_spatial_delta_mse",
    "num_total_params",
    "num_clamped_total",
    "fraction_clamped_total",
    "seconds_iter",
    "seconds_elapsed",
}


def _run_dir(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / "blank_objective_ref" / spec.model / spec.objective / spec.case.slug


def _component_flow_images(aux: dict[str, Any], out_dir: Path, scale_px: float) -> None:
    flow_to_pil(aux["displacement"], scale_px).save(out_dir / "combined_flow.png")
    for name, field in aux["fields"].items():
        flow_to_pil(field, scale_px).save(out_dir / f"{name}_flow.png")
    delta_to_pil(aux["fft_delta"]).save(out_dir / "fft_phase_visualization.png")


def _float_terms(terms: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in terms.items():
        if hasattr(value, "detach"):
            out[key] = float(value.detach().float().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _history_fields_ok(row: dict[str, Any]) -> bool:
    return REQUIRED_HISTORY_FIELDS.issubset(set(row))


def _backend(device):
    from wood.backends.instruct import InstructBackend

    return InstructBackend(device)


def optimize_one(spec: RunSpec, cfg: RunConfig, backend, device, output_dir: Path) -> dict[str, Any]:
    import torch

    from wood.core.geometry.combined_wood import CombinedWoodPerturbation, WoodGeometryConfig, load_wood_geometry_config
    from wood.core.losses import wood_loss

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    mat_root = Path(cfg.mat_root)
    image_path = resolve_image_path(mat_root, spec.case.face_id)
    print(f"[wood] running {spec.slug} image={image_path}")

    original = Image.open(image_path).convert("RGB")
    blank = make_blank_like(original, cfg.blank_value)
    original.save(output_dir / "original.png")
    blank.save(output_dir / "blank.png")
    original_tensor = pil_to_tensor(original, device)
    blank_tensor = pil_to_tensor(blank, device)

    reference = backend.prepare_blank_reference(original_tensor, blank_tensor, spec.case.prompt, spec.objective)
    geometry_config = load_wood_geometry_config(cfg.geometry_config_path) if cfg.geometry_config_path else WoodGeometryConfig()
    if cfg.init:
        geometry_config.init = cfg.init
    geometry = CombinedWoodPerturbation(
        original_tensor.shape[-2],
        original_tensor.shape[-1],
        original_tensor.shape[1],
        device,
        seed=spec.seed,
        config=geometry_config,
    )
    optimizer = torch.optim.Adam(geometry.parameters(), lr=cfg.lr)
    projection = geometry.project_()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    write_json(
        output_dir / "config_resolved.json",
        {
            **asdict(cfg),
            "spec": {
                "variant": "blank_objective_ref",
                "model": spec.model,
                "objective": spec.objective,
                "face_id": spec.case.face_id,
                "prompt": spec.case.prompt,
                "seed": spec.seed,
                "image_path": str(image_path),
            },
            "loss": "loss = -Z",
            "Z_definition": "Z = MSE(selected_instruct_objective(perturbed), selected_instruct_objective(blank))",
            "blank_value": cfg.blank_value,
            "geometry_config_path": cfg.geometry_config_path,
            "geometry_config_resolved": geometry_config.__dict__.copy(),
            "geometry_limits": geometry.limits_dict(),
            "no_visual_counter_loss": True,
            "model_weights_frozen": True,
        },
    )

    for iteration in range(1, cfg.iters + 1):
        iter_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        perturbed, aux = geometry(original_tensor)
        Z, terms = backend.compute_Z(perturbed, reference, spec.objective)
        scaled_Z = Z
        loss = wood_loss(Z)
        finite = bool(torch.isfinite(loss).item() and torch.isfinite(perturbed).all().item())
        if not finite:
            raise FloatingPointError(f"Non-finite Z/loss at iteration {iteration}")
        backward_scale = float(cfg.unet_backward_scale) if spec.objective == "unet_prediction" else 1.0
        (loss * backward_scale).backward()
        if backward_scale != 1.0:
            for parameter in geometry.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(backward_scale)
        grad_norms = geometry.grad_norms()
        optimizer.step()
        projection = geometry.project_()

        with torch.no_grad():
            metrics_original = tensor_pair_metrics(perturbed, original_tensor, prefix="")
            metrics_blank = tensor_pair_metrics(perturbed, blank_tensor, prefix="")

        seconds_iter = time.monotonic() - iter_started
        row: dict[str, Any] = {
            "iter": iteration,
            "Z": float(Z.detach().float().cpu()),
            "scaled_Z": float(scaled_Z.detach().float().cpu()),
            "loss": float(loss.detach().float().cpu()),
            "objective_name": spec.objective,
            "model_name": spec.model,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "seed": spec.seed,
            "blank_value": cfg.blank_value,
            "backward_scale": backward_scale,
            "iteration_count": cfg.iters,
            "seconds_iter": seconds_iter,
            "seconds_elapsed": time.monotonic() - started,
            "peak_vram_gb": torch_peak_gb(),
            "psnr_to_original": metrics_original["psnr"],
            "ssim_to_original": metrics_original["ssim"],
            "mse_to_original": metrics_original["mse"],
            "l2_to_original": metrics_original["l2"],
            "psnr_to_blank": metrics_blank["psnr"],
            "ssim_to_blank": metrics_blank["ssim"],
            "mse_to_blank": metrics_blank["mse"],
            "l2_to_blank": metrics_blank["l2"],
            **_float_terms(terms),
            **aux["diagnostics"],
            **grad_norms,
            **geometry.parameter_diagnostics(),
            **projection,
        }
        rows.append(row)
        append_jsonl(output_dir / "history.jsonl", row)
        if best is None or row["Z"] > best["row"]["Z"]:
            best = {
                "row": row,
                "theta_state": geometry.theta_state(),
                "perturbed": perturbed.detach().clone(),
            }

    if not rows or best is None:
        raise RuntimeError("No finite optimization iteration completed.")

    with torch.no_grad():
        final_perturbed_tensor, final_aux = geometry(original_tensor)

    final_perturbed = tensor_to_pil(final_perturbed_tensor)
    final_perturbed.save(output_dir / "perturbed.png")
    _component_flow_images(final_aux, output_dir, geometry.component_limit_for_flow)
    # Local replay/debug artifacts only. `.gitignore` excludes all `.pt` files
    # so these should not be pushed. The payload is intentionally theta-only,
    # not the full state_dict with large deterministic buffers.
    torch.save(geometry.theta_state(), output_dir / "theta_final.pt")
    torch.save(best["theta_state"], output_dir / "theta_best.pt")
    write_json(
        output_dir / "geometry_params_final.json",
        {
            "limits": geometry.limits_dict(),
            "parameter_diagnostics": geometry.parameter_diagnostics(),
            "last_projection": projection,
        },
    )
    write_json(
        output_dir / "geometry_params_best.json",
        {
            "best_iter_by_Z": best["row"]["iter"],
            "best_Z": best["row"]["Z"],
        },
    )

    if cfg.skip_final_edits:
        clean_edit = original.copy()
        perturbed_edit = final_perturbed.copy()
    else:
        clean_edit = backend.generate_edit(original, spec.case.prompt, spec.seed)
        perturbed_edit = backend.generate_edit(final_perturbed, spec.case.prompt, spec.seed)
    clean_edit.save(output_dir / "clean_edited.png")
    perturbed_edit.save(output_dir / "perturbed_edited.png")

    input_metrics = image_metrics(original, final_perturbed)
    output_metrics = image_metrics(clean_edit, perturbed_edit)
    save_sheet(
        output_dir / "comparison_sheet.png",
        [
            ("original", original),
            ("blank", blank),
            ("perturbed", final_perturbed),
            ("clean edit", clean_edit),
            ("perturbed edit", perturbed_edit),
            ("combined flow", Image.open(output_dir / "combined_flow.png")),
        ],
    )
    write_csv(output_dir / "history.csv", rows)

    elapsed = time.monotonic() - started
    final_row = rows[-1]
    summary = {
        "status": "done",
        "variant": "blank_objective_ref",
        "model": spec.model,
        "objective_name": spec.objective,
        "face_id": spec.case.face_id,
        "prompt": spec.case.prompt,
        "seed": spec.seed,
        "blank_value": cfg.blank_value,
        "iters": cfg.iters,
        "final_Z": final_row["Z"],
        "final_loss": final_row["loss"],
        "best_iter_by_Z": best["row"]["iter"],
        "best_Z": best["row"]["Z"],
        "mean_seconds_iter": float(sum(row["seconds_iter"] for row in rows) / max(len(rows), 1)),
        "elapsed_seconds": elapsed,
        "final_psnr_to_original": final_row["psnr_to_original"],
        "final_ssim_to_original": final_row["ssim_to_original"],
        "final_mse_to_original": final_row["mse_to_original"],
        "final_psnr_to_blank": final_row["psnr_to_blank"],
        "final_ssim_to_blank": final_row["ssim_to_blank"],
        "final_mse_to_blank": final_row["mse_to_blank"],
        "input_ssim": input_metrics["ssim"],
        "input_psnr": input_metrics["psnr"],
        "input_l2": input_metrics["l2"],
        "input_mse": input_metrics["mse"],
        "final_output_ssim": output_metrics["ssim"],
        "final_output_psnr": output_metrics["psnr"],
        "final_output_l2": output_metrics["l2"],
        "final_output_mse": output_metrics["mse"],
        "final_combined_max_disp_px": final_row["combined_max_disp_px"],
        "final_combined_mean_disp_px": final_row["combined_mean_disp_px"],
        "final_combined_p95_disp_px": final_row["combined_p95_disp_px"],
        "final_fraction_clamped_total": final_row["fraction_clamped_total"],
        "all_required_history_fields_populated": _history_fields_ok(final_row),
        "clamp_project_logic_active": final_row["num_total_params"] > 0,
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "run_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "DONE.json", {"status": "done", "elapsed_seconds": elapsed, "final_Z": final_row["Z"]})
    return summary


def _aggregate_summaries(run_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in run_root.glob("runs/blank_objective_ref/instruct/*/*/summary.json")]


def _time_estimates(rows: list[dict[str, Any]], wall_seconds: float, observed_iters: int) -> dict[str, Any]:
    by_objective: dict[str, list[float]] = {"vae_conditioning": [], "unet_prediction": []}
    for row in rows:
        objective = row.get("objective_name")
        if objective in by_objective:
            by_objective[objective].append(float(row.get("mean_seconds_iter", 0.0)))
    means = {key: sum(vals) / max(len(vals), 1) for key, vals in by_objective.items()}
    observed_iter_seconds = sum(float(row.get("mean_seconds_iter", 0.0)) for row in rows)
    fixed_overhead = max(0.0, float(wall_seconds) - float(observed_iters) * observed_iter_seconds)
    completed = max(len(rows), 1)
    scale_to_full = 8.0 / completed
    full_iter_seconds = observed_iter_seconds * scale_to_full
    full_overhead = fixed_overhead * scale_to_full
    return {
        "seconds_per_iteration_by_objective": means,
        "observed_completed_runs": len(rows),
        "estimated_full_matrix_seconds_per_iteration": full_iter_seconds,
        "estimated_fixed_overhead_seconds": full_overhead,
        "estimated_runtime_seconds_for_150_iterations": full_overhead + 150 * full_iter_seconds,
    }


def _write_top_summary(run_root: Path, cfg: RunConfig, started: float, summaries: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    wall = time.monotonic() - started
    status = "done" if not failures else "failed"
    estimates = _time_estimates(summaries, wall, cfg.iters)
    payload = {
        "status": status,
        "mode": cfg.mode,
        "variant": "blank_objective_ref",
        "iters": cfg.iters,
        "quick": cfg.quick,
        "all_cases": cfg.all_cases,
        "execution": "sequential",
        "wall_seconds": wall,
        "num_runs_attempted": len(summaries) + len(failures),
        "num_runs_completed": len(summaries),
        "num_failures": len(failures),
        "failures": failures,
        "summaries": summaries,
        "time_estimates": estimates,
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "all_per_iteration_logging_fields_populated": all(s.get("all_required_history_fields_populated", False) for s in summaries),
        "clamp_project_logic_active": all(s.get("clamp_project_logic_active", False) for s in summaries),
        "output_root": str(run_root),
    }
    write_json(run_root / "summary.json", payload)

    lines = [
        f"# WOOD {cfg.mode} summary",
        "",
        f"- status: {status}",
        "- variant: blank_objective_ref",
        "- execution: sequential",
        f"- iterations per run: {cfg.iters}",
        f"- runs attempted: {payload['num_runs_attempted']}",
        f"- runs completed: {payload['num_runs_completed']}",
        f"- failures: {payload['num_failures']}",
        f"- wall seconds: {wall:.2f}",
        f"- seconds/iter vae_conditioning: {estimates['seconds_per_iteration_by_objective'].get('vae_conditioning', 0):.3f}",
        f"- seconds/iter unet_prediction: {estimates['seconds_per_iteration_by_objective'].get('unet_prediction', 0):.3f}",
        f"- estimated 150-iteration full matrix: {estimates['estimated_runtime_seconds_for_150_iterations'] / 60:.1f} min",
        f"- peak VRAM GB: {payload.get('peak_vram_gb')}",
        f"- all required per-iteration fields populated: {payload['all_per_iteration_logging_fields_populated']}",
        f"- clamp/project logic active: {payload['clamp_project_logic_active']}",
        "",
    ]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure.get('spec')}: {failure.get('error')}")
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_matrix(cfg: RunConfig) -> dict[str, Any]:
    started = time.monotonic()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    label = "quick" if cfg.quick else "all"
    root = Path(cfg.output_root) / f"{run_id}_blank_objective_ref_{label}_sequential"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "launcher_config.json", asdict(cfg))
    print_resolved_cases(Path(cfg.mat_root))
    specs = build_matrix(quick=cfg.quick)
    if cfg.all_cases:
        specs = build_matrix(quick=False)
    device = torch_device()
    backend = _backend(device)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = _run_dir(root, spec)
        try:
            summaries.append(optimize_one(spec, cfg, backend, device, run_dir))
        except Exception as error:
            failures.append({"spec": spec.slug, "error": repr(error), "run_dir": str(run_dir)})
            write_json(run_dir / "FAILED.json", {"status": "failed", "error": repr(error)})
    return _write_top_summary(root, cfg, started, summaries, failures)
