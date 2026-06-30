"""Build a GLASS-style WOOD report from the latest completed run."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageOps


TITLE = "WOOD: InstructPix2Pix White-box Geometry Results"
SUBTITLE = "Combined differentiable perturbation results"
AUTHOR = "Parth Katiyar"
OUTPUT_BASENAME = "wood_report"

# Optional default run folder. Leave empty to auto-select the latest completed
# folder under outputs/blank_objective_ref. You can also override this from the
# CLI with: --run-folder 20260630_070226_blank_objective_ref_all_sequential
RUN_FOLDER_NAME = "20260630_083256_blank_objective_ref_all_sequential"

# Default report quality. Keep this True for the compact report currently used
# in the repo. Set to False, or pass --no-compress-report, to generate a larger
# high-quality report.
COMPRESS_REPORT = False

COMPRESSED_QUALITY = {
    "image_size": (420, 420),
    "strip_format": "jpg",
    "strip_quality": 82,
    "graph_dpi": 120,
    "compress_pdf_images": True,
    "pdf_image_quality": 68,
}

HIGH_QUALITY = {
    "image_size": (512, 512),
    "strip_format": "png",
    "strip_quality": None,
    "graph_dpi": 150,
    "compress_pdf_images": False,
    "pdf_image_quality": 95,
}

OBJECTIVE_NAMES = {
    "vae_conditioning": "VAE conditioning latent",
    "unet_prediction": "UNet denoising prediction",
}

OBJECTIVE_ORDER = {
    "vae_conditioning": 0,
    "unet_prediction": 1,
}

CASE_ORDER = {
    ("face_002", "add black sunglasses"): 0,
    ("face_002", "add headphones"): 1,
    ("face_005", "add black sunglasses"): 2,
    ("face_005", "add headphones"): 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the WOOD report from existing outputs.")
    parser.add_argument("--results-root", default="outputs/blank_objective_ref")
    parser.add_argument("--run-folder", default=RUN_FOLDER_NAME, help="Folder name under results-root. Falls back to latest if missing.")
    parser.add_argument("--run-root", default=None, help="Specific run root path. Overrides --run-folder.")
    parser.add_argument("--output-root", default="outputs/report_wood")
    parser.add_argument("--no-pdf", action="store_true")
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument("--compress-report", dest="compress_report", action="store_true", help="Generate compact report assets and a small PDF.")
    quality.add_argument("--no-compress-report", dest="compress_report", action="store_false", help="Generate high-quality report assets and a larger PDF.")
    parser.set_defaults(compress_report=COMPRESS_REPORT)
    return parser.parse_args()


def quality_settings(compress_report: bool) -> dict[str, Any]:
    return dict(COMPRESSED_QUALITY if compress_report else HIGH_QUALITY)


def slug(value: str) -> str:
    keep = []
    for char in value.lower():
        if char.isalnum():
            keep.append(char)
        elif char in {" ", "-", "_", "/", "."}:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    return f"{number:.{digits}f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_run_root(results_root: Path) -> Path:
    candidates = []
    for child in sorted(results_root.iterdir()) if results_root.exists() else []:
        if not child.is_dir():
            continue
        summaries = list(child.glob("runs/blank_objective_ref/instruct/*/*/summary.json"))
        top_summary = child / "summary.json"
        if summaries and top_summary.exists():
            payload = read_json(top_summary)
            if payload.get("status") == "done":
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No completed WOOD run roots found under {results_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_run_root(results_root: Path, run_root_arg: str | None, run_folder_name: str | None) -> Path:
    if run_root_arg:
        path = Path(run_root_arg)
        if path.exists():
            return path
        print(f"[wood-report] requested run root not found, falling back to latest: {path}")
        return find_latest_run_root(results_root)
    if run_folder_name:
        path = results_root / run_folder_name
        if path.exists():
            return path
        print(f"[wood-report] requested run folder not found, falling back to latest: {path}")
    return find_latest_run_root(results_root)


def _history_final(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    for key, value in rows[-1].items():
        number = to_float(value)
        out[key] = number if number is not None else value
    return out


def collect_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for summary_path in sorted(run_root.glob("runs/blank_objective_ref/instruct/*/*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config_path = run_dir / "config_resolved.json"
        config = read_json(config_path) if config_path.exists() else {}
        history_path = run_dir / "history.csv"
        history_rows = read_csv_rows(history_path) if history_path.exists() else []
        final = _history_final(history_rows)
        spec = config.get("spec", {})
        objective = str(summary.get("objective_name") or spec.get("objective") or run_dir.parent.name)
        face_id = str(summary.get("face_id") or spec.get("face_id") or run_dir.name.split("__")[0])
        prompt = str(summary.get("prompt") or spec.get("prompt") or "")
        images = {
            "original": run_dir / "original.png",
            "reference": run_dir / "blank.png",
            "perturbed": run_dir / "perturbed.png",
            "clean_edited": run_dir / "clean_edited.png",
            "perturbed_edited": run_dir / "perturbed_edited.png",
            "comparison_sheet": run_dir / "comparison_sheet.png",
        }
        for label, path in images.items():
            if not path.exists():
                missing.append({"case": f"{face_id} / {prompt}", "objective": objective, "artifact": label, "path": str(path)})
        run = {
            "objective": objective,
            "objective_name": OBJECTIVE_NAMES.get(objective, objective),
            "face_id": face_id,
            "prompt": prompt,
            "case": f"{face_id} / {prompt}",
            "case_slug": slug(f"{face_id}_{prompt}"),
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "history_path": str(history_path),
            "history_rows": history_rows,
            "summary": summary,
            "config": config,
            "final": {
                "final_Z": summary.get("final_Z", final.get("Z")),
                "best_Z": summary.get("best_Z"),
                "best_iter_by_Z": summary.get("best_iter_by_Z"),
                "final_loss": summary.get("final_loss", final.get("loss")),
                "input_ssim": summary.get("input_ssim"),
                "input_psnr": summary.get("input_psnr"),
                "input_l2": summary.get("input_l2"),
                "final_ssim_to_original": summary.get("final_ssim_to_original", final.get("ssim_to_original")),
                "final_psnr_to_original": summary.get("final_psnr_to_original", final.get("psnr_to_original")),
                "final_mse_to_original": summary.get("final_mse_to_original", final.get("mse_to_original")),
                "final_output_ssim": summary.get("final_output_ssim"),
                "final_output_psnr": summary.get("final_output_psnr"),
                "final_output_l2": summary.get("final_output_l2"),
                "final_output_mse": summary.get("final_output_mse"),
                "combined_mean_disp_px": final.get("combined_mean_disp_px", summary.get("final_combined_mean_disp_px")),
                "combined_p95_disp_px": final.get("combined_p95_disp_px", summary.get("final_combined_p95_disp_px")),
                "combined_max_disp_px": final.get("combined_max_disp_px", summary.get("final_combined_max_disp_px")),
                "jacobian_det_min": final.get("jacobian_det_min"),
                "foldover_fraction": final.get("foldover_fraction"),
                "smoothness_tv": final.get("smoothness_tv"),
                "fraction_clamped_total": final.get("fraction_clamped_total", summary.get("final_fraction_clamped_total")),
                "seconds_elapsed": summary.get("elapsed_seconds"),
                "seconds_per_iter": summary.get("mean_seconds_iter"),
                "peak_vram_gb": summary.get("peak_vram_gb"),
                "tps_mean_disp": final.get("tps_mean_disp"),
                "tps_max_disp": final.get("tps_max_disp"),
                "delaunay_mean_disp": final.get("delaunay_mean_disp"),
                "delaunay_max_disp": final.get("delaunay_max_disp"),
                "rolling_mean_disp": final.get("rolling_mean_disp"),
                "rolling_max_disp": final.get("rolling_max_disp"),
                "dct_mean_disp": final.get("dct_mean_disp"),
                "dct_max_disp": final.get("dct_max_disp"),
                "fft_phase_norm": final.get("fft_phase_norm"),
                "fft_phase_max_abs": final.get("fft_phase_max_abs"),
                "fft_spatial_delta_mse": final.get("fft_spatial_delta_mse"),
            },
            "images": images,
        }
        runs.append(run)
    runs.sort(key=lambda r: (OBJECTIVE_ORDER.get(r["objective"], 99), CASE_ORDER.get((r["face_id"], r["prompt"]), 99)))
    return runs, missing


def run_matrix_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_obj[run["objective"]].append(run)
    for objective, group in sorted(by_obj.items(), key=lambda item: OBJECTIVE_ORDER.get(item[0], 99)):
        rows.append(
            {
                "model": "InstructPix2Pix",
                "code_objective": objective,
                "report_objective_name": OBJECTIVE_NAMES.get(objective, objective),
                "num_cases": len(group),
                "iterations": group[0]["summary"].get("iters", len(group[0]["history_rows"])),
                "status": "done" if all((Path(r["run_dir"]) / "DONE.json").exists() for r in group) else "incomplete",
            }
        )
    return rows


def aggregate_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_obj[run["objective"]].append(run)
    metrics = [
        "final_Z",
        "final_loss",
        "final_ssim_to_original",
        "final_psnr_to_original",
        "final_output_ssim",
        "final_output_l2",
        "combined_max_disp_px",
        "combined_p95_disp_px",
        "fraction_clamped_total",
    ]
    for objective, group in sorted(by_obj.items(), key=lambda item: OBJECTIVE_ORDER.get(item[0], 99)):
        row: dict[str, Any] = {
            "model": "InstructPix2Pix",
            "code_objective": objective,
            "report_objective_name": OBJECTIVE_NAMES.get(objective, objective),
            "num_runs": len(group),
        }
        for metric in metrics:
            values = [to_float(run["final"].get(metric)) for run in group]
            values = [v for v in values if v is not None]
            row[f"mean_{metric}"] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def per_run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        final = run["final"]
        rows.append(
            {
                "model": "InstructPix2Pix",
                "objective": run["objective_name"],
                "code_objective": run["objective"],
                "face_id": run["face_id"],
                "prompt": run["prompt"],
                "final_Z": final.get("final_Z"),
                "final_loss": final.get("final_loss"),
                "best_iter_by_Z": final.get("best_iter_by_Z"),
                "ssim_to_original": final.get("final_ssim_to_original"),
                "psnr_to_original": final.get("final_psnr_to_original"),
                "output_ssim": final.get("final_output_ssim"),
                "output_l2": final.get("final_output_l2"),
                "max_disp_px": final.get("combined_max_disp_px"),
                "p95_disp_px": final.get("combined_p95_disp_px"),
                "fraction_clamped": final.get("fraction_clamped_total"),
                "seconds_per_iter": final.get("seconds_per_iter"),
            }
        )
    return rows


def best_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("highest final Z", "final_Z", True),
        ("highest output L2", "final_output_l2", True),
        ("lowest output SSIM", "final_output_ssim", False),
        ("highest SSIM to original", "final_ssim_to_original", True),
    ]
    rows = []
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_obj[run["objective"]].append(run)
    for objective, group in sorted(by_obj.items(), key=lambda item: OBJECTIVE_ORDER.get(item[0], 99)):
        for label, key, high in specs:
            candidates = [(to_float(run["final"].get(key)), run) for run in group]
            candidates = [(v, r) for v, r in candidates if v is not None]
            if not candidates:
                continue
            value, run = sorted(candidates, key=lambda item: item[0], reverse=high)[0]
            rows.append(
                {
                    "metric": label,
                    "objective": OBJECTIVE_NAMES.get(objective, objective),
                    "case": run["case"],
                    "value": value,
                    "ssim_to_original": run["final"].get("final_ssim_to_original"),
                    "output_ssim": run["final"].get("final_output_ssim"),
                    "max_disp_px": run["final"].get("combined_max_disp_px"),
                }
            )
    return rows


def copy_or_placeholder(path: Path, size: tuple[int, int], label: str) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    img = Image.new("RGB", size, "#f3f4f6")
    draw = ImageDraw.Draw(img)
    draw.text((18, 18), f"missing: {label}", fill="black")
    return img


def make_input_difference_image(run: dict[str, Any], output_root: Path, size: tuple[int, int]) -> Path:
    """Create an amplified absolute input-difference panel for a run strip."""
    diff_dir = output_root / "assets" / "diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    path = diff_dir / f"input_difference_{run['objective']}_{run['case_slug']}.png"

    original_path = run["images"]["original"]
    perturbed_path = run["images"]["perturbed"]
    if not original_path.exists() or not perturbed_path.exists():
        placeholder = copy_or_placeholder(Path("__missing__"), size, "Input Difference")
        placeholder.save(path, optimize=True)
        return path

    original = Image.open(original_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    perturbed = Image.open(perturbed_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(perturbed, original)
    diff = ImageEnhance.Brightness(diff).enhance(8.0)
    diff = ImageOps.autocontrast(diff, cutoff=0.5)
    diff.save(path, optimize=True)
    return path


def make_strip(run: dict[str, Any], output_root: Path, quality: dict[str, Any]) -> str:
    strips_dir = output_root / "assets" / "strips"
    strips_dir.mkdir(parents=True, exist_ok=True)
    base_size = tuple(quality["image_size"])
    diff_path = make_input_difference_image(run, output_root, base_size)
    cells = [
        ("Original", run["images"]["original"]),
        ("Objective Ref", run["images"]["reference"]),
        ("Perturbed", run["images"]["perturbed"]),
        ("Input Difference x8", diff_path),
        ("Clean Edit", run["images"]["clean_edited"]),
        ("Perturbed Edit", run["images"]["perturbed_edited"]),
    ]
    loaded = []
    for label, path in cells:
        img = copy_or_placeholder(path, base_size, label).resize(base_size, Image.Resampling.LANCZOS)
        loaded.append((label, img))
    label_h = 34
    canvas = Image.new("RGB", (base_size[0] * len(loaded), base_size[1] + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(loaded):
        x = idx * base_size[0]
        canvas.paste(img, (x, 0))
        draw.text((x + 8, base_size[1] + 9), label, fill="black")
    strip_format = str(quality["strip_format"]).lower()
    name = f"wood_{run['objective']}_{run['case_slug']}.{strip_format}"
    path = strips_dir / name
    if strip_format in {"jpg", "jpeg"}:
        canvas.save(path, quality=int(quality["strip_quality"]), optimize=True)
    else:
        canvas.save(path, optimize=True)
    return path.relative_to(output_root).as_posix()


def plot_lines(path: Path, title: str, ylabel: str, runs: list[dict[str, Any]], key: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.0, 5.3), dpi=dpi)
    for run in runs:
        xs, ys = [], []
        for row in run["history_rows"]:
            x = to_float(row.get("iter"))
            y = to_float(row.get(key))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            label = f"{run['face_id']} / {run['prompt'].replace('add ', '')}"
            plt.plot(xs, ys, linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_psnr_ssim(path: Path, title: str, runs: list[dict[str, Any]], dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=dpi, sharex=True)
    for ax, key, label in [
        (axes[0], "ssim_to_original", "SSIM to original"),
        (axes[1], "psnr_to_original", "PSNR to original"),
    ]:
        for run in runs:
            xs, ys = [], []
            for row in run["history_rows"]:
                x = to_float(row.get("iter"))
                y = to_float(row.get(key))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if xs:
                ax.plot(xs, ys, linewidth=1.5, label=f"{run['face_id']} / {run['prompt'].replace('add ', '')}")
        ax.set_title(label)
        ax.set_xlabel("iteration")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_components(path: Path, title: str, runs: list[dict[str, Any]], normalized: bool = False, dpi: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("tps_mean_disp", "TPS"),
        ("delaunay_mean_disp", "Delaunay"),
        ("rolling_mean_disp", "Rolling"),
        ("dct_mean_disp", "DCT"),
        ("fft_phase_norm", "FFT phase"),
        ("fft_spatial_delta_mse", "FFT delta MSE"),
    ]
    max_iter = max((len(run["history_rows"]) for run in runs), default=0)
    plt.figure(figsize=(10.0, 5.3), dpi=dpi)
    for key, label in metrics:
        xs, ys = [], []
        for idx in range(max_iter):
            vals = []
            for run in runs:
                if idx < len(run["history_rows"]):
                    val = to_float(run["history_rows"][idx].get(key))
                    if val is not None:
                        vals.append(val)
            if vals:
                xs.append(idx + 1)
                ys.append(sum(vals) / len(vals))
        if normalized and ys:
            denom = max(abs(v) for v in ys)
            if denom > 0:
                ys = [v / denom for v in ys]
        if xs:
            plt.plot(xs, ys, linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("normalized value" if normalized else "raw diagnostic value")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def final_bar(path: Path, title: str, rows: list[dict[str, Any]], metrics: list[str], labels: list[str], dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xlabels = [f"{r['objective']}\n{r['face_id']}\n{r['prompt'].replace('add ', '')}" for r in rows]
    x = list(range(len(rows)))
    width = 0.8 / len(metrics)
    plt.figure(figsize=(max(11, len(rows) * 0.7), 5.5), dpi=dpi)
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        values = [to_float(r.get(metric)) or 0.0 for r in rows]
        offsets = [v + (idx - (len(metrics) - 1) / 2) * width for v in x]
        plt.bar(offsets, values, width=width, label=label)
    plt.title(title)
    plt.xticks(x, xlabels, rotation=45, ha="right", fontsize=7)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def scatter_z_ssim(path: Path, runs: list[dict[str, Any]], dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 5.3), dpi=dpi)
    for run in runs:
        x = to_float(run["final"].get("final_ssim_to_original"))
        y = to_float(run["final"].get("final_Z"))
        if x is None or y is None:
            continue
        plt.scatter([x], [y], s=65, label=f"{run['objective_name']} / {run['face_id']} / {run['prompt'].replace('add ', '')}")
    plt.title("Final Z vs SSIM to original")
    plt.xlabel("final SSIM to original")
    plt.ylabel("final Z")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_graphs(runs: list[dict[str, Any]], output_root: Path, quality: dict[str, Any]) -> dict[str, Any]:
    graphs_dir = output_root / "assets" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(quality["graph_dpi"])
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_obj[run["objective"]].append(run)
    sections = []
    for objective, group in sorted(by_obj.items(), key=lambda item: OBJECTIVE_ORDER.get(item[0], 99)):
        name = OBJECTIVE_NAMES.get(objective, objective)
        base = slug(name)
        section = {"title": name, "graphs": []}
        for suffix, title, ylabel, key in [
            ("Z", f"{name}: Z vs iteration", "Z", "Z"),
            ("loss", f"{name}: loss vs iteration", "loss", "loss"),
        ]:
            path = graphs_dir / f"{base}_{suffix}.png"
            plot_lines(path, title, ylabel, group, key, dpi=dpi)
            section["graphs"].append({"title": title, "path": path.relative_to(output_root).as_posix()})
        path = graphs_dir / f"{base}_ssim_psnr.png"
        plot_psnr_ssim(path, f"{name}: SSIM and PSNR to original", group, dpi=dpi)
        section["graphs"].append({"title": "SSIM and PSNR to original", "path": path.relative_to(output_root).as_posix(), "compact": True})
        path = graphs_dir / f"{base}_components_raw.png"
        plot_components(path, f"{name}: component diagnostics", group, normalized=False, dpi=dpi)
        section["graphs"].append({"title": "Geometry component contribution", "path": path.relative_to(output_root).as_posix()})
        path = graphs_dir / f"{base}_components_normalized.png"
        plot_components(path, f"{name}: normalized component diagnostics", group, normalized=True, dpi=dpi)
        section["graphs"].append({"title": "Geometry component contribution normalized", "path": path.relative_to(output_root).as_posix()})
        sections.append(section)

    return {"sections": sections}


def _safe(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def table_html(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='small'>No rows available.</p>"
    out = ["<div class='table-wrap'><table><thead><tr>"]
    for _, label in columns:
        out.append(f"<th>{_safe(label)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for key, _ in columns:
            value = row.get(key)
            out.append(f"<td>{fmt(value) if isinstance(value, (int, float)) else _safe(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows available._\n"
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, (int, float)):
                value = fmt(value)
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def graph_section_html(section: dict[str, Any]) -> str:
    out = [f"<section class='graph-section'><h4>{_safe(section['title'])}</h4><div class='graph-grid'>"]
    for graph in section["graphs"]:
        cls = "graph compact" if graph.get("compact") else "graph"
        out.append(
            f"<figure class='{cls}'><figcaption>{_safe(graph['title'])}</figcaption>"
            f"<a href='{_safe(graph['path'])}'><img src='{_safe(graph['path'])}' alt='{_safe(graph['title'])}'></a></figure>"
        )
    out.append("</div></section>")
    return "\n".join(out)


def build_html(data: dict[str, Any]) -> str:
    matrix_cols = [("model", "model"), ("code_objective", "code objective"), ("report_objective_name", "objective"), ("num_cases", "cases"), ("iterations", "iterations"), ("status", "status")]
    aggregate_cols = [("model", "model"), ("report_objective_name", "objective"), ("num_runs", "runs"), ("mean_final_Z", "mean final Z"), ("mean_final_ssim_to_original", "mean SSIM original"), ("mean_final_psnr_to_original", "mean PSNR original"), ("mean_final_output_ssim", "mean output SSIM"), ("mean_final_output_l2", "mean output L2"), ("mean_combined_max_disp_px", "mean max disp px")]
    per_cols = [("objective", "objective"), ("face_id", "face"), ("prompt", "prompt"), ("final_Z", "final Z"), ("final_loss", "final loss"), ("best_iter_by_Z", "best iter"), ("ssim_to_original", "SSIM original"), ("psnr_to_original", "PSNR original"), ("output_ssim", "output SSIM"), ("output_l2", "output L2"), ("max_disp_px", "max disp px")]
    css = """
    :root { --ink:#17202a; --muted:#5d6d7e; --line:#d7dde5; --soft:#f6f8fb; --blue:#1f5fbf; --panel:#fbfcfe; }
    body { margin:0; font-family: Inter, "Segoe UI", Arial, sans-serif; color:var(--ink); background:white; }
    main { max-width: 1180px; margin: 0 auto; padding: 34px 28px 70px; }
    h1 { font-size: 34px; margin: 0 0 6px; letter-spacing: -0.02em; }
    h2 { margin-top: 52px; padding-top: 18px; border-top: 2px solid var(--line); font-size: 26px; }
    h3 { margin-top: 34px; font-size: 21px; }
    h4 { margin: 24px 0 12px; font-size: 17px; }
    p, li { line-height: 1.55; }
    code, pre { background: var(--soft); border:1px solid var(--line); border-radius: 6px; }
    code { padding: 1px 5px; }
    pre { padding: 12px 14px; overflow:auto; }
    .subtitle { color:var(--muted); font-size: 17px; margin-bottom: 2px; }
    .author { color:var(--muted); margin-top: 0; }
    .card { border:1px solid var(--line); border-radius: 12px; padding: 18px; margin: 22px 0; background:white; box-shadow: 0 1px 2px rgba(0,0,0,.03); }
    .small { color:var(--muted); font-size: 13px; }
    .table-wrap { overflow-x:auto; max-width:100%; margin: 12px 0 22px; }
    table { border-collapse: collapse; width:100%; margin: 12px 0 22px; font-size: 13px; }
    th, td { border:1px solid var(--line); padding: 7px 9px; vertical-align: top; }
    th { background: var(--soft); text-align:left; font-weight: 650; }
    .strip { width:100%; border:1px solid var(--line); border-radius:10px; display:block; }
    .graph-section { border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:18px; margin:18px 0 28px; }
    .graph-section > h4 { margin-top:0; color:#111827; }
    .graph-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 18px; align-items:start; }
    .graph { border:1px solid var(--line); border-radius: 10px; padding: 12px; background:white; margin:0; }
    .graph figcaption { font-weight:650; margin:0 0 10px; color:#243447; }
    .graph img { width:100%; display:block; }
    .graph.compact img { max-height:260px; object-fit:contain; }
    .path { font-family: ui-monospace, Consolas, monospace; font-size: 12px; word-break: break-all; color:#334; }
    .toc a { color: var(--blue); text-decoration: none; }
    @media print { main { max-width: none; } .card { break-inside: avoid; } }
    """
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_safe(TITLE)}</title><style>{css}</style></head><body><main>",
        f"<h1>{_safe(TITLE)}</h1><p class='subtitle'>{_safe(SUBTITLE)}</p><p class='author'>{_safe(AUTHOR)}</p>",
        "<div class='card'><h3>Overview</h3><p>WOOD is an InstructPix2Pix-only white-box geometry optimization experiment. The scalar objective is <code>Z</code>, and the optimized loss is <code>loss = -Z</code>. Model weights are frozen; only differentiable perturbation parameters are optimized.</p></div>",
        "<h2>1. Method</h2>",
        "<h3>1.1 Combined perturbation module</h3><p>The perturbation module jointly optimizes five differentiable components. TPS, Delaunay/piecewise affine, rolling shutter, and DCT operate through spatial displacement fields. FFT phase is included as a differentiable frequency-domain perturbation after the spatial warp. All active components are optimized together within each run.</p>",
        "<h3>1.2 Loss function</h3><pre>loss = -Z</pre><p><code>Z</code> is the MSE distance between the selected InstructPix2Pix internal representation of the perturbed image and a fixed objective reference representation. No visual counter-loss is used.</p>",
        "<h3>1.3 Internal objectives</h3><ul><li><b>VAE conditioning latent</b>: code objective <code>vae_conditioning</code>.</li><li><b>UNet denoising prediction</b>: code objective <code>unet_prediction</code>.</li></ul>",
        "<h3>1.4 Run matrix</h3>",
        table_html(data["matrix_rows"], matrix_cols),
        "<h2>2. Results</h2>",
        "<h3>2.1 Aggregate summary</h3>",
        table_html(data["aggregate_rows"], aggregate_cols),
        "<h3>2.2 Per-run final values</h3>",
        table_html(data["per_run_rows"], per_cols),
    ]
    for objective in ["vae_conditioning", "unet_prediction"]:
        group = [run for run in data["runs"] if run["objective"] == objective]
        if not group:
            continue
        parts.append(f"<h3>{_safe(OBJECTIVE_NAMES[objective])}</h3>")
        for run in group:
            final = run["final"]
            parts.append(
                "<div class='card'>"
                f"<h4>{_safe(run['face_id'])} — {_safe(run['prompt'])}</h4>"
                f"<img class='strip' src='{_safe(run['strip_path'])}' alt='image strip'>"
                "<h4>Final values</h4>"
                + table_html(
                    [
                        {"metric": "final Z", "value": final.get("final_Z"), "metric2": "final loss", "value2": final.get("final_loss")},
                        {"metric": "SSIM to original", "value": final.get("final_ssim_to_original"), "metric2": "PSNR to original", "value2": final.get("final_psnr_to_original")},
                        {"metric": "output SSIM", "value": final.get("final_output_ssim"), "metric2": "output L2", "value2": final.get("final_output_l2")},
                        {"metric": "max displacement px", "value": final.get("combined_max_disp_px"), "metric2": "p95 displacement px", "value2": final.get("combined_p95_disp_px")},
                    ],
                    [("metric", "metric"), ("value", "value"), ("metric2", "metric"), ("value2", "value")],
                )
                + "</div>"
            )
    parts.append("<h2>3. Graphs</h2>")
    for section in data["graph_sections"]:
        parts.append(graph_section_html(section))
    notes = [
        f"Completed runs collected: {len(data['runs'])}.",
        f"Run id: {data['run_id']}.",
        "The report uses the latest completed WOOD run available when generated.",
    ]
    if data["missing"]:
        notes.append(f"Missing artifacts recorded: {len(data['missing'])}.")
    parts.append("<h2>4. Notes</h2><ul>" + "".join(f"<li>{_safe(note)}</li>" for note in notes) + "</ul>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def build_markdown(data: dict[str, Any]) -> str:
    matrix_cols = [("model", "model"), ("code_objective", "code objective"), ("report_objective_name", "objective"), ("num_cases", "cases"), ("iterations", "iterations"), ("status", "status")]
    aggregate_cols = [("model", "model"), ("report_objective_name", "objective"), ("num_runs", "runs"), ("mean_final_Z", "mean final Z"), ("mean_final_ssim_to_original", "mean SSIM original"), ("mean_final_output_ssim", "mean output SSIM"), ("mean_final_output_l2", "mean output L2")]
    per_cols = [("objective", "objective"), ("face_id", "face"), ("prompt", "prompt"), ("final_Z", "final Z"), ("final_loss", "final loss"), ("ssim_to_original", "SSIM original"), ("output_ssim", "output SSIM"), ("output_l2", "output L2"), ("max_disp_px", "max disp px")]
    lines = [
        f"# {TITLE}",
        "",
        SUBTITLE,
        "",
        f"Author: {AUTHOR}",
        "",
        "## Method",
        "",
        "WOOD optimizes `Z` with `loss = -Z`. Model weights are frozen; only differentiable perturbation parameters are optimized.",
        "",
        "## Run matrix",
        "",
        md_table(data["matrix_rows"], matrix_cols),
        "## Aggregate summary",
        "",
        md_table(data["aggregate_rows"], aggregate_cols),
        "## Per-run final values",
        "",
        md_table(data["per_run_rows"], per_cols),
        "## Image strips",
        "",
    ]
    for run in data["runs"]:
        lines.extend([f"### {run['objective_name']} / {run['face_id']} / {run['prompt']}", "", f"![strip]({run['strip_path']})", ""])
    lines.extend(["## Graphs", ""])
    for section in data["graph_sections"]:
        lines.extend([f"### {section['title']}", ""])
        for graph in section["graphs"]:
            lines.extend([f"#### {graph['title']}", "", f"![{graph['title']}]({graph['path']})", ""])
    return "\n".join(lines)


def make_pdf(data: dict[str, Any], output_root: Path, pdf_path: Path, quality: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.45 * inch, leftMargin=0.45 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch)
    story: list[Any] = []

    def p(text: str, style: str = "BodyText") -> None:
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 0.08 * inch))

    def add_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], font_size: int = 6) -> None:
        if not rows:
            p("No rows available.")
            return
        data = [[label for _, label in cols]]
        for row in rows:
            data.append([fmt(row.get(key)) if isinstance(row.get(key), (int, float)) else str(row.get(key, "")) for key, _ in cols])
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.12 * inch))

    def compressed_pdf_image(rel_path: str, max_px: tuple[int, int]) -> Path | None:
        path = output_root / rel_path
        if not path.exists():
            return None
        if not quality["compress_pdf_images"]:
            return path
        pdf_image_dir = output_root / "assets" / "pdf_images"
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        out = pdf_image_dir / f"{slug(rel_path)}.jpg"
        with Image.open(path) as img_raw:
            img = img_raw.convert("RGB")
            img.thumbnail(max_px, Image.Resampling.LANCZOS)
            img.save(out, quality=int(quality["pdf_image_quality"]), optimize=True)
        return out

    def add_image(rel_path: str, max_w: float = 7.2 * inch, max_h: float = 4.5 * inch) -> None:
        max_px = (int(max_w / inch * 140), int(max_h / inch * 140))
        path = compressed_pdf_image(rel_path, max_px)
        if path is None:
            return
        with Image.open(path) as img:
            w, h = img.size
        scale = min(max_w / w, max_h / h)
        story.append(RLImage(str(path), width=w * scale, height=h * scale))
        story.append(Spacer(1, 0.14 * inch))

    story.append(Paragraph(TITLE, styles["Title"]))
    p(SUBTITLE, "Heading2")
    p(f"Author: {AUTHOR}")
    p("WOOD optimizes Z with loss = -Z. Model weights are frozen; only differentiable perturbation parameters are optimized.")
    p("Run matrix", "Heading2")
    add_table(data["matrix_rows"], [("model", "model"), ("code_objective", "code objective"), ("report_objective_name", "objective"), ("num_cases", "cases"), ("iterations", "iterations"), ("status", "status")])
    p("Aggregate summary", "Heading2")
    add_table(data["aggregate_rows"], [("model", "model"), ("report_objective_name", "objective"), ("num_runs", "runs"), ("mean_final_Z", "mean final Z"), ("mean_final_ssim_to_original", "mean SSIM"), ("mean_final_output_ssim", "output SSIM"), ("mean_final_output_l2", "output L2")])
    p("Per-run final values", "Heading2")
    add_table(data["per_run_rows"], [("objective", "objective"), ("face_id", "face"), ("prompt", "prompt"), ("final_Z", "Z"), ("final_loss", "loss"), ("ssim_to_original", "SSIM"), ("output_ssim", "out SSIM"), ("output_l2", "out L2"), ("max_disp_px", "max disp")], font_size=5)
    story.append(PageBreak())
    p("Image strips", "Heading2")
    for run in data["runs"]:
        p(f"{run['objective_name']} / {run['face_id']} / {run['prompt']}", "Heading3")
        add_image(run["strip_path"], max_w=7.2 * inch, max_h=2.0 * inch)
    story.append(PageBreak())
    p("Graphs", "Heading2")
    for section in data["graph_sections"]:
        p(section["title"], "Heading3")
        for graph in section["graphs"]:
            p(graph["title"], "Heading4")
            add_image(graph["path"])
    doc.build(story)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    results_root = Path(args.results_root)
    quality = quality_settings(bool(args.compress_report))
    run_root = resolve_run_root(results_root, args.run_root, args.run_folder)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "assets" / "tables").mkdir(parents=True, exist_ok=True)
    runs, missing = collect_runs(run_root)
    for run in runs:
        run["strip_path"] = make_strip(run, output_root, quality)
    graphs = make_graphs(runs, output_root, quality)
    data = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "author": AUTHOR,
        "run_id": run_root.name.split("_")[0] + "_" + run_root.name.split("_")[1] if "_" in run_root.name else run_root.name,
        "run_root": str(run_root),
        "runs": runs,
        "missing": missing,
        "matrix_rows": run_matrix_rows(runs),
        "aggregate_rows": aggregate_rows(runs),
        "per_run_rows": per_run_rows(runs),
        "graph_sections": graphs["sections"],
        "compress_report": bool(args.compress_report),
        "quality": quality,
    }
    write_csv(output_root / "assets" / "tables" / "run_matrix_summary.csv", data["matrix_rows"])
    write_csv(output_root / "assets" / "tables" / "aggregate_summary_by_objective.csv", data["aggregate_rows"])
    write_csv(output_root / "assets" / "tables" / "per_run_final_values.csv", data["per_run_rows"])
    (output_root / "missing_artifacts.md").write_text(
        "# Missing artifacts\n\n" + ("\n".join(f"- {m['objective']} / {m['case']}: {m['artifact']} ({m['path']})" for m in missing) if missing else "None.\n"),
        encoding="utf-8",
    )
    summary = {
        "run_id": data["run_id"],
        "num_runs": len(runs),
        "num_missing": len(missing),
        "html": f"{OUTPUT_BASENAME}.html",
        "markdown": f"{OUTPUT_BASENAME}.md",
        "pdf": f"{OUTPUT_BASENAME}.pdf",
        "tables": ["run_matrix_summary.csv", "aggregate_summary_by_objective.csv", "per_run_final_values.csv"],
        "compress_report": bool(args.compress_report),
        "quality": quality,
    }
    (output_root / "report_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html_text = build_html(data)
    md_text = build_markdown(data)
    (output_root / f"{OUTPUT_BASENAME}.html").write_text(html_text, encoding="utf-8")
    (output_root / f"{OUTPUT_BASENAME}.md").write_text(md_text, encoding="utf-8")
    if not args.no_pdf:
        make_pdf(data, output_root, output_root / f"{OUTPUT_BASENAME}.pdf", quality)
    print(f"[wood-report] run root: {run_root}")
    print(f"[wood-report] compress report: {bool(args.compress_report)}")
    print(f"[wood-report] wrote: {output_root / f'{OUTPUT_BASENAME}.html'}")
    print(f"[wood-report] wrote: {output_root / f'{OUTPUT_BASENAME}.md'}")
    if not args.no_pdf:
        print(f"[wood-report] wrote: {output_root / f'{OUTPUT_BASENAME}.pdf'}")


if __name__ == "__main__":
    main()
