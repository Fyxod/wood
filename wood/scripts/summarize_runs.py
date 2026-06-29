"""Summarize WOOD outputs and create lightweight report artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize WOOD run outputs.")
    parser.add_argument("--results-root", default="outputs/blank_objective_ref")
    parser.add_argument("--output-root", default="outputs/reports/blank_objective_ref")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _collect(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_root.glob("**/summary.json")):
        payload = _read_json(path)
        if "final_Z" not in payload:
            continue
        payload = dict(payload)
        payload["summary_path"] = str(path)
        payload["history_path"] = str(path.parent / "history.csv")
        payload["comparison_sheet"] = str(path.parent / "comparison_sheet.png")
        rows.append(payload)
    return rows


def _plot_curves(rows: list[dict[str, Any]], output_root: Path) -> list[str]:
    import pandas as pd
    import matplotlib.pyplot as plt

    graph_root = output_root / "metric_curves"
    graph_root.mkdir(parents=True, exist_ok=True)
    graph_paths: list[str] = []
    metrics = [
        ("Z", "Z vs iteration"),
        ("loss", "loss vs iteration"),
        ("psnr_to_original", "PSNR-to-original vs iteration"),
        ("ssim_to_original", "SSIM-to-original vs iteration"),
        ("fraction_clamped_total", "fraction clamped vs iteration"),
    ]
    for metric, title in metrics:
        plt.figure(figsize=(9, 5))
        any_rows = False
        for row in rows:
            history = Path(row["history_path"])
            if not history.exists():
                continue
            df = pd.read_csv(history)
            if metric not in df.columns:
                continue
            label = f"{row.get('objective_name')} | {row.get('face_id')} | {row.get('prompt')}"
            plt.plot(df["iter"], df[metric], label=label)
            any_rows = True
        if any_rows:
            plt.title(title)
            plt.xlabel("iteration")
            plt.ylabel(metric)
            plt.legend(fontsize=7)
            plt.tight_layout()
            path = graph_root / f"{metric}_curves.png"
            plt.savefig(path, dpi=140)
            graph_paths.append(str(path))
        plt.close()

    component_metrics = ["tps_max_disp", "delaunay_max_disp", "rolling_max_disp", "dct_max_disp"]
    plt.figure(figsize=(9, 5))
    any_rows = False
    for row in rows:
        history = Path(row["history_path"])
        if not history.exists():
            continue
        df = pd.read_csv(history)
        label_prefix = f"{row.get('objective_name')} | {row.get('face_id')} | {row.get('prompt')}"
        for metric in component_metrics:
            if metric in df.columns:
                plt.plot(df["iter"], df[metric], label=f"{label_prefix} | {metric}", alpha=0.7)
                any_rows = True
    if any_rows:
        plt.title("component max displacement vs iteration")
        plt.xlabel("iteration")
        plt.ylabel("max displacement / value")
        plt.legend(fontsize=6)
        plt.tight_layout()
        path = graph_root / "component_max_displacement_vs_iteration.png"
        plt.savefig(path, dpi=140)
        graph_paths.append(str(path))
    plt.close()

    if rows:
        plt.figure(figsize=(7, 5))
        for row in rows:
            plt.scatter(row.get("final_ssim_to_original", row.get("input_ssim", 0)), row.get("final_Z", 0), label=f"{row.get('objective_name')} {row.get('face_id')}")
        plt.title("final Z vs final SSIM-to-original")
        plt.xlabel("final SSIM to original")
        plt.ylabel("final Z")
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = graph_root / "final_Z_vs_final_SSIM_scatter.png"
        plt.savefig(path, dpi=140)
        graph_paths.append(str(path))
        plt.close()
    return graph_paths


def _write_report(rows: list[dict[str, Any]], graphs: list[str], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# WOOD blank-objective-reference report",
        "",
        "WOOD optimizes the scalar `Z` with `loss = -Z` for InstructPix2Pix only.",
        "The blank image is used as the objective-space reference, not as a visual counter-loss.",
        "",
        f"Completed runs found: {len(rows)}",
        "",
        "## Per-run image index",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row.get('objective_name')} / {row.get('face_id')} / {row.get('prompt')}",
                "",
                f"- final Z: {row.get('final_Z')}",
                f"- final loss: {row.get('final_loss')}",
                f"- final SSIM to original: {row.get('final_ssim_to_original')}",
                f"- final output SSIM: {row.get('final_output_ssim')}",
                f"- run dir: `{row.get('run_dir')}`",
                f"- comparison sheet: `{row.get('comparison_sheet')}`",
                "",
            ]
        )
    lines.extend(["## Graphs", ""])
    for graph in graphs:
        rel = Path(graph)
        lines.append(f"![{rel.stem}]({rel.as_posix()})")
        lines.append("")
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>WOOD report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} img{max-width:100%;border:1px solid #ddd;margin:8px 0}.card{border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}</style>",
        "</head><body>",
        "<h1>WOOD blank-objective-reference report</h1>",
        "<p>WOOD optimizes <code>Z</code> with <code>loss = -Z</code> for InstructPix2Pix only. The blank image is the objective-space reference.</p>",
        f"<p>Completed runs found: {len(rows)}</p>",
        "<h2>Runs</h2>",
    ]
    for row in rows:
        sheet = Path(row.get("comparison_sheet", ""))
        html.extend(
            [
                "<div class='card'>",
                f"<h3>{row.get('objective_name')} / {row.get('face_id')} / {row.get('prompt')}</h3>",
                f"<p>final Z: {row.get('final_Z')} | final loss: {row.get('final_loss')} | final SSIM to original: {row.get('final_ssim_to_original')}</p>",
                f"<p>run dir: <code>{row.get('run_dir')}</code></p>",
            ]
        )
        if sheet.exists():
            rel = sheet.resolve().relative_to(output_root.resolve()) if sheet.resolve().is_relative_to(output_root.resolve()) else sheet
            html.append(f"<img src='{rel.as_posix()}' alt='comparison sheet'>")
        html.append("</div>")
    html.append("<h2>Graphs</h2>")
    for graph in graphs:
        path = Path(graph)
        html.append(f"<h3>{path.stem}</h3><img src='{path.as_posix()}' alt='{path.stem}'>")
    html.append("</body></html>")
    (output_root / "report.html").write_text("\n".join(html), encoding="utf-8")

    image_lines = ["# WOOD image index", ""]
    for row in rows:
        image_lines.append(f"- {row.get('objective_name')} / {row.get('face_id')} / {row.get('prompt')}: `{row.get('comparison_sheet')}`")
    (output_root / "image_index.md").write_text("\n".join(image_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    rows = _collect(results_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "per_run_final_values.csv", rows)

    aggregate: list[dict[str, Any]] = []
    by_obj: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_obj.setdefault(str(row.get("objective_name")), []).append(row)
    for objective, vals in by_obj.items():
        aggregate.append(
            {
                "objective_name": objective,
                "num_runs": len(vals),
                "mean_final_Z": sum(float(v.get("final_Z", 0)) for v in vals) / max(len(vals), 1),
                "mean_final_ssim_to_original": sum(float(v.get("final_ssim_to_original", 0)) for v in vals) / max(len(vals), 1),
                "mean_final_output_l2": sum(float(v.get("final_output_l2", 0)) for v in vals) / max(len(vals), 1),
            }
        )
    _write_csv(output_root / "aggregate_summary.csv", aggregate)
    graphs = _plot_curves(rows, output_root)
    _write_report(rows, graphs, output_root)
    (output_root / "report_data_summary.json").write_text(
        json.dumps({"num_runs": len(rows), "graphs": graphs, "rows": rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"[wood] wrote report outputs to {output_root}")


if __name__ == "__main__":
    main()
