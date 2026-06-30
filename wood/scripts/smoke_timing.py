"""Smoke timing CLI for WOOD."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WOOD smoke timing.")
    parser.add_argument("--mat-root", required=True, help="Path to the MAT repository root.")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Run one representative case per objective.")
    parser.add_argument("--all-cases", action="store_true", help="Run all 8 WOOD combinations.")
    parser.add_argument("--blank-value", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--init", choices=["neutral", "small_random"], default=None, help="Override init in the geometry JSON.")
    parser.add_argument("--unet-backward-scale", type=float, default=8192.0)
    parser.add_argument("--geometry-config", default=None, help="JSON file controlling perturbation on/off toggles and hard ranges.")
    parser.add_argument("--skip-final-edits", action="store_true", help="Skip final Instruct edited outputs during debugging only.")
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.quick and not args.all_cases:
        raise SystemExit("Choose --quick or --all-cases for smoke timing.")
    from wood.core.runner import RunConfig, run_matrix

    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root) if args.output_root else repo_root / "outputs" / "smoke_timing"
    geometry_config = Path(args.geometry_config).resolve() if args.geometry_config else repo_root / "configs" / "geometry_default.json"
    cfg = RunConfig(
        mat_root=str(Path(args.mat_root).resolve()),
        output_root=str(output_root),
        iters=args.iters,
        blank_value=args.blank_value,
        lr=args.lr,
        seed=args.seed,
        init=args.init,
        unet_backward_scale=args.unet_backward_scale,
        geometry_config_path=str(geometry_config),
        quick=args.quick,
        all_cases=args.all_cases,
        mode="smoke_timing",
        skip_final_edits=args.skip_final_edits,
    )
    summary = run_matrix(cfg)
    print(f"[wood] smoke summary: {summary['output_root']}")
    print(f"[wood] estimated 150-iteration full matrix: {summary['time_estimates']['estimated_runtime_seconds_for_150_iterations'] / 60:.1f} min")


if __name__ == "__main__":
    main()
