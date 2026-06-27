#!/usr/bin/env python3
"""Run learning-curve sweeps for pinn_fracenergy.py and summarize metrics.

This script launches repeated training runs with different NUM_SAMPLES and seeds,
parses the printed test metrics, and writes per-run + aggregated CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


METRIC_PATTERNS = {
    "mse_1d": re.compile(r"MSE \(1D\):\s*([0-9.eE+-]+)"),
    "mse_2d": re.compile(r"MSE \(2D\):\s*([0-9.eE+-]+)"),
    "mse_3d": re.compile(r"MSE \(3D\):\s*([0-9.eE+-]+)"),
    "nrmse_1d": re.compile(r"NRMSE \(1D\):\s*([0-9.eE+-]+)"),
    "nrmse_2d": re.compile(r"NRMSE \(2D\):\s*([0-9.eE+-]+)"),
    "nrmse_3d": re.compile(r"NRMSE \(3D\):\s*([0-9.eE+-]+)"),
    "r2_1d": re.compile(r"R2 \(1D\):\s*([0-9.eE+-]+)"),
    "r2_2d": re.compile(r"R2 \(2D\):\s*([0-9.eE+-]+)"),
    "r2_3d": re.compile(r"R2 \(3D\):\s*([0-9.eE+-]+)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning-curve sweep runner")
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 5000, 10000],
        help="Sample sizes to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Base seeds (used for DATA_SEED/MODEL_SEED/SPLIT_SEED)",
    )
    parser.add_argument("--epochs-1d", type=int, default=400)
    parser.add_argument("--epochs-2d", type=int, default=400)
    parser.add_argument("--epochs-3d", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-size-3d", type=int, default=64)
    parser.add_argument(
        "--output-prefix",
        default="learning_curve",
        help="Prefix for CSV and log outputs",
    )
    parser.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing pinn_fracenergy.py",
    )
    return parser.parse_args()


def extract_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    missing = []
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(output)
        if not match:
            missing.append(key)
            continue
        metrics[key] = float(match.group(1))
    if missing:
        raise ValueError(f"Could not parse metrics: {', '.join(missing)}")
    return metrics


def run_one(
    workdir: Path,
    num_samples: int,
    seed: int,
    epochs_1d: int,
    epochs_2d: int,
    epochs_3d: int,
    batch_size: int,
    batch_size_3d: int,
) -> Tuple[Dict[str, float], str]:
    env = os.environ.copy()
    env.update(
        {
            "NUM_SAMPLES": str(num_samples),
            "DATA_SEED": str(seed),
            "MODEL_SEED": str(seed),
            "SPLIT_SEED": str(seed),
            "RUN_PLOTS": "0",
            "RUN_SENSITIVITY": "0",
            "RUN_3D_SWEEP": "0",
            "USE_EARLY_STOP_1D": "1",
            "USE_EARLY_STOP_2D": "1",
            "USE_EARLY_STOP_3D": "1",
            "EPOCHS_1D": str(epochs_1d),
            "EPOCHS_2D": str(epochs_2d),
            "EPOCHS_3D": str(epochs_3d),
            "BATCH_SIZE": str(batch_size),
            "BATCH_SIZE_3D": str(batch_size_3d),
        }
    )

    proc = subprocess.run(
        [sys.executable, "pinn_fracenergy.py"],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"Run failed (samples={num_samples}, seed={seed}, rc={proc.returncode})\n{output}"
        )
    metrics = extract_metrics(output)
    return metrics, output


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_n: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_n.setdefault(int(row["num_samples"]), []).append(row)

    out: List[Dict[str, object]] = []
    metric_cols = [
        "mse_1d",
        "mse_2d",
        "mse_3d",
        "nrmse_1d",
        "nrmse_2d",
        "nrmse_3d",
        "r2_1d",
        "r2_2d",
        "r2_3d",
    ]
    for n in sorted(by_n):
        group = by_n[n]
        row: Dict[str, object] = {"num_samples": n, "runs": len(group)}
        for col in metric_cols:
            vals = [float(g[col]) for g in group]
            row[f"{col}_mean"] = statistics.mean(vals)
            row[f"{col}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rows: List[Dict[str, object]] = []

    logs_dir = workdir / f"{args.output_prefix}_logs_{ts}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.samples) * len(args.seeds)
    idx = 0
    for n in args.samples:
        for seed in args.seeds:
            idx += 1
            print(f"[{idx}/{total}] Running NUM_SAMPLES={n}, SEED={seed}", flush=True)
            metrics, output = run_one(
                workdir=workdir,
                num_samples=n,
                seed=seed,
                epochs_1d=args.epochs_1d,
                epochs_2d=args.epochs_2d,
                epochs_3d=args.epochs_3d,
                batch_size=args.batch_size,
                batch_size_3d=args.batch_size_3d,
            )

            log_path = logs_dir / f"run_n{n}_seed{seed}.log"
            log_path.write_text(output, encoding="utf-8")

            row: Dict[str, object] = {
                "num_samples": n,
                "seed": seed,
                **metrics,
                "log_file": str(log_path),
            }
            run_rows.append(row)
            print(
                f"  -> 3D: R2={metrics['r2_3d']:.4f}, NRMSE={metrics['nrmse_3d']:.4f}, MSE={metrics['mse_3d']:.4f}",
                flush=True,
            )

    run_csv = workdir / f"{args.output_prefix}_results_{ts}.csv"
    run_fields = [
        "num_samples",
        "seed",
        "mse_1d",
        "mse_2d",
        "mse_3d",
        "nrmse_1d",
        "nrmse_2d",
        "nrmse_3d",
        "r2_1d",
        "r2_2d",
        "r2_3d",
        "log_file",
    ]
    write_csv(run_csv, run_rows, run_fields)

    summary_rows = summarize(run_rows)
    summary_csv = workdir / f"{args.output_prefix}_summary_{ts}.csv"
    summary_fields = ["num_samples", "runs"] + [
        f"{m}_{sfx}"
        for m in [
            "mse_1d",
            "mse_2d",
            "mse_3d",
            "nrmse_1d",
            "nrmse_2d",
            "nrmse_3d",
            "r2_1d",
            "r2_2d",
            "r2_3d",
        ]
        for sfx in ("mean", "std")
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    print(f"\nPer-run results: {run_csv}")
    print(f"Summary results: {summary_csv}")
    print(f"Raw logs dir:    {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
