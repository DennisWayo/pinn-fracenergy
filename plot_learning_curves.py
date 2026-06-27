#!/usr/bin/env python3
"""Generate per-dimension learning-curve figures from summary CSV output."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "codes" else SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning-curve metrics for 1D/2D/3D models."
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path to learning-curve summary CSV. If omitted, the newest generated summary is used.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where PNG files will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="learning_curve",
        help="Filename prefix for output figures.",
    )
    return parser.parse_args()


def read_summary(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if key == "runs":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda r: r["num_samples"])
    return rows


def plot_dimension(
    rows: List[Dict[str, float]], dim_key: str, out_path: str, runs: int
) -> None:
    dim_label = dim_key.upper()
    x = [int(r["num_samples"]) for r in rows]
    r2_mean = [r[f"r2_{dim_key}_mean"] for r in rows]
    r2_std = [r[f"r2_{dim_key}_std"] for r in rows]
    nrmse_mean = [r[f"nrmse_{dim_key}_mean"] for r in rows]
    nrmse_std = [r[f"nrmse_{dim_key}_std"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)

    axes[0].errorbar(
        x,
        r2_mean,
        yerr=r2_std,
        fmt="o-",
        capsize=4,
        linewidth=2,
        color="#0B3D91",
        label=r"$R^2$",
    )
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].errorbar(
        x,
        nrmse_mean,
        yerr=nrmse_std,
        fmt="s-",
        capsize=4,
        linewidth=2,
        color="#CC5500",
        label="NRMSE",
    )
    axes[1].set_ylabel("NRMSE")
    axes[1].set_xlabel("Number of samples (N)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].set_xscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(v) for v in x])

    fig.suptitle(f"{dim_label} Learning Curve (mean ± std over {runs} seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv
    if summary_csv is None:
        search_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = glob.glob(os.path.join(search_dir, "learning_curve*_summary_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "No learning-curve summary CSV found. Run learning_curve_study.py first "
                "or pass --summary-csv."
            )
        summary_csv = max(candidates, key=os.path.getmtime)

    rows = read_summary(summary_csv)
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {summary_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    runs = int(rows[0]["runs"])
    dims = ("1d", "2d", "3d")
    for dim in dims:
        out_path = os.path.join(args.out_dir, f"{args.prefix}_{dim.upper()}.png")
        plot_dimension(rows, dim, out_path, runs)
        print(out_path)


if __name__ == "__main__":
    main()
