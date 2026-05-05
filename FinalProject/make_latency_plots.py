#!/usr/bin/env python3
"""
Generate report-oriented plots from analysis/long_latency.csv and analysis/summary_latency.csv.

This intentionally creates a small number of figures suitable for a 10-page report:
    1. Distribution plot for the main representative image.
    2. System-mode comparison for the large image.
    3. Individual optimization comparison under no MIG/MPS.
    4. Scaling plot versus image size.
    5. Jitter summary table as Markdown.

No seaborn is used; only pandas/numpy/matplotlib.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SYSTEM_ORDER = ["none", "mps", "mig", "mig_mps"]
LOAD_ORDER = ["single", "busy"]
VARIANT_ORDER = ["unoptimized", "fusion_only", "graphs_only", "green_only", "all_determinism"]
IMAGE_PRIORITY = ["small_house_rgb", "suzanne_rgb", "gradient_hd_rgb", "four_k_rgb", "eight_k_rgb", "sixteen_k_rgb", "16k_rgb"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Gaussian-blur latency plots.")
    p.add_argument("--analysis-dir", default="analysis", type=Path, help="Directory containing processed CSVs.")
    p.add_argument("--fig-dir", default="figures", type=Path, help="Directory for generated figures.")
    p.add_argument("--main-image", default=None, help="Representative image stem for main distribution/optimization plots.")
    p.add_argument("--large-image", default=None, help="Large image stem for system-mode comparison.")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Figure file format.")
    p.add_argument("--dpi", default=180, type=int, help="DPI for raster outputs.")
    p.add_argument("--max-boxplot-categories", default=18, type=int, help="Avoid unreadable boxplots if too many categories match.")
    return p.parse_args()


def ordered_unique(values: Iterable[str], preferred: list[str]) -> list[str]:
    seen = {str(v) for v in values if pd.notna(v)}
    out = [v for v in preferred if v in seen]
    out.extend(sorted(seen - set(out)))
    return out


def pick_image(available: Iterable[str], requested: str | None, preferred: list[str]) -> str:
    available_set = {str(v) for v in available if pd.notna(v)}
    if requested:
        if requested not in available_set:
            raise ValueError(f"Requested image '{requested}' not found. Available: {sorted(available_set)}")
        return requested
    for image in preferred:
        if image in available_set:
            return image
    return sorted(available_set)[0]


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def label_case(row: pd.Series) -> str:
    return f"{row.system_mode}\n{row.load_mode}\n{row.variant}"


def sort_case_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["system_order"] = out["system_mode"].map(lambda x: SYSTEM_ORDER.index(x) if x in SYSTEM_ORDER else 999)
    out["load_order"] = out["load_mode"].map(lambda x: LOAD_ORDER.index(x) if x in LOAD_ORDER else 999)
    out["variant_order"] = out["variant"].map(lambda x: VARIANT_ORDER.index(x) if x in VARIANT_ORDER else 999)
    return out.sort_values(["system_order", "load_order", "variant_order", "system_mode", "load_mode", "variant"])


def plot_main_distribution(long_df: pd.DataFrame, main_image: str, fig_dir: Path, fmt: str, dpi: int, max_categories: int) -> None:
    # Main report distribution: compare unoptimized vs all_determinism across system/load modes.
    keep = long_df[
        (long_df["image_stem"] == main_image)
        & (long_df["variant"].isin(["unoptimized", "all_determinism"]))
        & (long_df["system_mode"].isin(SYSTEM_ORDER))
        & (long_df["load_mode"].isin(LOAD_ORDER))
    ].copy()

    if keep.empty:
        print("Skipping main distribution plot: no matching rows.")
        return

    groups = []
    labels = []
    for _, case in sort_case_df(keep[["system_mode", "load_mode", "variant"]].drop_duplicates()).iterrows():
        subset = keep[
            (keep["system_mode"] == case.system_mode)
            & (keep["load_mode"] == case.load_mode)
            & (keep["variant"] == case.variant)
        ]
        if not subset.empty:
            groups.append(subset["latency_ms"].to_numpy(dtype=float))
            labels.append(label_case(case))

    if len(groups) > max_categories:
        print(f"Skipping main distribution plot: {len(groups)} categories > {max_categories}.")
        return

    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(groups)), 5.6))
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("E2E latency (ms)")
    ax.set_title(f"E2E latency distribution on {main_image}")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, fig_dir / f"fig1_distribution_{main_image}.{fmt}", dpi)


def plot_system_mode_summary(summary: pd.DataFrame, large_image: str, fig_dir: Path, fmt: str, dpi: int) -> None:
    # Compare isolation/configuration methods for all_determinism on the large image.
    keep = summary[
        (summary["image_stem"] == large_image)
        & (summary["variant"] == "all_determinism")
        & (summary["system_mode"].isin(SYSTEM_ORDER))
        & (summary["load_mode"].isin(LOAD_ORDER))
    ].copy()

    if keep.empty:
        print("Skipping system-mode plot: no matching rows.")
        return

    keep = sort_case_df(keep)
    x_labels = [f"{r.system_mode}\n{r.load_mode}" for _, r in keep.iterrows()]
    x = np.arange(len(keep))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(keep)), 5.2))
    ax.bar(x - width, keep["mean_ms"], width, label="mean")
    ax.bar(x, keep["p99_ms"], width, label="p99")
    ax.bar(x + width, keep["max_ms"], width, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("E2E latency (ms)")
    ax.set_title(f"System-mode comparison for all optimizations on {large_image}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, fig_dir / f"fig2_system_modes_{large_image}.{fmt}", dpi)


def plot_individual_optimizations(summary: pd.DataFrame, main_image: str, fig_dir: Path, fmt: str, dpi: int) -> None:
    # Shows whether each individual latency-determinism optimization helped.
    keep = summary[
        (summary["image_stem"] == main_image)
        & (summary["system_mode"] == "none")
        & (summary["load_mode"] == "single")
        & (summary["variant"].isin(VARIANT_ORDER))
    ].copy()

    if keep.empty:
        print("Skipping individual-optimization plot: no matching rows.")
        return

    keep["variant_order"] = keep["variant"].map(lambda x: VARIANT_ORDER.index(x) if x in VARIANT_ORDER else 999)
    keep = keep.sort_values("variant_order")
    labels = keep["variant"].tolist()
    x = np.arange(len(keep))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, keep["mean_ms"], width, label="mean")
    ax.bar(x + width / 2, keep["p99_ms"], width, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("E2E latency (ms)")
    ax.set_title(f"Individual optimization effects without MIG/MPS on {main_image}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, fig_dir / f"fig3_individual_optimizations_{main_image}.{fmt}", dpi)


def plot_scaling(summary: pd.DataFrame, fig_dir: Path, fmt: str, dpi: int) -> None:
    # Scaling story: how image size changes mean and p99 for selected cases.
    selected_cases = [
        ("none", "single", "unoptimized"),
        ("none", "single", "all_determinism"),
        ("none", "busy", "unoptimized"),
        ("mps", "busy", "all_determinism"),
        ("mig_mps", "busy", "all_determinism"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted = 0
    for system_mode, load_mode, variant in selected_cases:
        keep = summary[
            (summary["system_mode"] == system_mode)
            & (summary["load_mode"] == load_mode)
            & (summary["variant"] == variant)
            & summary["mpix"].notna()
        ].copy()
        if keep.empty:
            continue
        keep = keep.sort_values("mpix")
        ax.plot(keep["mpix"], keep["p99_ms"], marker="o", label=f"{system_mode}/{load_mode}/{variant}")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("Skipping scaling plot: no selected cases found.")
        return

    ax.set_xlabel("Image size (MPixels)")
    ax.set_ylabel("p99 E2E latency (ms)")
    ax.set_title("Scaling of tail latency with image size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    savefig(fig, fig_dir / f"fig4_scaling_p99.{fmt}", dpi)


def plot_jitter_reduction(summary: pd.DataFrame, large_image: str, fig_dir: Path, fmt: str, dpi: int) -> None:
    # A compact jitter plot using p99-minus-median as a tail-jitter proxy.
    keep = summary[
        (summary["image_stem"] == large_image)
        & (summary["variant"].isin(["unoptimized", "all_determinism"]))
        & (summary["system_mode"].isin(SYSTEM_ORDER))
        & (summary["load_mode"].isin(LOAD_ORDER))
    ].copy()
    if keep.empty:
        print("Skipping jitter plot: no matching rows.")
        return

    keep = sort_case_df(keep)
    labels = [f"{r.system_mode}\n{r.load_mode}\n{r.variant}" for _, r in keep.iterrows()]
    x = np.arange(len(keep))

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(keep)), 5))
    ax.bar(x, keep["p99_minus_median_us"] / 1_000.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("p99 - median latency (ms)")
    ax.set_title(f"Tail-jitter proxy on {large_image}")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, fig_dir / f"fig5_tail_jitter_{large_image}.{fmt}", dpi)


def make_markdown_tables(summary: pd.DataFrame, fig_dir: Path, main_image: str, large_image: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    md_path = fig_dir / "report_tables.md"

    def compact_table(image: str) -> str:
        keep = summary[
            (summary["image_stem"] == image)
            & (summary["variant"].isin(["unoptimized", "all_determinism", "fusion_only", "graphs_only", "green_only"]))
        ].copy()
        if keep.empty:
            return f"\n## {image}\n\nNo data.\n"
        keep = sort_case_df(keep)
        cols = ["system_mode", "load_mode", "variant", "n", "mean_ms", "p95_ms", "p99_ms", "max_ms", "std_over_mean_pct"]
        keep = keep[cols].copy()
        for col in ["mean_ms", "p95_ms", "p99_ms", "max_ms", "std_over_mean_pct"]:
            keep[col] = keep[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        return f"\n## {image}\n\n" + keep.to_markdown(index=False) + "\n"

    text = "# Report-ready latency tables\n"
    text += compact_table(main_image)
    if large_image != main_image:
        text += compact_table(large_image)

    md_path.write_text(text, encoding="utf-8")
    print(f"Wrote {md_path}")


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    fig_dir = args.fig_dir.resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    long_path = analysis_dir / "long_latency.csv"
    summary_path = analysis_dir / "summary_latency.csv"

    if not long_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {long_path} or {summary_path}. Run collect_latency_data.py first."
        )

    long_df = pd.read_csv(long_path)
    summary = pd.read_csv(summary_path)

    main_image = pick_image(summary["image_stem"].unique(), args.main_image, ["suzanne_rgb", "gradient_hd_rgb", "small_house_rgb", "four_k_rgb"])
    large_image = pick_image(summary["image_stem"].unique(), args.large_image, ["four_k_rgb", "eight_k_rgb", "suzanne_rgb", "gradient_hd_rgb"])

    print(f"Using main image:  {main_image}")
    print(f"Using large image: {large_image}")

    plot_main_distribution(long_df, main_image, fig_dir, args.format, args.dpi, args.max_boxplot_categories)
    plot_system_mode_summary(summary, large_image, fig_dir, args.format, args.dpi)
    plot_individual_optimizations(summary, main_image, fig_dir, args.format, args.dpi)
    plot_scaling(summary, fig_dir, args.format, args.dpi)
    plot_jitter_reduction(summary, large_image, fig_dir, args.format, args.dpi)
    make_markdown_tables(summary, fig_dir, main_image, large_image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
