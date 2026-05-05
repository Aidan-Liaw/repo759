#!/usr/bin/env python3
"""
Collect and summarize Gaussian-blur latency CSVs produced by the comprehensive run script.

Expected result tree, matching the current zsh script:
    results/<system_mode>/<load_mode>/<variant>/<image_stem>.csv

Outputs:
    analysis/long_latency.csv
    analysis/summary_latency.csv
    analysis/summary_with_baselines.csv
    analysis/detected_columns.csv

The script is deliberately tolerant of unknown CSV schemas. It tries to auto-detect the
best E2E latency column, but you should override it with --latency-column if the guessed
column is wrong.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Keep this map in one place so plots and summaries can attach dimensions even if the
# C++ CSV does not write width/height columns.
IMAGE_DIMENSIONS: dict[str, tuple[int, int]] = {
    "tiny_jelly_rgb": (256, 256),
    "small_house_rgb": (512, 512),
    "medium_shelf_rgb": (1280, 853),
    "hd_jaguar_rgb": (1920, 1080),
    "panorama_snow_rgb": (2047, 652),
    "wide_brick_rgb": (2048, 1062),
    "suzanne_rgb": (2048, 1152),
    "colonia_rgb": (2048, 1536),
    "menger_square_rgb": (2048, 2048),
    "gradient_hd_rgb": (2048, 1152),
    "line_art_rgb": (2048, 1152),
    "four_k_rgb": (3840, 2160),
    "eight_k_rgb": (7680, 4320),
    # If you generate one later, these are the usual 16:9 16K dimensions.
    "sixteen_k_rgb": (15360, 8640),
    "16k_rgb": (15360, 8640),
}

SYSTEM_ORDER = ["none", "mps", "mig", "mig_mps"]
LOAD_ORDER = ["single", "busy"]
VARIANT_ORDER = ["unoptimized", "fusion_only", "graphs_only", "green_only", "all_determinism"]

EXCLUDE_NUMERIC_NAME_RE = re.compile(
    r"(^|_)(iter|iteration|idx|index|sample|warmup|width|height|rows|cols|channels|"
    r"pixels|mpix|radius|r|sigma|device|gpu|block|threads?|sm|chunk|seed|success|error)(_|$)",
    re.IGNORECASE,
)

LATENCY_HINT_RE = re.compile(
    r"(e2e|end[_ -]?to[_ -]?end|latency|elapsed|duration|runtime|run[_ -]?time|total[_ -]?time|wall)",
    re.IGNORECASE,
)

UNIT_RE = re.compile(r"(^|[_\[(])(ns|nanosec|nanoseconds|us|µs|microsec|microseconds|ms|millisec|milliseconds|s|sec|seconds)([]_) ]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPath:
    system_mode: str
    load_mode: str
    variant: str
    image_stem: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect and summarize Gaussian-blur latency results.")
    p.add_argument("--results-root", default="results", type=Path, help="Root directory containing result CSVs.")
    p.add_argument("--out-dir", default="analysis", type=Path, help="Directory for processed CSV outputs.")
    p.add_argument("--latency-column", default=None, help="Exact latency column to use. Overrides auto-detection.")
    p.add_argument(
        "--input-unit",
        default="auto",
        choices=["auto", "ns", "us", "ms", "s"],
        help="Unit of the chosen latency column. Use auto to infer from column name.",
    )
    p.add_argument(
        "--allow-nonpositive",
        action="store_true",
        help="Keep zero/negative latency values instead of filtering them out.",
    )
    p.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail if any CSV path does not match results/system/load/variant/image.csv.",
    )
    return p.parse_args()


def parse_result_path(csv_path: Path, results_root: Path) -> ParsedPath | None:
    rel = csv_path.relative_to(results_root)
    parts = rel.parts
    if len(parts) < 4:
        return None
    return ParsedPath(system_mode=parts[-4], load_mode=parts[-3], variant=parts[-2], image_stem=csv_path.stem)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def score_latency_column(df: pd.DataFrame, col: str) -> int:
    name = col.strip().lower()
    score = 0

    if EXCLUDE_NUMERIC_NAME_RE.search(name):
        score -= 100
    if LATENCY_HINT_RE.search(name):
        score += 50
    if "e2e" in name or "end_to_end" in name or "end-to-end" in name:
        score += 50
    if "total" in name:
        score += 25
    if "kernel" in name:
        # Kernel-only timing is useful, but lower priority than E2E timing.
        score += 10
    if "latency" in name:
        score += 20
    if UNIT_RE.search(name):
        score += 15

    values = numeric_series(df, col)
    valid = values.dropna()
    if len(valid) == 0:
        return -10_000
    if len(valid) >= 10:
        score += 10
    if valid.nunique() <= 1:
        score -= 15
    if (valid > 0).mean() > 0.95:
        score += 5

    return score


def detect_latency_column(df: pd.DataFrame, override: str | None) -> str:
    if override is not None:
        if override not in df.columns:
            raise KeyError(f"Requested --latency-column '{override}' not found. Available columns: {list(df.columns)}")
        return override

    candidates = []
    for col in df.columns:
        values = numeric_series(df, col)
        if values.notna().sum() > 0:
            candidates.append((score_latency_column(df, col), col))

    if not candidates:
        raise ValueError(f"No numeric columns found. Columns: {list(df.columns)}")

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, best_col = candidates[0]

    if best_score < 0:
        raise ValueError(
            "Could not confidently detect a latency column. "
            f"Best candidate was '{best_col}' with score {best_score}. "
            f"Available columns: {list(df.columns)}. Use --latency-column."
        )

    return best_col


def infer_unit(column: str, requested_unit: str) -> str:
    if requested_unit != "auto":
        return requested_unit

    name = column.lower()
    # Order matters: check ns/us/ms before plain "s".
    if re.search(r"(^|[_\[(])(ns|nanosec|nanoseconds)([]_) ]|$)", name):
        return "ns"
    if re.search(r"(^|[_\[(])(us|µs|microsec|microseconds)([]_) ]|$)", name):
        return "us"
    if re.search(r"(^|[_\[(])(ms|millisec|milliseconds)([]_) ]|$)", name):
        return "ms"
    if re.search(r"(^|[_\[(])(s|sec|seconds)([]_) ]|$)", name):
        return "s"

    # CUDA event timings are commonly reported in milliseconds. If your code writes
    # bare numbers in microseconds instead, rerun with --input-unit us.
    return "ms"


def to_microseconds(values: pd.Series, unit: str) -> pd.Series:
    if unit == "ns":
        return values / 1_000.0
    if unit == "us":
        return values
    if unit == "ms":
        return values * 1_000.0
    if unit == "s":
        return values * 1_000_000.0
    raise ValueError(f"Unsupported unit: {unit}")


def width_height_for(image_stem: str, df: pd.DataFrame) -> tuple[float, float]:
    lower_cols = {c.lower(): c for c in df.columns}
    width_col = next((lower_cols[x] for x in ["width", "image_width", "w"] if x in lower_cols), None)
    height_col = next((lower_cols[x] for x in ["height", "image_height", "h"] if x in lower_cols), None)

    if width_col is not None and height_col is not None:
        w = pd.to_numeric(df[width_col], errors="coerce").dropna()
        h = pd.to_numeric(df[height_col], errors="coerce").dropna()
        if not w.empty and not h.empty:
            return float(w.iloc[0]), float(h.iloc[0])

    if image_stem in IMAGE_DIMENSIONS:
        w, h = IMAGE_DIMENSIONS[image_stem]
        return float(w), float(h)

    return math.nan, math.nan


def read_one_csv(csv_path: Path, results_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame | None, dict]:
    parsed = parse_result_path(csv_path, results_root)
    if parsed is None:
        if args.strict_paths:
            raise ValueError(f"CSV path does not match expected layout: {csv_path}")
        return None, {"source_csv": str(csv_path), "status": "skipped_bad_path"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return None, {"source_csv": str(csv_path), "status": f"read_error: {exc}"}

    df = clean_columns(df)
    try:
        latency_col = detect_latency_column(df, args.latency_column)
        unit = infer_unit(latency_col, args.input_unit)
        raw = numeric_series(df, latency_col)
        latency_us = to_microseconds(raw, unit)
    except Exception as exc:
        return None, {
            "source_csv": str(csv_path),
            "status": f"latency_detection_error: {exc}",
            "columns": ";".join(df.columns),
        }

    valid_mask = latency_us.notna()
    if not args.allow_nonpositive:
        valid_mask &= latency_us > 0

    if not valid_mask.any():
        return None, {
            "source_csv": str(csv_path),
            "status": "no_valid_latency_values",
            "detected_latency_column": latency_col,
            "detected_unit": unit,
            "columns": ";".join(df.columns),
        }

    selected = df.loc[valid_mask].reset_index(drop=True)
    latency_us = latency_us.loc[valid_mask].reset_index(drop=True)

    width, height = width_height_for(parsed.image_stem, selected)
    pixels = width * height if not (math.isnan(width) or math.isnan(height)) else math.nan

    long_df = pd.DataFrame(
        {
            "system_mode": parsed.system_mode,
            "load_mode": parsed.load_mode,
            "variant": parsed.variant,
            "image_stem": parsed.image_stem,
            "width": width,
            "height": height,
            "pixels": pixels,
            "mpix": pixels / 1_000_000.0 if not math.isnan(pixels) else math.nan,
            "sample_idx": np.arange(len(latency_us), dtype=int),
            "latency_us": latency_us.astype(float),
            "latency_ms": latency_us.astype(float) / 1_000.0,
            "source_csv": str(csv_path),
            "detected_latency_column": latency_col,
            "detected_unit": unit,
        }
    )

    return long_df, {
        "source_csv": str(csv_path),
        "status": "ok",
        "rows_in_csv": len(df),
        "valid_latency_rows": len(long_df),
        "detected_latency_column": latency_col,
        "detected_unit": unit,
        "columns": ";".join(df.columns),
    }


def percentile(q: float):
    def f(values: pd.Series) -> float:
        return float(np.percentile(values.dropna().to_numpy(dtype=float), q))

    f.__name__ = f"p{int(q)}_us"
    return f


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["system_mode", "load_mode", "variant", "image_stem", "width", "height", "pixels", "mpix"]

    summary = (
        long_df.groupby(group_cols, dropna=False)["latency_us"]
        .agg(
            n="count",
            mean_us="mean",
            median_us="median",
            std_us="std",
            min_us="min",
            p90_us=percentile(90),
            p95_us=percentile(95),
            p99_us=percentile(99),
            max_us="max",
        )
        .reset_index()
    )

    summary["range_us"] = summary["max_us"] - summary["min_us"]
    summary["p99_minus_median_us"] = summary["p99_us"] - summary["median_us"]
    summary["max_minus_median_us"] = summary["max_us"] - summary["median_us"]
    summary["std_over_mean_pct"] = 100.0 * summary["std_us"] / summary["mean_us"]
    summary["mean_ms"] = summary["mean_us"] / 1_000.0
    summary["median_ms"] = summary["median_us"] / 1_000.0
    summary["p95_ms"] = summary["p95_us"] / 1_000.0
    summary["p99_ms"] = summary["p99_us"] / 1_000.0
    summary["max_ms"] = summary["max_us"] / 1_000.0
    summary["range_ms"] = summary["range_us"] / 1_000.0
    summary["throughput_mpix_s"] = summary["mpix"] / (summary["mean_us"] / 1_000_000.0)

    return summary.sort_values(
        by=["image_stem", "load_mode", "system_mode", "variant"],
        key=lambda s: s.map(order_key) if s.name in {"system_mode", "load_mode", "variant"} else s,
    )


def order_key(value: object) -> int:
    text = str(value)
    if text in SYSTEM_ORDER:
        return SYSTEM_ORDER.index(text)
    if text in LOAD_ORDER:
        return LOAD_ORDER.index(text)
    if text in VARIANT_ORDER:
        return VARIANT_ORDER.index(text)
    return 10_000


def add_baseline_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    # Baseline 1: global baseline for each image: none/single/unoptimized.
    global_base = out[
        (out["system_mode"] == "none") & (out["load_mode"] == "single") & (out["variant"] == "unoptimized")
    ][["image_stem", "mean_us", "p99_us", "max_us", "std_us"]].rename(
        columns={
            "mean_us": "global_base_mean_us",
            "p99_us": "global_base_p99_us",
            "max_us": "global_base_max_us",
            "std_us": "global_base_std_us",
        }
    )
    out = out.merge(global_base, on="image_stem", how="left")

    for metric in ["mean", "p99", "max", "std"]:
        out[f"{metric}_ratio_vs_none_single_unoptimized"] = out[f"{metric}_us"] / out[f"global_base_{metric}_us"]
        out[f"{metric}_reduction_pct_vs_none_single_unoptimized"] = 100.0 * (
            1.0 - out[f"{metric}_ratio_vs_none_single_unoptimized"]
        )

    # Baseline 2: for each image/system/load, compare variants against unoptimized.
    local_base = out[out["variant"] == "unoptimized"][
        ["image_stem", "system_mode", "load_mode", "mean_us", "p99_us", "max_us", "std_us"]
    ].rename(
        columns={
            "mean_us": "local_base_mean_us",
            "p99_us": "local_base_p99_us",
            "max_us": "local_base_max_us",
            "std_us": "local_base_std_us",
        }
    )
    out = out.merge(local_base, on=["image_stem", "system_mode", "load_mode"], how="left")
    for metric in ["mean", "p99", "max", "std"]:
        out[f"{metric}_ratio_vs_same_mode_unoptimized"] = out[f"{metric}_us"] / out[f"local_base_{metric}_us"]
        out[f"{metric}_reduction_pct_vs_same_mode_unoptimized"] = 100.0 * (
            1.0 - out[f"{metric}_ratio_vs_same_mode_unoptimized"]
        )

    return out


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    csv_paths = sorted(results_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {results_root}")

    frames: list[pd.DataFrame] = []
    detections: list[dict] = []

    for csv_path in csv_paths:
        frame, detection = read_one_csv(csv_path, results_root, args)
        detections.append(detection)
        if frame is not None and not frame.empty:
            frames.append(frame)

    detection_df = pd.DataFrame(detections)
    detection_df.to_csv(out_dir / "detected_columns.csv", index=False)

    if not frames:
        raise RuntimeError(
            "No usable latency data found. Inspect analysis/detected_columns.csv and rerun with --latency-column/--input-unit."
        )

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(out_dir / "long_latency.csv", index=False)

    summary = summarize(long_df)
    summary.to_csv(out_dir / "summary_latency.csv", index=False)

    summary_with_baselines = add_baseline_ratios(summary)
    summary_with_baselines.to_csv(out_dir / "summary_with_baselines.csv", index=False)

    ok = int((detection_df["status"] == "ok").sum()) if "status" in detection_df.columns else 0
    print(f"Read {len(csv_paths)} CSV files; {ok} produced usable latency rows.")
    print(f"Long data: {out_dir / 'long_latency.csv'}")
    print(f"Summary:   {out_dir / 'summary_latency.csv'}")
    print(f"Baselines: {out_dir / 'summary_with_baselines.csv'}")
    print(f"Detection: {out_dir / 'detected_columns.csv'}")

    if ok < len(csv_paths):
        print("Some CSVs were skipped or had detection issues; inspect detected_columns.csv.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
