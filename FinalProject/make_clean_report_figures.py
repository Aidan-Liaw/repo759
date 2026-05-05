#!/usr/bin/env python3
"""
Clean report-oriented plots for Gaussian blur latency data.

Input expected by default:
  analysis/long_latency.csv

Outputs:
  analysis/report_clean/*.csv, *.md
  figures/report_clean/*.png

This script is intentionally conservative:
  - keeps 8000 ms samples in a separate timeout/failure analysis;
  - excludes timeout-like samples from steady-state plots;
  - avoids plotting unknown/failed 8K stress cases together with normal latency cases;
  - emphasizes p99 inflation under contention and p99-median tail jitter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SYSTEM_ORDER = ["none", "mig", "mps", "mig_mps"]
LOAD_ORDER = ["single", "busy"]
VARIANT_ORDER = [
    "unoptimized",
    "fusion_only",
    "graphs_only",
    "green_only",
    "all_determinism",
]
CORE_VARIANTS = ["unoptimized", "all_determinism"]


def image_short(name: str) -> str:
    s = str(name)
    low = s.lower()
    if "mengerschwamm" in low:
        return "8K stress"
    if "suzanne" in low:
        return "Suzanne"
    if "scan" in low or "belgium" in low:
        return "Scan"
    if "house" in low:
        return "House"
    return s[:28]


def pretty_system(s: str) -> str:
    return {
        "none": "No isolation",
        "mig": "MIG",
        "mps": "MPS",
        "mig_mps": "MIG+MPS",
    }.get(str(s), str(s))


def pretty_variant(v: str) -> str:
    return {
        "unoptimized": "Unoptimized",
        "fusion_only": "Fusion only",
        "graphs_only": "Graphs only",
        "green_only": "Green only",
        "all_determinism": "All determinism",
    }.get(str(v), str(v))


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def sort_key(row: pd.Series) -> tuple:
    return (
        SYSTEM_ORDER.index(row["system_mode"]) if row["system_mode"] in SYSTEM_ORDER else 99,
        LOAD_ORDER.index(row["load_mode"]) if row["load_mode"] in LOAD_ORDER else 99,
        VARIANT_ORDER.index(row["variant"]) if row["variant"] in VARIANT_ORDER else 99,
        str(row.get("image_short", row.get("image", ""))),
    )


def summarise_group(g: pd.DataFrame) -> pd.Series:
    x = g["latency_ms"].to_numpy(dtype=float)
    return pd.Series(
        {
            "n": int(len(x)),
            "mean_ms": float(np.mean(x)) if len(x) else np.nan,
            "median_ms": float(np.median(x)) if len(x) else np.nan,
            "std_ms": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "p95_ms": float(np.percentile(x, 95)) if len(x) else np.nan,
            "p99_ms": float(np.percentile(x, 99)) if len(x) else np.nan,
            "max_ms": float(np.max(x)) if len(x) else np.nan,
            "p99_minus_median_ms": float(np.percentile(x, 99) - np.median(x)) if len(x) else np.nan,
            "max_minus_median_ms": float(np.max(x) - np.median(x)) if len(x) else np.nan,
        }
    )


def add_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["system_rank"] = out["system_mode"].map({v: i for i, v in enumerate(SYSTEM_ORDER)}).fillna(99)
    out["load_rank"] = out["load_mode"].map({v: i for i, v in enumerate(LOAD_ORDER)}).fillna(99)
    out["variant_rank"] = out["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)}).fillna(99)
    out["image_rank"] = out["image_short"].map({"House": 0, "Scan": 1, "Suzanne": 2, "8K stress": 3}).fillna(50)
    return out


def hbar(df: pd.DataFrame, x: str, label: str, title: str, xlabel: str, outfile: Path, *, vline: float | None = None, max_items: int | None = None) -> None:
    plot = df.copy()
    if max_items is not None:
        plot = plot.head(max_items)
    if plot.empty:
        return

    labels = plot[label].astype(str).to_list()
    vals = plot[x].astype(float).to_numpy()
    y = np.arange(len(plot))

    height = max(4.5, min(12.0, 0.34 * len(plot) + 1.8))
    plt.figure(figsize=(11.5, height))
    plt.barh(y, vals)
    if vline is not None:
        plt.axvline(vline, linestyle="--", linewidth=1)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="analysis/long_latency.csv", help="Input long latency CSV")
    ap.add_argument("--outdir", default="figures/report_clean", help="Output figure directory")
    ap.add_argument("--table-dir", default="analysis/report_clean", help="Output table directory")
    ap.add_argument("--timeout-ms", type=float, default=7900.0, help="Treat latencies >= this as timeout-like")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum non-timeout samples required for steady-state summaries")
    args = ap.parse_args()

    in_path = Path(args.input)
    fig_dir = Path(args.outdir)
    tab_dir = Path(args.table_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if "image" not in df.columns and "image_stem" in df.columns:
        df["image"] = df["image_stem"]
    required = {"system_mode", "load_mode", "variant", "image", "latency_ms"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {in_path}: {sorted(missing)}")

    df = df.copy()
    df["image_short"] = df["image"].map(image_short)
    df["is_timeout_like"] = df["latency_ms"] >= args.timeout_ms

    keys = ["system_mode", "load_mode", "variant", "image", "image_short"]

    raw = df.groupby(keys, dropna=False).apply(summarise_group).reset_index()
    timeout = df.groupby(keys, dropna=False).agg(
        n_raw=("latency_ms", "size"),
        timeout_like_count=("is_timeout_like", "sum"),
        timeout_like_frac=("is_timeout_like", "mean"),
    ).reset_index()
    raw = raw.merge(timeout, on=keys, how="left")
    raw.to_csv(tab_dir / "summary_raw_with_timeout_flags.csv", index=False)

    steady_rows = df[~df["is_timeout_like"]].copy()
    steady = steady_rows.groupby(keys, dropna=False).apply(summarise_group).reset_index()
    steady = steady.merge(timeout, on=keys, how="left")
    steady = steady[steady["n"] >= args.min_n].copy()
    steady = add_order_columns(steady)
    steady.to_csv(tab_dir / "summary_steady_state_excluding_timeouts.csv", index=False)

    # Timeout/failure figure: separate from latency plots.
    failures = raw[raw["timeout_like_frac"] > 0].copy()
    if not failures.empty:
        failures["label"] = failures.apply(
            lambda r: f"{pretty_system(r.system_mode)} / {r.load_mode} / {pretty_variant(r.variant)} / {r.image_short}",
            axis=1,
        )
        failures = failures.sort_values(["timeout_like_frac", "timeout_like_count"], ascending=[False, False])
        failures.to_csv(tab_dir / "timeout_like_failures.csv", index=False)
        hbar(
            failures,
            x="timeout_like_frac",
            label="label",
            title="Timeout-like samples (kept separate from steady-state latency plots)",
            xlabel=f"Fraction of samples >= {args.timeout_ms:g} ms",
            outfile=fig_dir / "01_timeout_like_failures.png",
            vline=None,
            max_items=25,
        )

    # Contention penalty: busy p99 / single p99, steady-state only.
    single = steady[steady["load_mode"] == "single"].copy()
    busy = steady[steady["load_mode"] == "busy"].copy()
    merge_keys = ["system_mode", "variant", "image", "image_short"]
    penalty = busy.merge(single, on=merge_keys, suffixes=("_busy", "_single"), how="inner")
    if not penalty.empty:
        penalty["p99_busy_over_single"] = safe_ratio(penalty["p99_ms_busy"], penalty["p99_ms_single"])
        penalty["median_busy_over_single"] = safe_ratio(penalty["median_ms_busy"], penalty["median_ms_single"])
        penalty["jitter_busy_over_single"] = safe_ratio(
            penalty["p99_minus_median_ms_busy"], penalty["p99_minus_median_ms_single"]
        )
        penalty["label"] = penalty.apply(
            lambda r: f"{pretty_system(r.system_mode)} / {pretty_variant(r.variant)} / {r.image_short}",
            axis=1,
        )
        penalty = penalty.replace([np.inf, -np.inf], np.nan)
        penalty.to_csv(tab_dir / "contention_penalty_steady_state.csv", index=False)

        plot = penalty.dropna(subset=["p99_busy_over_single"]).sort_values("p99_busy_over_single", ascending=False)
        hbar(
            plot,
            x="p99_busy_over_single",
            label="label",
            title="Contention sensitivity: p99 latency with busy-wait competitor vs alone",
            xlabel="busy p99 / single p99 (lower is better; 1 = no penalty)",
            outfile=fig_dir / "02_contention_penalty_p99_ratio.png",
            vline=1.0,
            max_items=30,
        )

    # Core p99 busy vs single: unoptimized vs all_determinism only.
    core = penalty[penalty["variant"].isin(CORE_VARIANTS)].copy() if not penalty.empty else pd.DataFrame()
    if not core.empty:
        core["sort"] = core.apply(
            lambda r: (
                SYSTEM_ORDER.index(r["system_mode"]) if r["system_mode"] in SYSTEM_ORDER else 99,
                VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99,
                {"House": 0, "Scan": 1, "Suzanne": 2}.get(r["image_short"], 99),
            ),
            axis=1,
        )
        core = core.sort_values("sort")
        labels = core.apply(lambda r: f"{pretty_system(r.system_mode)} / {pretty_variant(r.variant)} / {r.image_short}", axis=1).to_list()
        y = np.arange(len(core))
        h = 0.38
        plt.figure(figsize=(11.5, max(5, 0.45 * len(core) + 1.8)))
        plt.barh(y - h / 2, core["p99_ms_single"], height=h, label="single")
        plt.barh(y + h / 2, core["p99_ms_busy"], height=h, label="busy")
        plt.yticks(y, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("p99 E2E latency (ms)")
        plt.title("Busy-wait effect on p99 latency for core configurations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "03_core_p99_single_vs_busy.png", dpi=220)
        plt.close()

    # Tail jitter in busy mode, steady-state only.
    busy_jitter = steady[steady["load_mode"] == "busy"].copy()
    if not busy_jitter.empty:
        busy_jitter["label"] = busy_jitter.apply(
            lambda r: f"{pretty_system(r.system_mode)} / {pretty_variant(r.variant)} / {r.image_short}",
            axis=1,
        )
        plot = busy_jitter.sort_values("p99_minus_median_ms", ascending=False)
        hbar(
            plot,
            x="p99_minus_median_ms",
            label="label",
            title="Tail-jitter proxy under busy-wait contention",
            xlabel="p99 - median latency (ms), lower is better",
            outfile=fig_dir / "04_busy_tail_jitter_p99_minus_median.png",
            vline=None,
            max_items=30,
        )

    # all_determinism vs unoptimized table and plot focused on busy mode.
    base = steady[steady["variant"] == "unoptimized"].copy()
    det = steady[steady["variant"] == "all_determinism"].copy()
    compare = det.merge(
        base,
        on=["system_mode", "load_mode", "image", "image_short"],
        suffixes=("_all_det", "_unopt"),
        how="inner",
    )
    if not compare.empty:
        compare["p99_ratio_all_det_over_unopt"] = safe_ratio(compare["p99_ms_all_det"], compare["p99_ms_unopt"])
        compare["jitter_ratio_all_det_over_unopt"] = safe_ratio(
            compare["p99_minus_median_ms_all_det"], compare["p99_minus_median_ms_unopt"]
        )
        compare["p99_pct_change_all_det_vs_unopt"] = 100.0 * (compare["p99_ratio_all_det_over_unopt"] - 1.0)
        compare["jitter_pct_change_all_det_vs_unopt"] = 100.0 * (compare["jitter_ratio_all_det_over_unopt"] - 1.0)
        compare["label"] = compare.apply(
            lambda r: f"{pretty_system(r.system_mode)} / {r.load_mode} / {r.image_short}",
            axis=1,
        )
        compare = compare.replace([np.inf, -np.inf], np.nan)
        compare.to_csv(tab_dir / "all_determinism_vs_unoptimized_steady_state.csv", index=False)

        # Only busy mode is the most defensible comparison for determinism under contention.
        plot = compare[compare["load_mode"] == "busy"].dropna(subset=["jitter_ratio_all_det_over_unopt"]).copy()
        plot = plot.sort_values("jitter_ratio_all_det_over_unopt")
        hbar(
            plot,
            x="jitter_ratio_all_det_over_unopt",
            label="label",
            title="Effect of all-determinism on tail jitter under busy-wait contention",
            xlabel="(all_determinism p99-median) / (unoptimized p99-median), lower is better",
            outfile=fig_dir / "05_all_det_vs_unopt_busy_jitter_ratio.png",
            vline=1.0,
            max_items=25,
        )

        plot = compare[compare["load_mode"] == "busy"].dropna(subset=["p99_ratio_all_det_over_unopt"]).copy()
        plot = plot.sort_values("p99_ratio_all_det_over_unopt")
        hbar(
            plot,
            x="p99_ratio_all_det_over_unopt",
            label="label",
            title="Effect of all-determinism on p99 latency under busy-wait contention",
            xlabel="all_determinism p99 / unoptimized p99, lower is better",
            outfile=fig_dir / "06_all_det_vs_unopt_busy_p99_ratio.png",
            vline=1.0,
            max_items=25,
        )

    # Individual optimizations: no isolation, single vs busy, p99 ratio. This is more compact than raw bars.
    indiv = penalty[(penalty["system_mode"] == "none") & (penalty["variant"].isin(VARIANT_ORDER))].copy() if not penalty.empty else pd.DataFrame()
    if not indiv.empty:
        indiv["label"] = indiv.apply(lambda r: f"{pretty_variant(r.variant)} / {r.image_short}", axis=1)
        indiv = indiv.sort_values("p99_busy_over_single", ascending=False)
        hbar(
            indiv,
            x="p99_busy_over_single",
            label="label",
            title="Which software variants are sensitive to contention? No MIG/MPS case",
            xlabel="busy p99 / single p99 (lower is better; 1 = no penalty)",
            outfile=fig_dir / "07_no_isolation_variant_contention_sensitivity.png",
            vline=1.0,
            max_items=25,
        )

    # Mean vs jitter trade-off, annotated for core configurations and high-jitter points.
    plot = steady.copy()
    if not plot.empty:
        plt.figure(figsize=(10.5, 7.2))
        for (sys_mode, load_mode), g in plot.groupby(["system_mode", "load_mode"], sort=False):
            plt.scatter(g["mean_ms"], g["p99_minus_median_ms"], label=f"{pretty_system(sys_mode)} / {load_mode}", s=45)
        # annotate the most important points only
        annotate = plot[
            (plot["variant"].isin(CORE_VARIANTS))
            | (plot["p99_minus_median_ms"] >= plot["p99_minus_median_ms"].quantile(0.85))
        ].copy()
        # limit annotations to avoid clutter
        annotate = annotate.sort_values("p99_minus_median_ms", ascending=False).head(18)
        for _, r in annotate.iterrows():
            label = f"{pretty_system(r.system_mode)}/{r.load_mode}/{pretty_variant(r.variant)}/{r.image_short}"
            plt.annotate(label, (r["mean_ms"], r["p99_minus_median_ms"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        plt.xlabel("Mean latency (ms)")
        plt.ylabel("p99 - median latency (ms)")
        plt.title("Throughput-vs-determinism trade-off, excluding timeout-like samples")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "08_mean_vs_tail_jitter_annotated.png", dpi=220)
        plt.close()

    # Markdown report snippets.
    with open(tab_dir / "report_snippets.md", "w", encoding="utf-8") as f:
        f.write("# Clean report snippets\n\n")
        f.write("## Recommended interpretation\n\n")
        f.write(
            "The normal-latency analysis excludes timeout-like samples (latency >= "
            f"{args.timeout_ms:g} ms). These samples are not discarded silently; they are reported separately "
            "as stress-test failures. This prevents the 8K failure case from dominating the steady-state latency plots.\n\n"
        )
        if not failures.empty:
            f.write("## Timeout-like failures\n\n")
            ff = failures[["system_mode", "load_mode", "variant", "image_short", "timeout_like_count", "n_raw", "timeout_like_frac"]].head(20)
            f.write(ff.to_markdown(index=False))
            f.write("\n\n")
        if not penalty.empty:
            f.write("## Largest contention penalties, steady-state\n\n")
            pp = penalty.sort_values("p99_busy_over_single", ascending=False)[[
                "system_mode", "variant", "image_short", "p99_ms_single", "p99_ms_busy", "p99_busy_over_single"
            ]].head(15)
            f.write(pp.to_markdown(index=False, floatfmt=".4g"))
            f.write("\n\n")
        if not compare.empty:
            f.write("## all_determinism vs unoptimized under busy contention\n\n")
            cc = compare[compare["load_mode"] == "busy"][[
                "system_mode", "image_short", "p99_ms_unopt", "p99_ms_all_det", "p99_pct_change_all_det_vs_unopt",
                "p99_minus_median_ms_unopt", "p99_minus_median_ms_all_det", "jitter_pct_change_all_det_vs_unopt"
            ]].sort_values("jitter_pct_change_all_det_vs_unopt")
            f.write(cc.to_markdown(index=False, floatfmt=".4g"))
            f.write("\n")

    print(f"Wrote clean figures to: {fig_dir}")
    print(f"Wrote clean tables to:  {tab_dir}")
    print("Most useful files:")
    for p in [
        fig_dir / "02_contention_penalty_p99_ratio.png",
        fig_dir / "03_core_p99_single_vs_busy.png",
        fig_dir / "04_busy_tail_jitter_p99_minus_median.png",
        fig_dir / "05_all_det_vs_unopt_busy_jitter_ratio.png",
        fig_dir / "06_all_det_vs_unopt_busy_p99_ratio.png",
        tab_dir / "report_snippets.md",
    ]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
