#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN = Path("analysis/long_latency.csv")
OUT = Path("analysis/emergency")
FIG = Path("figures/emergency")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# Normalise expected names.
if "image" not in df.columns and "image_stem" in df.columns:
    df["image"] = df["image_stem"]

required = {"system_mode", "load_mode", "variant", "image", "latency_ms"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

# Treat 8000 ms as timeout/saturation events.
# Do not delete them from the raw analysis; make both raw and filtered summaries.
df["is_timeout_like"] = df["latency_ms"] >= 7900.0

group_cols = ["system_mode", "load_mode", "variant", "image"]

def summarise(x):
    lat = x["latency_ms"].to_numpy()
    return pd.Series({
        "n": len(lat),
        "timeout_like_count": int(x["is_timeout_like"].sum()),
        "timeout_like_frac": float(x["is_timeout_like"].mean()),
        "mean_ms": np.mean(lat),
        "median_ms": np.median(lat),
        "std_ms": np.std(lat, ddof=1) if len(lat) > 1 else 0.0,
        "p95_ms": np.percentile(lat, 95),
        "p99_ms": np.percentile(lat, 99),
        "max_ms": np.max(lat),
        "p99_minus_median_ms": np.percentile(lat, 99) - np.median(lat),
        "max_minus_median_ms": np.max(lat) - np.median(lat),
    })

raw_summary = df.groupby(group_cols, dropna=False).apply(summarise).reset_index()
raw_summary.to_csv(OUT / "summary_raw_including_timeouts.csv", index=False)

filtered = df[~df["is_timeout_like"]].copy()
filtered_summary = filtered.groupby(group_cols, dropna=False).apply(summarise).reset_index()
filtered_summary.to_csv(OUT / "summary_filtered_excluding_8000ms.csv", index=False)

# Worst raw configurations: useful for saying contention can catastrophically break determinism.
worst = raw_summary.sort_values(
    ["timeout_like_frac", "p99_ms", "max_ms"],
    ascending=[False, False, False],
).head(30)
worst.to_csv(OUT / "worst_raw_configurations.csv", index=False)

# Contention penalty: busy p99 / single p99.
def make_contention_penalty(summary, name):
    single = summary[summary["load_mode"] == "single"].copy()
    busy = summary[summary["load_mode"] == "busy"].copy()

    keys = ["system_mode", "variant", "image"]
    merged = busy.merge(
        single,
        on=keys,
        suffixes=("_busy", "_single"),
        how="inner",
    )

    merged["p99_busy_over_single"] = merged["p99_ms_busy"] / merged["p99_ms_single"]
    merged["median_busy_over_single"] = merged["median_ms_busy"] / merged["median_ms_single"]
    merged["jitter_busy_over_single"] = (
        merged["p99_minus_median_ms_busy"] /
        merged["p99_minus_median_ms_single"].replace(0, np.nan)
    )

    merged.to_csv(OUT / f"contention_penalty_{name}.csv", index=False)
    return merged

penalty_raw = make_contention_penalty(raw_summary, "raw")
penalty_filtered = make_contention_penalty(filtered_summary, "filtered")

# Improvement table: all_determinism vs unoptimized.
def make_improvement(summary, name):
    base = summary[summary["variant"] == "unoptimized"].copy()
    opt = summary[summary["variant"] == "all_determinism"].copy()

    keys = ["system_mode", "load_mode", "image"]
    merged = opt.merge(
        base,
        on=keys,
        suffixes=("_all_det", "_unopt"),
        how="inner",
    )

    for metric in ["mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms", "p99_minus_median_ms"]:
        merged[f"{metric}_ratio_all_det_over_unopt"] = (
            merged[f"{metric}_all_det"] / merged[f"{metric}_unopt"].replace(0, np.nan)
        )
        merged[f"{metric}_pct_change_all_det_vs_unopt"] = (
            100.0 * (merged[f"{metric}_all_det"] - merged[f"{metric}_unopt"]) /
            merged[f"{metric}_unopt"].replace(0, np.nan)
        )

    merged.to_csv(OUT / f"all_determinism_vs_unoptimized_{name}.csv", index=False)
    return merged

improve_raw = make_improvement(raw_summary, "raw")
improve_filtered = make_improvement(filtered_summary, "filtered")

# Markdown summary for report.
with open(OUT / "report_ready_tables.md", "w") as f:
    f.write("# Emergency latency analysis tables\n\n")

    f.write("## Worst raw configurations, including 8000 ms timeout-like events\n\n")
    f.write(worst[[
        "system_mode", "load_mode", "variant", "image",
        "n", "timeout_like_count", "timeout_like_frac",
        "median_ms", "p99_ms", "max_ms",
        "p99_minus_median_ms"
    ]].to_markdown(index=False))
    f.write("\n\n")

    f.write("## Contention penalty, raw: busy p99 / single p99\n\n")
    tmp = penalty_raw.sort_values("p99_busy_over_single", ascending=False).head(30)
    f.write(tmp[[
        "system_mode", "variant", "image",
        "p99_ms_single", "p99_ms_busy", "p99_busy_over_single",
        "timeout_like_count_busy"
    ]].to_markdown(index=False))
    f.write("\n\n")

    f.write("## Contention penalty, filtered excluding >=7900 ms\n\n")
    tmp = penalty_filtered.sort_values("p99_busy_over_single", ascending=False).head(30)
    f.write(tmp[[
        "system_mode", "variant", "image",
        "p99_ms_single", "p99_ms_busy", "p99_busy_over_single",
    ]].to_markdown(index=False))
    f.write("\n\n")

    f.write("## all_determinism vs unoptimized, filtered\n\n")
    tmp = improve_filtered.sort_values("p99_ms_pct_change_all_det_vs_unopt").head(40)
    f.write(tmp[[
        "system_mode", "load_mode", "image",
        "p99_ms_unopt", "p99_ms_all_det",
        "p99_ms_pct_change_all_det_vs_unopt",
        "p99_minus_median_ms_unopt", "p99_minus_median_ms_all_det",
        "p99_minus_median_ms_pct_change_all_det_vs_unopt",
    ]].to_markdown(index=False))
    f.write("\n")

# Plot 1: timeout-like fraction by configuration.
timeout_plot = raw_summary.copy()
timeout_plot["label"] = (
    timeout_plot["system_mode"] + "/" +
    timeout_plot["load_mode"] + "/" +
    timeout_plot["variant"] + "/" +
    timeout_plot["image"].str.slice(0, 24)
)
timeout_plot = timeout_plot.sort_values("timeout_like_frac", ascending=False).head(25)

plt.figure(figsize=(14, 7))
plt.bar(range(len(timeout_plot)), timeout_plot["timeout_like_frac"])
plt.xticks(range(len(timeout_plot)), timeout_plot["label"], rotation=75, ha="right")
plt.ylabel("Fraction of samples >= 7900 ms")
plt.title("Timeout-like tail events by configuration")
plt.tight_layout()
plt.savefig(FIG / "timeout_like_fraction.png", dpi=200)
plt.close()

# Plot 2: contention penalty, raw.
plot = penalty_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["p99_busy_over_single"])
plot = plot.sort_values("p99_busy_over_single", ascending=False).head(25)
plot["label"] = plot["system_mode"] + "/" + plot["variant"] + "/" + plot["image"].str.slice(0, 24)

plt.figure(figsize=(14, 7))
plt.bar(range(len(plot)), plot["p99_busy_over_single"])
plt.axhline(1.0, linestyle="--")
plt.xticks(range(len(plot)), plot["label"], rotation=75, ha="right")
plt.ylabel("busy p99 / single p99")
plt.title("Contention penalty: how much busy-wait load inflates p99 latency")
plt.tight_layout()
plt.savefig(FIG / "contention_penalty_raw.png", dpi=200)
plt.close()

# Plot 3: all_determinism vs unoptimized, filtered, p99 ratio.
plot = improve_filtered.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["p99_ms_ratio_all_det_over_unopt"]
)
plot["label"] = plot["system_mode"] + "/" + plot["load_mode"] + "/" + plot["image"].str.slice(0, 24)
plot = plot.sort_values("p99_ms_ratio_all_det_over_unopt").head(30)

plt.figure(figsize=(14, 7))
plt.bar(range(len(plot)), plot["p99_ms_ratio_all_det_over_unopt"])
plt.axhline(1.0, linestyle="--")
plt.xticks(range(len(plot)), plot["label"], rotation=75, ha="right")
plt.ylabel("all_determinism p99 / unoptimized p99")
plt.title("Filtered p99 ratio: all_determinism vs unoptimized")
plt.tight_layout()
plt.savefig(FIG / "all_determinism_vs_unoptimized_p99_filtered.png", dpi=200)
plt.close()

# Plot 4: mean-vs-jitter tradeoff, filtered.
plot = filtered_summary.copy()
plot["label"] = plot["system_mode"] + "/" + plot["load_mode"] + "/" + plot["variant"]

plt.figure(figsize=(10, 7))
for (system_mode, load_mode), g in plot.groupby(["system_mode", "load_mode"]):
    plt.scatter(g["mean_ms"], g["p99_minus_median_ms"], label=f"{system_mode}/{load_mode}", s=40)
plt.xlabel("Mean latency (ms)")
plt.ylabel("p99 - median latency (ms)")
plt.title("Mean latency vs tail-jitter proxy, excluding >=7900 ms")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "mean_vs_tail_jitter_filtered.png", dpi=200)
plt.close()

print("Wrote:")
print(f"  {OUT}")
print(f"  {FIG}")
print()
print("Most useful report file:")
print(f"  {OUT / 'report_ready_tables.md'}")
