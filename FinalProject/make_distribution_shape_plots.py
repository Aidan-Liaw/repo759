#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analysis/long_latency.csv")

if "image" not in df.columns and "image_stem" in df.columns:
    df["image"] = df["image_stem"]

# Exclude timeout-like samples from steady-state distribution plots.
df = df[df["latency_ms"] < 7900].copy()

OUT = Path("figures/report_clean")
OUT.mkdir(parents=True, exist_ok=True)

IMAGE = "Two_Suzanne_heads"

configs = [
    ("none", "single", "unoptimized", "No isolation / single / unoptimized"),
    ("none", "busy", "unoptimized", "No isolation / busy / unoptimized"),
    ("none", "busy", "all_determinism", "No isolation / busy / all determinism"),
    ("mig", "busy", "unoptimized", "MIG / busy / unoptimized"),
    ("mig", "busy", "all_determinism", "MIG / busy / all determinism"),
]

# 1. Latency-vs-iteration scatter/time-series.
plt.figure(figsize=(11, 6))

for system_mode, load_mode, variant, label in configs:
    g = df[
        (df["image"] == IMAGE)
        & (df["system_mode"] == system_mode)
        & (df["load_mode"] == load_mode)
        & (df["variant"] == variant)
    ].copy()

    if g.empty:
        continue

    g = g.sort_values("sample_idx")
    plt.scatter(g["sample_idx"], g["latency_ms"], s=6, alpha=0.45, label=label)

plt.xlabel("Sample index")
plt.ylabel("E2E latency (ms)")
plt.title(f"Raw latency samples over time on {IMAGE}")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "09_latency_vs_iteration_suzanne.png", dpi=200)
plt.close()

# 2. ECDF plot.
plt.figure(figsize=(11, 6))

for system_mode, load_mode, variant, label in configs:
    g = df[
        (df["image"] == IMAGE)
        & (df["system_mode"] == system_mode)
        & (df["load_mode"] == load_mode)
        & (df["variant"] == variant)
    ].copy()

    if g.empty:
        continue

    x = np.sort(g["latency_ms"].to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)

    plt.plot(x, y, label=label)

plt.xlabel("E2E latency (ms)")
plt.ylabel("Fraction of samples ≤ latency")
plt.title(f"Latency ECDF on {IMAGE}, excluding timeout-like samples")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "10_latency_ecdf_suzanne.png", dpi=200)
plt.close()

# 3. Jittered strip plot for shape.
rows = []
labels = []

for idx, (system_mode, load_mode, variant, label) in enumerate(configs):
    g = df[
        (df["image"] == IMAGE)
        & (df["system_mode"] == system_mode)
        & (df["load_mode"] == load_mode)
        & (df["variant"] == variant)
    ].copy()

    if g.empty:
        continue

    rng = np.random.default_rng(1234 + idx)
    x = idx + rng.normal(0, 0.035, size=len(g))
    plt.scatter(x, g["latency_ms"], s=6, alpha=0.35)
    rows.append(idx)
    labels.append(label)

plt.xticks(rows, labels, rotation=35, ha="right")
plt.ylabel("E2E latency (ms)")
plt.title(f"Raw latency distribution shape on {IMAGE}")
plt.tight_layout()
plt.savefig(OUT / "11_latency_strip_suzanne.png", dpi=200)
plt.close()

print("Wrote:")
print(OUT / "09_latency_vs_iteration_suzanne.png")
print(OUT / "10_latency_ecdf_suzanne.png")
print(OUT / "11_latency_strip_suzanne.png")
