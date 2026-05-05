#!/usr/bin/env bash
set -euo pipefail

# Run from the project root after gaussian_blur_comprehensive.sh has created results/.
# Optional overrides:
#   LATENCY_COLUMN=e2e_latency_us INPUT_UNIT=us ./run_analysis.sh
#   MAIN_IMAGE=suzanne_rgb LARGE_IMAGE=four_k_rgb ./run_analysis.sh

RESULTS_ROOT="${RESULTS_ROOT:-results}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis}"
FIG_DIR="${FIG_DIR:-figures}"
INPUT_UNIT="${INPUT_UNIT:-auto}"
LATENCY_COLUMN="${LATENCY_COLUMN:-}"
MAIN_IMAGE="${MAIN_IMAGE:-Scan_feb_1985_Belgium007}"
LARGE_IMAGE="${LARGE_IMAGE:-Two_Suzanne_heads}"

collect_args=(--results-root "$RESULTS_ROOT" --out-dir "$ANALYSIS_DIR" --input-unit "$INPUT_UNIT")
if [[ -n "$LATENCY_COLUMN" ]]; then
        collect_args+=(--latency-column "$LATENCY_COLUMN")
fi

python3 collect_latency_data.py "${collect_args[@]}"
python3 make_latency_plots.py \
        --analysis-dir "$ANALYSIS_DIR" \
        --fig-dir "$FIG_DIR" \
        --main-image "$MAIN_IMAGE" \
        --large-image "$LARGE_IMAGE"
