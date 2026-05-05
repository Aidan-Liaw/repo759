#!/usr/bin/env zsh
#SBATCH --job-name=gaussian_blur
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:a100:1 -c 1
#SBATCH --constraint=a100
#SBATCH --output=Slurm-%j.out
#SBATCH --error=Slurm-%j.err

# University Slurm run: sbatch gaussian_blur_comprehensive.sh
# Lambda run:            ./gaussian_blur_comprehensive.sh

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        module load nvidia/cuda/13.0.0
fi

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
BUILD_ROOT="${BUILD_ROOT:-$PROJECT_DIR/build_tests}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"

ARCH="80"
R="${R:-3}"
THREADS_PER_BLOCK="${THREADS_PER_BLOCK:-256}"
TEST_ITERATIONS="${TEST_ITERATIONS:-1000}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-20}"
SIGMA="${SIGMA:-1.5}"
GPU_DEVICE_INDEX="${GPU_DEVICE_INDEX:-0}"
GREEN_CONTEXT_SM_COUNT="${GREEN_CONTEXT_SM_COUNT:-8}"
WRITE_OUTPUT_IMAGES="${WRITE_OUTPUT_IMAGES:-1}"
OUTPUT_IMAGE_EXTENSION="${OUTPUT_IMAGE_EXTENSION:-ppm}"

# CUDA 13.0 on the university servers only supports the Driver API Green Context path.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        GREEN_CONTEXT_API="${GREEN_CONTEXT_API:-2}"
        RUN_MIG_TESTS="${RUN_MIG_TESTS:-0}"
        RUN_MPS_TESTS="${RUN_MPS_TESTS:-0}"
else
        GREEN_CONTEXT_API="${GREEN_CONTEXT_API:-0}"
        RUN_MIG_TESTS="${RUN_MIG_TESTS:-1}"
        RUN_MPS_TESTS="${RUN_MPS_TESTS:-1}"
fi

CONFIGURE_MIG="${CONFIGURE_MIG:-$RUN_MIG_TESTS}"
DISABLE_MIG_AFTER="${DISABLE_MIG_AFTER:-0}"
MIG_PROFILE="${MIG_PROFILE:-1g.5gb}"

MPS_PIPE_DIRECTORY="${MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-${USER:-user}-${SLURM_JOB_ID:-$$}}"
MPS_LOG_DIRECTORY="${MPS_LOG_DIRECTORY:-$RESULT_ROOT/mps_logs}"
MPS_STATIC_PARTITIONING="${MPS_STATIC_PARTITIONING:-1}"
MPS_SM_PARTITION_CHUNKS="${MPS_SM_PARTITION_CHUNKS:-2}"
MPS_BUSY_SM_PARTITION_CHUNKS="${MPS_BUSY_SM_PARTITION_CHUNKS:-2}"
MPS_CREATE_BUSY_PARTITION="${MPS_CREATE_BUSY_PARTITION:-1}"
MPS_SM_PARTITION_ID=""
MPS_BUSY_SM_PARTITION_ID=""

images=(
        images_selected/*
)
mkdir -p "$BUILD_ROOT" "$RESULT_ROOT" "$OUTPUT_ROOT"

# REF: https://zsh.sourceforge.io/Doc/Release/Arithmetic-Evaluation.html
# REF: https://stackoverflow.com/questions/6022384/bash-tool-to-get-nth-line-from-a-file
# REF: https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
# REF: https://stackoverflow.com/questions/4181703/how-to-concatenate-string-variables-in-bash

mig_is_enabled() {
        nvidia-smi -L 2>/dev/null | grep -q "MIG"
}

configure_mig() {
        if [[ "$RUN_MIG_TESTS" != "1" ]]; then
                return 1
        fi

        if mig_is_enabled; then
                return 0
        fi

        if [[ "$CONFIGURE_MIG" != "1" ]]; then
                echo "Skipping MIG: MIG is not enabled." >&2
                return 1
        fi

        if ! sudo -n true 2>/dev/null; then
                echo "Skipping MIG: sudo is unavailable." >&2
                return 1
        fi

        sudo nvidia-smi -mig 1 || return 1
        sleep 5
        sudo nvidia-smi mig -dci >/dev/null 2>&1
        sudo nvidia-smi mig -dgi >/dev/null 2>&1
        sudo nvidia-smi mig -cgi "$MIG_PROFILE" -C || return 1
        nvidia-smi -L
        return 0
}

disable_mig() {
        if [[ "$DISABLE_MIG_AFTER" != "1" ]]; then
                return 0
        fi

        if sudo -n true 2>/dev/null; then
                sudo nvidia-smi mig -dci >/dev/null 2>&1
                sudo nvidia-smi mig -dgi >/dev/null 2>&1
                sudo nvidia-smi -mig 0 >/dev/null 2>&1
        fi
}

mps_gpu_uuid() {
        if [[ -n "${MPS_GPU_UUID:-}" ]]; then
                printf '%s\n' "$MPS_GPU_UUID"
                return 0
        fi

        nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null \
                | sed -n "$((GPU_DEVICE_INDEX + 1))p" \
                | tr -d '[:space:]'
}

mps_add_partition() {
        local label="$1"
        local chunks="$2"
        local gpu_uuid=""
        local partition_out=""
        local partition_id=""

        gpu_uuid="$(mps_gpu_uuid)"
        if [[ -z "$gpu_uuid" ]]; then
                echo "Failed to determine GPU UUID for $label MPS partition." >&2
                return 1
        fi

        partition_out="$(echo "sm_partition add $gpu_uuid $chunks" | nvidia-cuda-mps-control 2>&1)" || {
                echo "Failed to create $label MPS partition with $chunks chunks on $gpu_uuid." >&2
                echo "$partition_out" >&2
                return 1
        }

        partition_id="$(printf '%s\n' "$partition_out" | awk 'NF { line = $0 } END { print line }')"
        if [[ -z "$partition_id" ]]; then
                echo "MPS partition creation for $label returned an empty partition id." >&2
                echo "$partition_out" >&2
                return 1
        fi

        printf '%s\n' "$partition_id"
}

start_mps() {
        if [[ "$RUN_MPS_TESTS" != "1" ]]; then
                return 1
        fi

        mkdir -p "$MPS_PIPE_DIRECTORY" "$MPS_LOG_DIRECTORY"
        CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIRECTORY"
        CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIRECTORY"
        export CUDA_MPS_PIPE_DIRECTORY
        export CUDA_MPS_LOG_DIRECTORY

        if [[ "$MPS_STATIC_PARTITIONING" == "1" ]]; then
                nvidia-cuda-mps-control -d -S >/dev/null 2>&1 || return 1

                MPS_SM_PARTITION_ID="$(mps_add_partition main "$MPS_SM_PARTITION_CHUNKS")" || {
                        stop_mps
                        return 1
                }
                CUDA_MPS_SM_PARTITION="$MPS_SM_PARTITION_ID"
                export CUDA_MPS_SM_PARTITION
                echo "Using main MPS SM partition: $MPS_SM_PARTITION_ID" >&2

                if [[ "$MPS_CREATE_BUSY_PARTITION" == "1" ]]; then
                        MPS_BUSY_SM_PARTITION_ID="$(mps_add_partition busy "$MPS_BUSY_SM_PARTITION_CHUNKS")" || {
                                stop_mps
                                return 1
                        }
                        echo "Using busy-load MPS SM partition: $MPS_BUSY_SM_PARTITION_ID" >&2
                fi

                echo lspart | nvidia-cuda-mps-control >&2 || true
        else
                nvidia-cuda-mps-control -d >/dev/null 2>&1 || return 1
        fi

        return 0
}

stop_mps() {
        if [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
                echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
        fi

        unset CUDA_MPS_SM_PARTITION
        MPS_SM_PARTITION_ID=""
        MPS_BUSY_SM_PARTITION_ID=""
}

build_test() {
        local variant="$1"
        local system_mode="$2"
        local load_mode="$3"
        local run_busy_wait_only="$4"
        local build_name="${system_mode}_${load_mode}_${variant}"
        local build_dir="$BUILD_ROOT/$build_name"

        local use_fused="OFF"
        local use_graphs="OFF"
        local use_green="OFF"
        local use_streams="OFF"
        local use_pinned="OFF"
        local use_pitched="OFF"
        local use_shared="OFF"
        local use_constant="OFF"
        local assume_mask_uploaded="OFF"
        local write_output_images="OFF"

        if [[ "$WRITE_OUTPUT_IMAGES" == "1" ]]; then
                write_output_images="ON"
        fi

        case "$variant" in
                unoptimized)
                        ;;
                fusion_only)
                        use_fused="ON"
                        ;;
                graphs_only)
                        use_streams="ON"
                        use_graphs="ON"
                        ;;
                green_only)
                        use_streams="ON"
                        use_green="ON"
                        ;;
                all_determinism)
                        use_streams="ON"
                        use_fused="ON"
                        use_graphs="ON"
                        use_green="ON"
                        use_constant="ON"
                        assume_mask_uploaded="ON"
                        ;;
                busy_wait_only)
                        ;;
                *)
                        echo "Unknown variant: $variant" >&2
                        return 1
                        ;;
        esac

        local assume_mig="OFF"
        local assume_mps="OFF"
        if [[ "$system_mode" == "mig" || "$system_mode" == "mig_mps" ]]; then
                assume_mig="ON"
        fi
        if [[ "$system_mode" == "mps" || "$system_mode" == "mig_mps" ]]; then
                assume_mps="ON"
        fi

        local use_busy_wait="OFF"
        if [[ "$load_mode" == "busy" && "$run_busy_wait_only" != "1" ]]; then
                use_busy_wait="ON"
        fi

        cmake -S "$PROJECT_DIR" -B "$build_dir" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
                -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -Xptxas=-O3 -Xptxas=-v" \
                -DCMAKE_CXX_FLAGS_RELEASE="-O3 -Wall" \
                -DTEST_COMPREHENSIVE_CSV=ON \
                -DTEST_CSV_HEADER=ON \
                -DTEST_WRITE_OUTPUT_IMAGE="$write_output_images" \
                -DTEST_TEST_ITERATIONS="$TEST_ITERATIONS" \
                -DTEST_WARMUP_ITERATIONS="$WARMUP_ITERATIONS" \
                -DTEST_USE_RGB=ON \
                -DTEST_USE_PINNED_HOST_MEMORY="$use_pinned" \
                -DTEST_USE_PITCHED_LINEAR_MEMORY="$use_pitched" \
                -DTEST_USE_CUDA_STREAMS="$use_streams" \
                -DTEST_USE_CUDA_GRAPHS="$use_graphs" \
                -DTEST_USE_GREEN_CONTEXTS="$use_green" \
                -DTEST_GREEN_CONTEXT_API="$GREEN_CONTEXT_API" \
                -DTEST_GREEN_CONTEXT_SM_COUNT="$GREEN_CONTEXT_SM_COUNT" \
                -DTEST_USE_FUSED_GAUSSIAN="$use_fused" \
                -DTEST_USE_BUSY_WAIT_COMPETITOR="$use_busy_wait" \
                -DTEST_RUN_BUSY_WAIT_ONLY="$run_busy_wait_only" \
                -DTEST_ASSUME_MIG="$assume_mig" \
                -DTEST_ASSUME_MPS="$assume_mps" \
                -DCONVOLUTION_USE_SHARED_MEMORY="$use_shared" \
                -DCONVOLUTION_USE_CONSTANT_MASK="$use_constant" \
                -DCONVOLUTION_ASSUME_MASK_ALREADY_UPLOADED="$assume_mask_uploaded" \
                -DCONVOLUTION_ENABLE_FUSED_KERNEL=ON || return 1

        cmake --build "$build_dir" -j || return 1
        return 0
}

run_one_image() {
        local variant="$1"
        local system_mode="$2"
        local load_mode="$3"
        local image_path="$4"
        local build_name="${system_mode}_${load_mode}_${variant}"
        local executable="$BUILD_ROOT/$build_name/FinalProject"
        local image_stem="${image_path:t:r}"
        local output_dir="$RESULT_ROOT/$system_mode/$load_mode/$variant"
        local output_file="$output_dir/$image_stem.csv"
        local log_file="$output_dir/$image_stem.log"
        local image_output_dir="$OUTPUT_ROOT/$system_mode/$load_mode/$variant"
        local image_output_file="$image_output_dir/$image_stem.$OUTPUT_IMAGE_EXTENSION"

        mkdir -p "$output_dir"
        if [[ "$WRITE_OUTPUT_IMAGES" == "1" ]]; then
                mkdir -p "$image_output_dir"
        fi

        if [[ ! -f "$image_path" ]]; then
                echo "Skipping missing image: $image_path" >&2
                return 0
        fi

        echo "Running $system_mode/$load_mode/$variant/$image_stem" >&2
        if [[ "$WRITE_OUTPUT_IMAGES" == "1" ]]; then
                "$executable" "$image_path" "$R" "$THREADS_PER_BLOCK" "$TEST_ITERATIONS" "$SIGMA" "$GPU_DEVICE_INDEX" "$image_output_file" > "$output_file" 2> "$log_file"
        else
                "$executable" "$image_path" "$R" "$THREADS_PER_BLOCK" "$TEST_ITERATIONS" "$SIGMA" "$GPU_DEVICE_INDEX" > "$output_file" 2> "$log_file"
        fi
}

run_mps_busy_image() {
        local variant="$1"
        local system_mode="$2"
        local image_path="$3"
        local gaussian_build_name="${system_mode}_single_${variant}"
        local busy_build_name="${system_mode}_busy_busy_wait_only"
        local gaussian_executable="$BUILD_ROOT/$gaussian_build_name/FinalProject"
        local busy_executable="$BUILD_ROOT/$busy_build_name/FinalProject"
        local image_stem="${image_path:t:r}"
        local output_dir="$RESULT_ROOT/$system_mode/busy/$variant"
        local output_file="$output_dir/$image_stem.csv"
        local log_file="$output_dir/$image_stem.log"
        local busy_log_file="$output_dir/$image_stem.busy.log"
        local image_output_dir="$OUTPUT_ROOT/$system_mode/busy/$variant"
        local image_output_file="$image_output_dir/$image_stem.$OUTPUT_IMAGE_EXTENSION"
        local busy_pid=""

        mkdir -p "$output_dir"
        if [[ "$WRITE_OUTPUT_IMAGES" == "1" ]]; then
                mkdir -p "$image_output_dir"
        fi

        if [[ ! -f "$image_path" ]]; then
                echo "Skipping missing image: $image_path" >&2
                return 0
        fi

        if [[ -n "${MPS_BUSY_SM_PARTITION_ID:-}" ]]; then
                CUDA_MPS_SM_PARTITION="$MPS_BUSY_SM_PARTITION_ID" "$busy_executable" "$GPU_DEVICE_INDEX" > /dev/null 2> "$busy_log_file" &
        else
                "$busy_executable" "$GPU_DEVICE_INDEX" > /dev/null 2> "$busy_log_file" &
        fi
        busy_pid=$!
        sleep 2

        echo "Running $system_mode/busy/$variant/$image_stem with external MPS busy wait" >&2
        if [[ "$WRITE_OUTPUT_IMAGES" == "1" ]]; then
                "$gaussian_executable" "$image_path" "$R" "$THREADS_PER_BLOCK" "$TEST_ITERATIONS" "$SIGMA" "$GPU_DEVICE_INDEX" "$image_output_file" > "$output_file" 2> "$log_file"
        else
                "$gaussian_executable" "$image_path" "$R" "$THREADS_PER_BLOCK" "$TEST_ITERATIONS" "$SIGMA" "$GPU_DEVICE_INDEX" > "$output_file" 2> "$log_file"
        fi

        kill "$busy_pid" >/dev/null 2>&1
        wait "$busy_pid" >/dev/null 2>&1
}

run_variant_set() {
        local system_mode="$1"
        shift
        local variants=( "$@" )

        for variant in "${variants[@]}"; do
                if build_test "$variant" "$system_mode" "single" 0; then
                        for image_path in "${images[@]}"; do
                                run_one_image "$variant" "$system_mode" "single" "$image_path"
                        done
                else
                        echo "Skipping build failure: $system_mode/single/$variant" >&2
                fi

                if [[ "$system_mode" == "mps" || "$system_mode" == "mig_mps" ]]; then
                        if build_test "busy_wait_only" "$system_mode" "busy" 1 >/dev/null 2>&1 && build_test "$variant" "$system_mode" "single" 0; then
                                for image_path in "${images[@]}"; do
                                        run_mps_busy_image "$variant" "$system_mode" "$image_path"
                                done
                        else
                                echo "Skipping build failure: $system_mode/busy/$variant" >&2
                        fi
                else
                        if build_test "$variant" "$system_mode" "busy" 0; then
                                for image_path in "${images[@]}"; do
                                        run_one_image "$variant" "$system_mode" "busy" "$image_path"
                                done
                        else
                                echo "Skipping build failure: $system_mode/busy/$variant" >&2
                        fi
                fi
        done
}

run_system_mode() {
        local system_mode="$1"

        case "$system_mode" in
                none)
                        ;;
                mps)
                        start_mps || { echo "Skipping MPS tests." >&2; return 0; }
                        ;;
                mig)
                        configure_mig || { echo "Skipping MIG tests." >&2; return 0; }
                        ;;
                mig_mps)
                        configure_mig || { echo "Skipping MIG+MPS tests." >&2; return 0; }
                        start_mps || { echo "Skipping MIG+MPS tests." >&2; return 0; }
                        ;;
        esac

        if [[ "$system_mode" == "none" ]]; then
                run_variant_set "$system_mode" unoptimized fusion_only graphs_only green_only all_determinism
        else
                run_variant_set "$system_mode" unoptimized all_determinism
        fi

        if [[ "$system_mode" == "mps" || "$system_mode" == "mig_mps" ]]; then
                stop_mps
        fi
}

run_system_mode none

if [[ "$RUN_MPS_TESTS" == "1" ]]; then
        run_system_mode mps
fi

if [[ "$RUN_MIG_TESTS" == "1" ]]; then
        run_system_mode mig
        if [[ "$RUN_MPS_TESTS" == "1" ]]; then
                run_system_mode mig_mps
        fi
fi

stop_mps
disable_mig

echo "Results written to: $RESULT_ROOT" >&2
if [[ "$WRITE_OUTPUT_IMAGES" == "1" ]]; then
        echo "Output images written to: $OUTPUT_ROOT" >&2
fi
