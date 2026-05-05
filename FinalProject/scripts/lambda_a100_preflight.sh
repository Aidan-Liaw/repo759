#!/usr/bin/env bash
set -uo pipefail

MIN_CUDA13_DRIVER_BRANCH=580
MIN_GREENCTX_DRIVER_FULL="590.44.01"
MIN_GREENCTX_TOOLKIT="13.1"

FAILS=0
WARNS=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; WARNS=$((WARNS + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; FAILS=$((FAILS + 1)); }

section() {
    printf '\n==== %s ====\n' "$*"
}

cmd_exists() {
    command -v "$1" >/dev/null 2>&1
}

ver_ge() {
    # Ubuntu has dpkg; this handles versions like 590.44.01 correctly.
    dpkg --compare-versions "$1" ge "$2"
}

first_num() {
    printf '%s' "$1" | grep -oE '^[0-9]+' || true
}

cuda_ver_ge() {
    # Compare major.minor strings, e.g. 13.2 >= 13.1.
    local have="$1"
    local want="$2"
    local have_major have_minor want_major want_minor

    have_major="$(printf '%s' "$have" | cut -d. -f1)"
    have_minor="$(printf '%s' "$have" | cut -d. -f2)"
    want_major="$(printf '%s' "$want" | cut -d. -f1)"
    want_minor="$(printf '%s' "$want" | cut -d. -f2)"

    have_minor="${have_minor:-0}"
    want_minor="${want_minor:-0}"

    if [ "$have_major" -gt "$want_major" ]; then
        return 0
    elif [ "$have_major" -eq "$want_major" ] && [ "$have_minor" -ge "$want_minor" ]; then
        return 0
    else
        return 1
    fi
}

find_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME" ]; then
        printf '%s\n' "$CUDA_HOME"
        return
    fi

    if [ -n "${CUDA_PATH:-}" ] && [ -d "$CUDA_PATH" ]; then
        printf '%s\n' "$CUDA_PATH"
        return
    fi

    if [ -d /usr/local/cuda ]; then
        printf '%s\n' /usr/local/cuda
        return
    fi

    local newest
    newest="$(find /usr/local -maxdepth 1 -type d -name 'cuda-*' 2>/dev/null | sort -V | tail -n1 || true)"
    if [ -n "$newest" ]; then
        printf '%s\n' "$newest"
    fi
}

get_nvcc_version() {
    local nvcc_bin="$1"
    "$nvcc_bin" --version 2>/dev/null \
        | awk -F'release ' '/release/ {print $2}' \
        | cut -d, -f1 \
        | head -n1
}

find_file_first() {
    for p in "$@"; do
        if [ -e "$p" ]; then
            printf '%s\n' "$p"
            return
        fi
    done
}

section "Basic system identity"
printf 'Date: %s\n' "$(date -Is)"
printf 'User: %s\n' "$(id)"
printf 'Kernel: %s\n' "$(uname -a)"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    printf 'OS: %s\n' "${PRETTY_NAME:-unknown}"
fi

section "Sudo access"
if ! cmd_exists sudo; then
    fail "sudo is not installed."
else
    if sudo -n true 2>/dev/null; then
        pass "Passwordless sudo works. This is sufficient for driver/MIG/MPS administration checks."
    else
        fail "Passwordless sudo failed. On a Lambda Cloud base image this usually means sudo is unavailable, requires a password you may not have, or is blocked."
        warn "You can manually test interactive sudo with: sudo -v"
    fi
fi

section "NVIDIA driver and CUDA driver API support"
if ! cmd_exists nvidia-smi; then
    fail "nvidia-smi not found. NVIDIA driver is not installed or not on PATH."
else
    nvidia-smi || fail "nvidia-smi exists but failed to query the GPU."

    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
    driver_branch="$(first_num "$driver_version")"

    if [ -z "$driver_version" ]; then
        fail "Could not read NVIDIA driver version."
    else
        printf 'Detected NVIDIA driver: %s\n' "$driver_version"

        if [ "$driver_branch" -ge "$MIN_CUDA13_DRIVER_BRANCH" ] 2>/dev/null; then
            pass "Driver branch is >= ${MIN_CUDA13_DRIVER_BRANCH}; CUDA 13.x driver compatibility requirement is met."
        else
            fail "Driver branch is < ${MIN_CUDA13_DRIVER_BRANCH}; current driver does not meet CUDA 13.x support."
        fi

        if ver_ge "$driver_version" "$MIN_GREENCTX_DRIVER_FULL"; then
            pass "Driver is >= ${MIN_GREENCTX_DRIVER_FULL}; suitable for CUDA 13.1+ green-context Runtime API work."
        else
            warn "Driver is < ${MIN_GREENCTX_DRIVER_FULL}. CUDA 13.x minor compatibility may still work at branch 580+, but CUDA 13.1+ feature testing is safer with 590.44.01+."
        fi
    fi

    smi_cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1 || true)"
    if [ -n "$smi_cuda" ]; then
        printf 'nvidia-smi reported CUDA driver API max version: %s\n' "$smi_cuda"
    else
        warn "Could not parse CUDA Version from nvidia-smi output."
    fi
fi

section "A100 / MIG capability"
if cmd_exists nvidia-smi; then
    gpu_names="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true)"
    printf '%s\n' "$gpu_names"

    if printf '%s\n' "$gpu_names" | grep -qi 'A100'; then
        pass "Detected an NVIDIA A100-class GPU."
    else
        fail "Did not detect A100 in nvidia-smi GPU name output."
    fi

    mig_query="$(nvidia-smi --query-gpu=index,name,mig.mode.current,mig.mode.pending --format=csv,noheader 2>&1 || true)"
    if printf '%s\n' "$mig_query" | grep -qi 'not a valid field'; then
        fail "Driver/nvidia-smi does not expose MIG query fields."
    else
        printf '%s\n' "$mig_query"
        if printf '%s\n' "$mig_query" | grep -qi 'Enabled'; then
            pass "MIG is already enabled on at least one GPU."
        elif printf '%s\n' "$mig_query" | grep -qi 'Disabled'; then
            warn "MIG is supported/queryable but currently disabled. Use the MIG toggle probe below to test whether Lambda allows enabling it."
        else
            warn "MIG query returned unexpected output; inspect the lines above."
        fi
    fi
fi

section "CUDA Toolkit / Runtime API for green contexts"
cuda_home="$(find_cuda_home || true)"
if [ -z "$cuda_home" ]; then
    fail "Could not find CUDA Toolkit under CUDA_HOME, CUDA_PATH, /usr/local/cuda, or /usr/local/cuda-*."
else
    printf 'Detected CUDA home: %s\n' "$cuda_home"

    nvcc_bin=""
    if cmd_exists nvcc; then
        nvcc_bin="$(command -v nvcc)"
    elif [ -x "$cuda_home/bin/nvcc" ]; then
        nvcc_bin="$cuda_home/bin/nvcc"
    fi

    if [ -z "$nvcc_bin" ]; then
        fail "nvcc not found. For CUDA Runtime API green-context development, install CUDA Toolkit 13.1+."
    else
        nvcc_version="$(get_nvcc_version "$nvcc_bin")"
        printf 'Detected nvcc: %s\n' "$nvcc_bin"
        printf 'Detected CUDA Toolkit release from nvcc: %s\n' "${nvcc_version:-unknown}"

        if [ -n "$nvcc_version" ] && cuda_ver_ge "$nvcc_version" "$MIN_GREENCTX_TOOLKIT"; then
            pass "CUDA Toolkit is >= ${MIN_GREENCTX_TOOLKIT}; green contexts should be exposed in the Runtime API headers/libraries."
        else
            fail "CUDA Toolkit is < ${MIN_GREENCTX_TOOLKIT}; green contexts are not available through the CUDA Runtime API."
        fi
    fi

    header="$(find_file_first \
        "$cuda_home/include/cuda_runtime_api.h" \
        /usr/local/cuda/include/cuda_runtime_api.h \
        /usr/local/cuda-*/include/cuda_runtime_api.h 2>/dev/null || true)"

    if [ -z "$header" ]; then
        fail "cuda_runtime_api.h not found."
    else
        printf 'Detected CUDA runtime header: %s\n' "$header"
        if grep -q 'cudaGreenCtxCreate' "$header" \
            && grep -q 'cudaDeviceGetDevResource' "$header" \
            && grep -q 'cudaExecutionContext_t' "$header"; then
            pass "CUDA Runtime API header contains green-context APIs."
        else
            fail "CUDA Runtime API header does not contain green-context APIs."
        fi
    fi

    libcudart="$(ldconfig -p 2>/dev/null | awk '/libcudart\.so/ {print $NF; exit}' || true)"
    if [ -z "$libcudart" ]; then
        libcudart="$(find "$cuda_home" /usr/local/cuda* -path '*lib*' -name 'libcudart.so*' 2>/dev/null | sort -V | tail -n1 || true)"
    fi

    if [ -z "$libcudart" ]; then
        fail "libcudart.so not found."
    else
        printf 'Detected libcudart: %s\n' "$libcudart"
        if cmd_exists nm; then
            if nm -D "$libcudart" 2>/dev/null | grep -q 'cudaGreenCtxCreate'; then
                pass "libcudart exports cudaGreenCtxCreate."
            else
                fail "libcudart does not appear to export cudaGreenCtxCreate."
            fi
        elif cmd_exists strings; then
            if strings "$libcudart" | grep -q 'cudaGreenCtxCreate'; then
                pass "libcudart appears to contain cudaGreenCtxCreate."
            else
                fail "libcudart does not appear to contain cudaGreenCtxCreate."
            fi
        else
            warn "Neither nm nor strings found; could not inspect libcudart symbols."
        fi
    fi
fi

section "Driver upgrade path via apt dry-run"
if ! cmd_exists apt-cache; then
    warn "apt-cache not found; cannot inspect Ubuntu/NVIDIA driver packages."
else
    mapfile -t driver_pkgs < <(
        apt-cache search --names-only '^nvidia-driver-[0-9]+(-server|-open)?$' \
            | awk '{print $1}' \
            | sort -u
    )

    best_pkg=""
    best_branch=0

    for pkg in "${driver_pkgs[@]}"; do
        branch="$(printf '%s\n' "$pkg" | sed -n 's/^nvidia-driver-\([0-9][0-9]*\).*/\1/p')"
        candidate="$(apt-cache policy "$pkg" 2>/dev/null | awk '/Candidate:/ {print $2}')"

        if [ -n "$branch" ] && [ "$branch" -ge "$MIN_CUDA13_DRIVER_BRANCH" ] 2>/dev/null \
            && [ -n "$candidate" ] && [ "$candidate" != "(none)" ]; then
            printf 'Found CUDA 13-capable driver package candidate: %s -> %s\n' "$pkg" "$candidate"

            if [ "$branch" -gt "$best_branch" ]; then
                best_branch="$branch"
                best_pkg="$pkg"
            fi
        fi
    done

    if [ -z "$best_pkg" ]; then
        warn "No apt-visible nvidia-driver-580+ package candidate found. You may need to add NVIDIA's CUDA Ubuntu 22.04 repository before upgrading."
    else
        pass "Best apt-visible NVIDIA driver package candidate: $best_pkg"

        if sudo -n true 2>/dev/null; then
            tmp_dryrun="$(mktemp)"
            if sudo -n apt-get -s install "$best_pkg" >"$tmp_dryrun" 2>&1; then
                pass "apt dry-run install succeeded for $best_pkg."
            else
                fail "apt dry-run install failed for $best_pkg. Inspect: $tmp_dryrun"
                tail -n 40 "$tmp_dryrun"
            fi
        else
            warn "Skipping apt install dry-run because passwordless sudo was not confirmed."
        fi
    fi
fi

section "MPS availability and non-destructive daemon test"
mps_control="$(command -v nvidia-cuda-mps-control || true)"
mps_server="$(command -v nvidia-cuda-mps-server || true)"

if [ -z "$mps_control" ]; then
    fail "nvidia-cuda-mps-control not found."
else
    pass "Found nvidia-cuda-mps-control at $mps_control"

    if strings "$mps_control" 2>/dev/null | grep -q -- '--static-partitioning'; then
        pass "MPS control binary appears to support --static-partitioning / -S."
    else
        warn "Could not confirm --static-partitioning support from binary strings. This may indicate an older MPS binary."
    fi
fi

if [ -z "$mps_server" ]; then
    warn "nvidia-cuda-mps-server not found on PATH. The control daemon may still know where it is, but PATH is incomplete."
else
    pass "Found nvidia-cuda-mps-server at $mps_server"
fi

if [ -n "$mps_control" ]; then
    tmp_root="$(mktemp -d)"
    export CUDA_MPS_PIPE_DIRECTORY="$tmp_root/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$tmp_root/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

    mps_out="$tmp_root/mps.out"

    if "$mps_control" -d >"$mps_out" 2>&1; then
        sleep 1
        if echo get_server_list | "$mps_control" >"$tmp_root/server_list.out" 2>&1; then
            pass "MPS control daemon starts and accepts commands with per-user pipe/log directories."
        else
            fail "MPS control daemon started but did not accept commands."
            cat "$tmp_root/server_list.out"
        fi

        echo quit | "$mps_control" >/dev/null 2>&1 || true
    else
        fail "MPS control daemon failed to start."
        cat "$mps_out"
    fi

    rm -rf "$tmp_root"
fi

section "Summary"
printf 'Failures: %d\n' "$FAILS"
printf 'Warnings: %d\n' "$WARNS"

if [ "$FAILS" -eq 0 ]; then
    printf 'OVERALL: PASS\n'
    exit 0
else
    printf 'OVERALL: FAIL\n'
    exit 1
fi