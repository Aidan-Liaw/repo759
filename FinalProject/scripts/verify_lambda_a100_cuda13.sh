#!/usr/bin/env bash
set -uo pipefail

FAILS=0
WARNS=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; WARNS=$((WARNS + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; FAILS=$((FAILS + 1)); }
section() { printf '\n==== %s ====\n' "$*"; }

ver_ge() {
    dpkg --compare-versions "$1" ge "$2"
}

cuda_ver_ge() {
    local have="$1" want="$2"
    local hm hn wm wn

    hm="$(printf '%s' "$have" | cut -d. -f1)"
    hn="$(printf '%s' "$have" | cut -d. -f2)"
    wm="$(printf '%s' "$want" | cut -d. -f1)"
    wn="$(printf '%s' "$want" | cut -d. -f2)"

    hn="${hn:-0}"
    wn="${wn:-0}"

    [ "$hm" -gt "$wm" ] || { [ "$hm" -eq "$wm" ] && [ "$hn" -ge "$wn" ]; }
}

section "System"
date -Is
id
uname -a
[ -f /etc/os-release ] && . /etc/os-release && printf 'OS: %s\n' "$PRETTY_NAME"

section "NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found after reboot."
else
    nvidia-smi || fail "nvidia-smi failed."

    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    branch="$(printf '%s' "$driver" | grep -oE '^[0-9]+')"

    printf 'Driver version: %s\n' "$driver"

    if [ "$branch" -ge 580 ] 2>/dev/null; then
        pass "Driver branch is >= 580, satisfying CUDA 13.x minimum driver branch."
    else
        fail "Driver branch is < 580."
    fi

    if ver_ge "$driver" "590.44.01"; then
        pass "Driver is >= 590.44.01, suitable for CUDA 13.1+ green-context Runtime API testing."
    else
        warn "Driver is < 590.44.01. CUDA 13.x may still work at branch 580+, but CUDA 13.1+ feature testing is safer on 590.44.01+."
    fi
fi

section "GPU identity and MIG visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_names="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true)"
    printf '%s\n' "$gpu_names"

    if printf '%s\n' "$gpu_names" | grep -qi 'A100'; then
        pass "Detected A100."
    else
        fail "A100 not detected."
    fi

    mig_state="$(nvidia-smi --query-gpu=index,name,mig.mode.current,mig.mode.pending --format=csv,noheader 2>&1 || true)"
    printf '%s\n' "$mig_state"

    if printf '%s\n' "$mig_state" | grep -qi 'not a valid field'; then
        fail "MIG query fields are not available."
    else
        pass "MIG query fields are available."
    fi
fi

section "CUDA Toolkit"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc not found."
else
    nvcc --version
    nvcc_ver="$(nvcc --version | awk -F'release ' '/release/ {print $2}' | cut -d, -f1 | head -n1)"

    if cuda_ver_ge "$nvcc_ver" "13.1"; then
        pass "CUDA Toolkit is >= 13.1."
    else
        fail "CUDA Toolkit is < 13.1."
    fi
fi

section "Green-context Runtime API header/library check"
header="$CUDA_HOME/include/cuda_runtime_api.h"
if [ ! -f "$header" ]; then
    fail "Missing $header."
else
    if grep -q 'cudaGreenCtxCreate' "$header" \
        && grep -q 'cudaDeviceGetDevResource' "$header" \
        && grep -q 'cudaExecutionContext_t' "$header"; then
        pass "cuda_runtime_api.h contains green-context Runtime API declarations."
    else
        fail "cuda_runtime_api.h does not contain green-context Runtime API declarations."
    fi
fi

libcudart="$(ldconfig -p 2>/dev/null | awk '/libcudart\.so/ {print $NF; exit}' || true)"
if [ -z "$libcudart" ]; then
    libcudart="$(find "$CUDA_HOME" -name 'libcudart.so*' 2>/dev/null | sort -V | tail -n1 || true)"
fi

if [ -z "$libcudart" ]; then
    fail "libcudart.so not found."
else
    printf 'libcudart: %s\n' "$libcudart"
    if nm -D "$libcudart" 2>/dev/null | grep -q 'cudaGreenCtxCreate'; then
        pass "libcudart exports cudaGreenCtxCreate."
    else
        fail "libcudart does not export cudaGreenCtxCreate."
    fi
fi

section "MPS tools"
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    pass "nvidia-cuda-mps-control found: $(command -v nvidia-cuda-mps-control)"

    tmp_root="$(mktemp -d)"
    export CUDA_MPS_PIPE_DIRECTORY="$tmp_root/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$tmp_root/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

    if nvidia-cuda-mps-control -d >"$tmp_root/mps.out" 2>&1; then
        sleep 1
        if echo get_server_list | nvidia-cuda-mps-control >"$tmp_root/server_list.out" 2>&1; then
            pass "MPS daemon starts and accepts commands."
        else
            fail "MPS daemon started but did not accept commands."
            cat "$tmp_root/server_list.out"
        fi
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    else
        fail "MPS daemon failed to start."
        cat "$tmp_root/mps.out"
    fi

    rm -rf "$tmp_root"
else
    fail "nvidia-cuda-mps-control not found."
fi

section "MIG enable probe"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "This verifier does not enable MIG automatically."
    echo "To test MIG state-changing permissions on a fresh instance:"
    echo "  sudo nvidia-smi -i 0 -mig 1"
    echo "Then verify:"
    echo "  nvidia-smi -i 0 --query-gpu=pci.bus_id,mig.mode.current,mig.mode.pending --format=csv"
    echo "On A100 VMs, pending-enabled MIG may require a reboot if GPU reset is blocked by the hypervisor."
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