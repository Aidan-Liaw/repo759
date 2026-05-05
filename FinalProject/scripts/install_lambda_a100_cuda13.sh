#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   chmod +x install_lambda_a100_cuda13.sh
#   ./install_lambda_a100_cuda13.sh
#
# Optional:
#   DRIVER_FLAVOR=proprietary ./install_lambda_a100_cuda13.sh
#   DRIVER_FLAVOR=open ./install_lambda_a100_cuda13.sh
#   AUTO_REBOOT=1 ./install_lambda_a100_cuda13.sh
#
# Defaults:
#   proprietary -> installs NVIDIA proprietary driver meta-package cuda-drivers
#   open        -> installs NVIDIA open kernel module meta-package nvidia-open

DRIVER_FLAVOR="${DRIVER_FLAVOR:-proprietary}"
AUTO_REBOOT="${AUTO_REBOOT:-0}"

DISTRO="ubuntu2204"
ARCH="x86_64"
CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/${CUDA_KEYRING_DEB}"

log()  { printf '\n==== %s ====\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"
}

log "Basic checks"
need_cmd sudo
need_cmd apt-get
need_cmd dpkg

if ! sudo -n true 2>/dev/null; then
    fail "Passwordless sudo failed. You need sudo for driver installation, MIG, and MPS administration."
fi
pass "Passwordless sudo works."

if [ -f /etc/os-release ]; then
    . /etc/os-release
    printf 'OS: %s\n' "${PRETTY_NAME:-unknown}"
    if [ "${VERSION_ID:-}" != "22.04" ]; then
        warn "This script is written for Ubuntu 22.04. Detected VERSION_ID=${VERSION_ID:-unknown}."
    fi
fi

printf 'Kernel: %s\n' "$(uname -r)"

log "Check that the A100 is visible on PCIe before driver install"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y pciutils ca-certificates gnupg wget curl lsb-release

if lspci | grep -iE 'nvidia|3d controller|vga compatible controller'; then
    pass "PCI devices include NVIDIA/accelerator hardware."
else
    warn "lspci did not show an NVIDIA device. If this is truly a 1x A100 instance, stop here and check the Lambda instance type."
fi

log "Install build prerequisites for NVIDIA kernel modules"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    dkms \
    linux-headers-"$(uname -r)" \
    pkg-config

log "Remove stale/conflicting Ubuntu NVIDIA packages if present"
sudo apt-get remove -y '^nvidia-.*' '^libnvidia-.*' || true
sudo apt-get autoremove -y || true

log "Install NVIDIA CUDA repository keyring"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cd "$tmpdir"

wget -q --show-progress "$CUDA_REPO_URL"
sudo dpkg -i "$CUDA_KEYRING_DEB"

sudo apt-get update

log "Show available NVIDIA driver/toolkit candidates"
apt-cache policy cuda-drivers || true
apt-cache policy nvidia-open || true
apt-cache policy cuda-toolkit-13 || true
apt-cache policy cuda-toolkit || true

log "Install NVIDIA driver"
case "$DRIVER_FLAVOR" in
    proprietary)
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers
        ;;
    open)
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-open
        ;;
    *)
        fail "Unknown DRIVER_FLAVOR=$DRIVER_FLAVOR. Use proprietary or open."
        ;;
esac

log "Install CUDA Toolkit 13.x"
if apt-cache policy cuda-toolkit-13 | grep -q 'Candidate: (none)'; then
    warn "cuda-toolkit-13 is not available; falling back to cuda-toolkit."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit
else
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13
fi

log "Configure CUDA environment"
cuda_dir="$(find /usr/local -maxdepth 1 -type d -name 'cuda-13*' | sort -V | tail -n1 || true)"

if [ -z "$cuda_dir" ] && [ -d /usr/local/cuda ]; then
    cuda_dir="/usr/local/cuda"
fi

if [ -z "$cuda_dir" ]; then
    warn "Could not find /usr/local/cuda-13* yet. This may resolve after package post-install steps."
else
    sudo ln -sfn "$cuda_dir" /usr/local/cuda

    sudo tee /etc/profile.d/cuda.sh >/dev/null <<'EOF'
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
EOF

    sudo tee /etc/ld.so.conf.d/cuda.conf >/dev/null <<'EOF'
/usr/local/cuda/lib64
EOF

    sudo ldconfig
    pass "CUDA environment configured for $cuda_dir."
fi

log "Write post-reboot verifier"
cat > "$HOME/verify_lambda_a100_cuda13.sh" <<'EOF'
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
EOF

chmod +x "$HOME/verify_lambda_a100_cuda13.sh"
pass "Verifier written to $HOME/verify_lambda_a100_cuda13.sh"

log "Installation finished"
echo "You should now reboot."
echo "After reboot, run:"
echo "  ~/verify_lambda_a100_cuda13.sh"

if [ "$AUTO_REBOOT" = "1" ]; then
    log "AUTO_REBOOT=1, rebooting now"
    sudo reboot
fi