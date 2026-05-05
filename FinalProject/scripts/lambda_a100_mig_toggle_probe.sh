#!/usr/bin/env bash
set -uo pipefail

GPU_ID="${GPU_ID:-0}"
ALLOW_MIG_TOGGLE="${ALLOW_MIG_TOGGLE:-0}"
RESTORE_MIG="${RESTORE_MIG:-1}"

fail() { printf '[FAIL] %s\n' "$*"; exit 1; }
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

cmd_exists() {
    command -v "$1" >/dev/null 2>&1
}

mig_state() {
    nvidia-smi \
        --query-gpu=index,name,mig.mode.current,mig.mode.pending \
        --format=csv,noheader \
        -i "$GPU_ID" 2>/dev/null || true
}

mig_current() {
    mig_state | awk -F, '{gsub(/^ +| +$/, "", $3); print $3}' | head -n1
}

mig_pending() {
    mig_state | awk -F, '{gsub(/^ +| +$/, "", $4); print $4}' | head -n1
}

printf 'GPU_ID=%s\n' "$GPU_ID"
printf 'ALLOW_MIG_TOGGLE=%s\n' "$ALLOW_MIG_TOGGLE"
printf 'RESTORE_MIG=%s\n' "$RESTORE_MIG"

cmd_exists nvidia-smi || fail "nvidia-smi not found."
cmd_exists sudo || fail "sudo not found."

if ! sudo -n true 2>/dev/null; then
    fail "Passwordless sudo failed. Cannot verify MIG administration capability."
fi
pass "Passwordless sudo works."

printf '\n==== GPU identity ====\n'
nvidia-smi -L || fail "nvidia-smi -L failed."

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" 2>/dev/null | head -n1 || true)"
printf 'Selected GPU name: %s\n' "$gpu_name"

if ! printf '%s\n' "$gpu_name" | grep -qi 'A100'; then
    fail "Selected GPU is not reported as A100."
fi
pass "Selected GPU is A100-class."

printf '\n==== Running compute processes ====\n'
apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [ -n "$apps" ]; then
    printf '%s\n' "$apps"
    warn "There are active compute processes. MIG toggling may fail or kill/interrupt work."
else
    pass "No running compute processes reported by nvidia-smi."
fi

printf '\n==== Current MIG state ====\n'
state_before="$(mig_state)"
printf '%s\n' "$state_before"

current_before="$(mig_current)"
pending_before="$(mig_pending)"

if [ "$current_before" = "Enabled" ]; then
    pass "MIG is already enabled."

    printf '\n==== MIG profiles ====\n'
    sudo -n nvidia-smi mig -lgip || warn "Could not list GPU instance profiles."

    exit 0
fi

if [ "$ALLOW_MIG_TOGGLE" != "1" ]; then
    warn "MIG is not currently enabled."
    fail "Set ALLOW_MIG_TOGGLE=1 to actually test whether sudo nvidia-smi -i $GPU_ID -mig 1 works."
fi

printf '\n==== Enabling MIG ====\n'
set +e
enable_out="$(sudo -n nvidia-smi -i "$GPU_ID" -mig 1 2>&1)"
enable_status=$?
set -e 2>/dev/null || true

printf '%s\n' "$enable_out"

if [ "$enable_status" -ne 0 ]; then
    fail "MIG enable command failed. This likely means the cloud/hypervisor/driver setup does not permit MIG mode changes from inside the instance."
fi

sleep 2

printf '\n==== MIG state after enable attempt ====\n'
state_after="$(mig_state)"
printf '%s\n' "$state_after"

current_after="$(mig_current)"
pending_after="$(mig_pending)"

if [ "$current_after" = "Enabled" ]; then
    pass "MIG enable succeeded immediately."
elif [ "$pending_after" = "Enabled" ]; then
    warn "MIG is pending enable. On A100 VM passthrough this can require a reboot if GPU reset is blocked."
else
    fail "MIG did not become enabled or pending-enabled after the command."
fi

printf '\n==== MIG profiles ====\n'
sudo -n nvidia-smi mig -lgip || warn "Could not list GPU instance profiles. This may fail until reboot if MIG is pending-enabled."

if [ "$RESTORE_MIG" = "1" ]; then
    printf '\n==== Restoring MIG disabled state ====\n'
    set +e
    restore_out="$(sudo -n nvidia-smi -i "$GPU_ID" -mig 0 2>&1)"
    restore_status=$?
    set -e 2>/dev/null || true

    printf '%s\n' "$restore_out"

    if [ "$restore_status" -eq 0 ]; then
        pass "MIG disable command succeeded."
    else
        warn "MIG disable command failed. The instance may need reboot/repave to clear pending MIG state."
    fi

    printf '\n==== Final MIG state ====\n'
    mig_state
else
    warn "RESTORE_MIG=0, leaving MIG state as-is."
fi

printf '\nOVERALL: MIG admin probe completed.\n'