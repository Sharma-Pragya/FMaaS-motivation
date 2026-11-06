#!/bin/bash
# Sourced by run_<device>.sh. Confirms the GPU in this machine matches the
# device label the script writes results under.
#
# A mislabelled result is the worst failure mode here: it lands in
# results/<backbone>/<device>/ looking legitimate, and nothing downstream can
# tell that an L4 number is filed under a10. Cheap to check, expensive to miss.
#
# Expects EXPECT_GPU (a substring of the nvidia-smi product name) to be set.
# Set FORCE=1 to run anyway — e.g. a device name this check does not know.

check_gpu() {
    local expect="$1" label="$2" name

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[guard] WARNING: nvidia-smi not found — cannot verify this is a ${label}." >&2
        echo "[guard]          Results will be filed under '${label}'. Ctrl-C now if wrong." >&2
        sleep 5
        return 0
    fi

    # nvidia-smi can exist but fail (driver not loaded), and it prints that
    # failure to stdout, not stderr. Under `set -e` with pipefail the failing
    # pipeline inside this assignment would abort the caller, so swallow the
    # status and validate the string instead.
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    if [[ -z "$name" || "$name" == *"failed"* || "$name" == *"Error"* ]]; then
        echo "[guard] WARNING: nvidia-smi gave no usable GPU name — cannot verify this is a ${label}." >&2
        echo "[guard]          Results will be filed under '${label}'. Ctrl-C now if wrong." >&2
        sleep 5
        return 0
    fi

    if [[ "$name" == *"$expect"* ]]; then
        echo "[guard] GPU check OK: '${name}' matches expected '${expect}'."
        return 0
    fi

    if [[ -n "${FORCE:-}" ]]; then
        echo "[guard] WARNING: GPU is '${name}', expected '${expect}' — FORCE=1, continuing." >&2
        return 0
    fi

    echo "[guard] ERROR: this machine has '${name}', but this script files results" >&2
    echo "[guard]        under device label '${label}' (expects '${expect}')." >&2
    echo "[guard]        Run the matching run_<device>.sh, or set FORCE=1 to override." >&2
    exit 1
}
