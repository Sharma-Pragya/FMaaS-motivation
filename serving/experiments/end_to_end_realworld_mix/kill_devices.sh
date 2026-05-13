#!/usr/bin/env bash
# Kill long-horizon device servers on all configured devices.
#
# Usage:
#   bash experiments/long_horizon/kill_devices.sh
#
# Optional:
#   PORT_CLEANUP=1 FIRST_PORT=8000 LAST_PORT=8999 bash experiments/long_horizon/kill_devices.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SERVING_DIR/.." && pwd)"

FIRST_PORT="${FIRST_PORT:-8000}"
LAST_PORT="${LAST_PORT:-8999}"
GRACE_SECS="${GRACE_SECS:-2}"
PORT_CLEANUP="${PORT_CLEANUP:-0}"

cd "$SERVING_DIR"

mapfile -t HOSTS < <(
    PYTHONPATH="$SERVING_DIR:$REPO_ROOT:${PYTHONPATH:-}" python - <<'PY'
from experiments.end_to_end_realworld_mix import user_config as cfg
for dev in cfg.devices.values():
    print(dev["ip"])
PY
)

read -r SSH_USER SSH_KEY < <(
    PYTHONPATH="$SERVING_DIR:$REPO_ROOT:${PYTHONPATH:-}" python - <<'PY'
from site_manager.config import username, ssh_key
print(username, ssh_key or "")
PY
)

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10)
if [[ -n "${SSH_KEY:-}" && "${SSH_KEY}" != "None" ]]; then
    SSH_KEY="${SSH_KEY/#\~/$HOME}"
    SSH_OPTS+=(-i "$SSH_KEY")
fi

REMOTE_CMD=$(cat <<EOF
set +e
echo "[kill] host=\$(hostname)"
gpu_pids="\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {print \$1}' | sort -u)"
if [[ -n "\${gpu_pids}" ]]; then
    echo "[kill] GPU compute PIDs: \${gpu_pids}"
    sudo kill -9 \${gpu_pids} >/dev/null 2>&1 || true
else
    echo "[kill] no GPU compute PIDs found"
fi
pkill -TERM -f "device/main.py" >/dev/null 2>&1 || true
sleep ${GRACE_SECS}
pkill -KILL -f "device/main.py" >/dev/null 2>&1 || true
if [[ "${PORT_CLEANUP}" == "1" ]]; then
    echo "[kill] port cleanup ${FIRST_PORT}-${LAST_PORT}"
    for port in \$(seq ${FIRST_PORT} ${LAST_PORT}); do
        fuser -TERM "\${port}/tcp" >/dev/null 2>&1 || true
    done
    sleep ${GRACE_SECS}
    for port in \$(seq ${FIRST_PORT} ${LAST_PORT}); do
        fuser -k "\${port}/tcp" >/dev/null 2>&1 || true
    done
fi
echo "[kill] done"
EOF
)

echo "[kill_devices] SSH user: ${SSH_USER}"
echo "[kill_devices] Hosts: ${HOSTS[*]}"
echo "[kill_devices] Port cleanup: ${PORT_CLEANUP} (${FIRST_PORT}-${LAST_PORT})"

for host in "${HOSTS[@]}"; do
    echo ""
    echo "================================================================"
    echo "[kill_devices] ${host}"
    echo "================================================================"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "$REMOTE_CMD")" || {
        echo "[kill_devices] WARNING: failed to clean ${host}" >&2
    }
done

echo ""
echo "[kill_devices] Done."
