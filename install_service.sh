#!/usr/bin/env bash
# ==============================================================================
# install_service.sh — Setup & Enable Litter Detection Service to Run on Boot
# ==============================================================================
set -euo pipefail

# Ensure script is run with sudo or root
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] This installer must be run as root or with sudo:"
    echo "        sudo bash install_service.sh"
    exit 1
fi

# Detect calling user and workspace path
REAL_USER="${SUDO_USER:-$(id -un)}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================================="
echo "Installing Litter Detection Systemd Service"
echo "=============================================================================="
echo "Target User      : ${REAL_USER}"
echo "Workspace Dir    : ${WORKDIR}"

# Clean up any redundant local unit templates or stale build artifacts
rm -f "${WORKDIR}/litter-detection.service" 2>/dev/null || true
find "${WORKDIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${WORKDIR}" -type d -name "__pycache__" -empty -delete 2>/dev/null || true
echo "[OK] Cleaned workspace of duplicate and temporary files."

# Determine Python executable (prefer workspace .venv, fall back to system python3)
if [ -x "${WORKDIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${WORKDIR}/.venv/bin/python"
elif [ -x "${WORKDIR}/.venv/bin/python3" ]; then
    PYTHON_BIN="${WORKDIR}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] No suitable Python interpreter found in .venv or system PATH."
    exit 1
fi
echo "Python Binary    : ${PYTHON_BIN}"

# Generate systemd service unit file directly in systemd directory
SERVICE_PATH="/etc/systemd/system/litter-detection.service"

cat <<EOF > "${SERVICE_PATH}"
[Unit]
Description=Litter Detection & GNSS Event Logger Service
After=network-online.target time-sync.target camera-relay.service
Wants=network-online.target
Requires=camera-relay.service

[Service]
Type=simple
User=${REAL_USER}
WorkingDirectory=${WORKDIR}
ExecStart=${PYTHON_BIN} ${WORKDIR}/litter_event_logger.py --headless --camera-id 10
Restart=always

RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=15s
StandardOutput=journal
StandardError=journal
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "${SERVICE_PATH}"
echo "[OK] Created systemd service unit: ${SERVICE_PATH}"

# Reset any stale failed state, reload systemd, and enable service on boot
systemctl reset-failed litter-detection.service 2>/dev/null || true
systemctl daemon-reload
systemctl enable litter-detection.service
echo "[OK] Service enabled to start automatically on boot."

echo ""
echo "=============================================================================="
echo "Litter Detection Service Successfully Installed & Ready!"
echo "=============================================================================="
echo "Service Commands:"
echo "  Start service now   : sudo systemctl start litter-detection"
echo "  Stop service        : sudo systemctl stop litter-detection"
echo "  Restart service     : sudo systemctl restart litter-detection"
echo "  Check live status   : sudo systemctl status litter-detection"
echo "  View live logs      : sudo journalctl -u litter-detection -f"
echo "=============================================================================="
