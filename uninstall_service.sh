#!/usr/bin/env bash
# ==============================================================================
# uninstall_service.sh — Disable & Remove Litter Detection Service
# ==============================================================================
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] This uninstaller must be run as root or with sudo:"
    echo "        sudo bash uninstall_service.sh"
    exit 1
fi

SERVICE_NAME="litter-detection.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping and disabling ${SERVICE_NAME}..."
systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
systemctl reset-failed "${SERVICE_NAME}" 2>/dev/null || true

# Remove unit file and any wants symlinks
if [ -f "${SERVICE_PATH}" ]; then
    rm -f "${SERVICE_PATH}"
    echo "[OK] Removed ${SERVICE_PATH}"
fi

rm -f "/etc/systemd/system/multi-user.target.wants/${SERVICE_NAME}" 2>/dev/null || true

# Clean up any leftover duplicate local unit files or stale build artifacts
rm -f "${WORKDIR}/${SERVICE_NAME}" 2>/dev/null || true
find "${WORKDIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${WORKDIR}" -type d -name "__pycache__" -empty -delete 2>/dev/null || true
echo "[OK] Cleaned workspace and removed orphaned service links."

systemctl daemon-reload
echo ""
echo "=============================================================================="
echo "Litter Detection service has been completely uninstalled and cleaned up."
echo "=============================================================================="
