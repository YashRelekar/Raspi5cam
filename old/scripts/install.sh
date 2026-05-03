#!/usr/bin/env bash
# scripts/install.sh
#
# Install system-level packages and Python dependencies for Raspi5cam
# on Raspberry Pi OS Bookworm (64-bit), Raspberry Pi 5 + IMX219.
#
# Usage:
#   sudo bash scripts/install.sh
#
# Run as root (required for apt).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "=== Raspi5cam installer (Raspberry Pi 5 + IMX219) ==="
echo "Repository: ${REPO_DIR}"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    libatlas-base-dev \
    libopencv-dev \
    python3-opencv \
    python3-picamera2 \
    libjpeg-dev \
    libpng-dev \
    git \
    wget

# ── 2. Verify camera overlay in /boot/firmware/config.txt ────────────────────
CONFIG_FILE="/boot/firmware/config.txt"
if grep -q "dtoverlay=imx219" "${CONFIG_FILE}" 2>/dev/null; then
    echo "[ok] dtoverlay=imx219 already present in ${CONFIG_FILE}"
else
    echo ""
    echo "[WARN] dtoverlay=imx219,cam0 NOT found in ${CONFIG_FILE}."
    echo "       Add the following under the [all] section and reboot:"
    echo ""
    echo "         [all]"
    echo "         dtoverlay=imx219,cam0"
    echo ""
fi

# ── 3. Python virtual environment ─────────────────────────────────────────────
VENV_DIR="${REPO_DIR}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR} …"
    # --system-site-packages lets the venv use apt-installed picamera2/opencv
    "${PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# ── 4. Python dependencies ────────────────────────────────────────────────────
pip install -r "${REPO_DIR}/requirements.txt"

# tflite-runtime is the lightweight inference backend (no full TF needed on Pi)
pip install tflite-runtime || echo "tflite-runtime installation failed; try: pip install tensorflow"

# ── 5. Download ML models ─────────────────────────────────────────────────────
bash "${REPO_DIR}/scripts/download_models.sh"

echo ""
echo "=== Installation complete ==="
echo "Activate the virtual environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo "Then start emotion detection with:"
echo "    python emotion_detection.py"
echo ""
echo "NOTE: If you just added dtoverlay=imx219,cam0 to ${CONFIG_FILE},"
echo "      a reboot is required before the camera will be detected."
