#!/usr/bin/env bash
# scripts/download_models.sh
#
# Obtain the TFLite model required by Raspi5cam emotion detection.
#
# face_emotion.tflite
#   Downloaded from this project's GitHub Releases ('models-latest' tag).
#   Falls back to a local build using scripts/build_face_model.py if the
#   release asset is not yet available (requires TensorFlow).
#
# Usage:
#   bash scripts/download_models.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_DIR}/models"
RELEASE_URL="https://github.com/YashRelekar/Raspi5cam/releases/download/models-latest"

mkdir -p "${MODELS_DIR}"

# ---------------------------------------------------------------------------
# Helper: download a file from GitHub Releases; return 0 on success.
# ---------------------------------------------------------------------------
download_release_asset() {
    local filename="$1"
    local dest="${MODELS_DIR}/${filename}"
    local url="${RELEASE_URL}/${filename}"

    echo "  Trying ${url} …"
    if wget -q --show-progress -O "${dest}" "${url}" 2>/dev/null; then
        echo "  [ok]   ${filename} ($(du -sh "${dest}" | cut -f1))"
        return 0
    else
        rm -f "${dest}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# face_emotion.tflite — download from release, or build locally as fallback
# ---------------------------------------------------------------------------
get_face_model() {
    local dest="${MODELS_DIR}/face_emotion.tflite"

    if [ -f "${dest}" ]; then
        echo "  [skip] face_emotion.tflite already exists"
        return 0
    fi

    # 1) Try the pre-built release asset first (no TF required on Pi)
    if download_release_asset "face_emotion.tflite"; then
        return 0
    fi

    # 2) Fall back to the upstream tripletee release (same model)
    local upstream_url="https://github.com/YashRelekar/tripletee/releases/download/models-latest/face_emotion.tflite"
    echo "  Trying upstream release at ${upstream_url} …"
    if wget -q --show-progress -O "${dest}" "${upstream_url}" 2>/dev/null; then
        echo "  [ok]   face_emotion.tflite from upstream ($(du -sh "${dest}" | cut -f1))"
        return 0
    else
        rm -f "${dest}"
    fi

    # 3) Fall back to local build (needs TensorFlow — not suitable for Pi)
    echo "  Release asset not available; attempting local build …"
    echo "  NOTE: This requires TensorFlow: pip install tensorflow"
    if command -v python3 &>/dev/null; then
        if python3 "${REPO_DIR}/scripts/build_face_model.py"; then
            echo "  [ok]   face_emotion.tflite (built locally)"
            return 0
        fi
    fi

    echo ""
    echo "  [WARN] Could not obtain face_emotion.tflite."
    echo "         Options:"
    echo "           a) Build on a machine with TensorFlow installed:"
    echo "              pip install tensorflow"
    echo "              python3 scripts/build_face_model.py"
    echo "              Then copy models/face_emotion.tflite to your Pi."
    echo "           b) Trigger the CI build on GitHub:"
    echo "              Actions → 'Build & publish TFLite models' → Run workflow"
}

# ---------------------------------------------------------------------------
echo "=== Raspi5cam model setup ==="
get_face_model
echo "=== Done ==="
