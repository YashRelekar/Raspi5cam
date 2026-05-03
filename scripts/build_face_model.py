#!/usr/bin/env python3
"""scripts/build_face_model.py

Download the pre-trained FER-2013 mini-XCEPTION Keras model from the
oarriaga/face_classification GitHub repository and convert it to TFLite
format, saving the result as models/face_emotion.tflite.

The source model achieves ~65 % accuracy on FER-2013 and is publicly
available under the MIT licence at:
  https://github.com/oarriaga/face_classification

Label order (index → emotion):
  0: angry, 1: disgust, 2: fear, 3: happy,
  4: sad, 5: surprise, 6: neutral

Model input:  (1, 64, 64, 1) float32, normalised to [-1, 1]
Model output: (1, 7) float32 softmax probabilities

NOTE — Raspberry Pi users:
  You do NOT need to run this script on the Pi.
  The pre-built model is downloaded automatically by:

    bash scripts/download_models.sh

  Run this script only on a machine where TensorFlow is installed
  (the GitHub Actions CI workflow does this automatically):

    pip install tensorflow
    python3 scripts/build_face_model.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SHA-pinned raw-content URL for reproducibility.
# This is the mini-XCEPTION model (~847 KB) trained on FER-2013.
# Source: https://github.com/oarriaga/face_classification  (MIT licence)
HDF5_URL = (
    "https://raw.githubusercontent.com/oarriaga/face_classification/"
    "b861d21b0e76ca5514cdeb5b56a689b7318584f4/"
    "trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5"
)

REPO_ROOT = Path(__file__).parent.parent
OUT_PATH = REPO_ROOT / "models" / "face_emotion.tflite"

# ---------------------------------------------------------------------------


def _download_hdf5() -> str:
    """Download the HDF5 model to a temporary file; return its path."""
    print(f"  Source: {HDF5_URL}")
    tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
    tmp.close()
    urllib.request.urlretrieve(HDF5_URL, tmp.name)
    size_kb = Path(tmp.name).stat().st_size // 1024
    print(f"  Downloaded {size_kb} KB → {tmp.name}")
    return tmp.name


def _convert_to_tflite(hdf5_path: str) -> bytes:
    """Load a Keras HDF5 model and return TFLite flatbuffer bytes."""
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        sys.exit(
            "\nERROR: TensorFlow is required to build the model.\n"
            "\n"
            "  On a Raspberry Pi, you do NOT need to run this script.\n"
            "  Just run:  bash scripts/download_models.sh\n"
            "  which downloads the pre-built model from GitHub Releases.\n"
            "\n"
            "  On a development machine, install TensorFlow and retry:\n"
            "    pip install tensorflow\n"
            "    python3 scripts/build_face_model.py\n"
        )

    print(f"  TensorFlow {tf.__version__}")
    print("  Loading Keras model …")
    model = tf.keras.models.load_model(hdf5_path, compile=False)
    print(f"  Input  shape : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")

    print("  Converting to TFLite (dynamic-range quantisation) …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        print(f"[skip] {OUT_PATH} already exists — delete it to rebuild.")
        return

    print("=== Building models/face_emotion.tflite ===")
    hdf5_path = _download_hdf5()
    try:
        tflite_bytes = _convert_to_tflite(hdf5_path)
    finally:
        os.unlink(hdf5_path)

    OUT_PATH.write_bytes(tflite_bytes)
    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"[ok] Saved {size_kb} KB → {OUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
