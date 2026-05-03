"""Facial emotion recognition using a TFLite model.

The model is expected to be a MobileNetV2 / mini-XCEPTION CNN trained on
FER-2013, exported as ``face_emotion.tflite``.

Download the pre-built model with::

    bash scripts/download_models.sh

Pipeline:
    1. Detect faces with OpenCV's Haar cascade.
    2. Crop and pre-process each face to 64 × 64 (or model's native size) greyscale.
    3. Run TFLite interpreter for emotion classification.
    4. Return the top-1 emotion label and confidence score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# Default input size for the oarriaga/face_classification mini-XCEPTION model.
# The actual size is read from the model metadata at load time; this is only
# used as a fallback.
MODEL_INPUT_SIZE = (64, 64)


class FaceEmotionRecognizer:
    """Detect faces and classify emotions from a BGR frame.

    Args:
        model_path: Path to the ``.tflite`` model file.
        labels: Ordered list of emotion label strings.
        confidence_threshold: Minimum confidence to report a result.
    """

    def __init__(
        self,
        model_path: str = "models/face_emotion.tflite",
        labels: Optional[list] = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        self._model_path = Path(model_path)
        self._labels = labels or [
            # Index order matches the oarriaga/face_classification FER-2013
            # mini-XCEPTION model (see scripts/build_face_model.py).
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]
        self._threshold = confidence_threshold
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._face_cascade = None
        # Actual model input size, resolved from metadata after load.
        self._model_input_size: tuple = MODEL_INPUT_SIZE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the TFLite model and the face detector."""
        self._load_face_detector()
        self._load_tflite_model()

    def _load_face_detector(self) -> None:
        """Load OpenCV's Haar cascade face detector."""
        try:
            import cv2  # type: ignore

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError("Haar cascade XML not found")
            log.info("Face detector loaded (Haar cascade)")
        except Exception as exc:  # noqa: BLE001
            log.warning("Face detector unavailable: %s", exc)
            self._face_cascade = None

    def _load_tflite_model(self) -> None:
        """Load the TFLite interpreter."""
        if not self._model_path.exists():
            log.warning(
                "Face emotion model not found at '%s'. "
                "Run 'bash scripts/download_models.sh' to download it.",
                self._model_path,
            )
            return

        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
        except ImportError:
            try:
                from tensorflow import lite as tflite  # type: ignore
            except ImportError:
                log.warning(
                    "Neither tflite_runtime nor TensorFlow is installed; "
                    "face emotion recognition disabled"
                )
                return

        try:
            self._interpreter = tflite.Interpreter(model_path=str(self._model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            # Read the actual spatial dimensions from the model's input tensor
            # shape (1, H, W, C) so we resize correctly regardless of which
            # model file is deployed.  Store as (width, height) for cv2.resize.
            shape = self._input_details[0].get("shape")
            if shape is not None and len(shape) == 4:
                self._model_input_size = (int(shape[2]), int(shape[1]))
            log.info(
                "Face emotion TFLite model loaded from '%s' (input %dx%d)",
                self._model_path,
                self._model_input_size[0],
                self._model_input_size[1],
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load face emotion model: %s", exc)
            self._interpreter = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> list:
        """Detect all faces and classify emotions.

        Args:
            frame: BGR uint8 NumPy array from the camera.

        Returns:
            List of dicts with keys ``"bbox"``, ``"emotion"``, and
            ``"confidence"`` — one entry per detected face.  Returns an
            empty list when no faces are found or the model is not loaded.
        """
        if self._face_cascade is None or self._interpreter is None:
            return []

        import cv2  # type: ignore

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        results = []
        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, self._model_input_size)
            # Normalise to [-1, 1] to match the mini-XCEPTION training
            # preprocessing from oarriaga/face_classification.
            face_input = (face_resized.astype(np.float32) / 255.0 - 0.5) * 2.0
            face_input = face_input[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)

            self._interpreter.set_tensor(
                self._input_details[0]["index"], face_input
            )
            self._interpreter.invoke()
            probs = self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )[0]

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])

            if confidence < self._threshold:
                continue

            results.append(
                {
                    "bbox": (x, y, w, h),
                    "emotion": self._labels[top_idx],
                    "confidence": confidence,
                }
            )
            log.debug(
                "Face emotion: %s (%.2f) at bbox=%s",
                self._labels[top_idx],
                confidence,
                (x, y, w, h),
            )

        return results

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FaceEmotionRecognizer":
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self._interpreter = None
