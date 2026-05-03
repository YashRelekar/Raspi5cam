#!/usr/bin/env python3
"""Raspi5cam – Real-time facial emotion detection.

Captures frames from the IMX219 camera (Raspberry Pi 5) and runs
face detection + emotion classification on each frame, displaying
annotated results in an OpenCV preview window.

Usage::

    python emotion_detection.py
    python emotion_detection.py --config config.yaml
    python emotion_detection.py --no-preview  # headless / SSH mode

The IMX219 camera must be enabled in /boot/firmware/config.txt::

    [all]
    dtoverlay=imx219,cam0

Dependencies (see requirements.txt and scripts/install.sh)::

    picamera2, opencv-python, numpy, tflite-runtime (or tensorflow)

Download the face emotion TFLite model before running::

    bash scripts/download_models.sh
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Optional

import yaml

from src.emotion.face_emotion import FaceEmotionRecognizer
from src.hardware.camera import Camera
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)

# Colour map for emotion bounding-box overlays (BGR)
EMOTION_COLOURS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (130, 0,   130),
    "happy":    (0,   200, 200),
    "sad":      (200, 100, 0),
    "surprise": (0,   165, 255),
    "neutral":  (180, 180, 180),
}
DEFAULT_COLOUR = (255, 255, 255)


def load_config(path: str) -> dict:
    """Load YAML configuration from *path*."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def draw_annotations(frame, results: list, show_confidence: bool = True):
    """Draw bounding boxes and emotion labels onto *frame* in-place.

    Args:
        frame: BGR uint8 NumPy array.
        results: List of dicts from ``FaceEmotionRecognizer.predict()``.
        show_confidence: Whether to append the confidence percentage.
    """
    import cv2  # type: ignore

    for r in results:
        x, y, w, h = r["bbox"]
        emotion = r["emotion"]
        confidence = r["confidence"]
        colour = EMOTION_COLOURS.get(emotion.lower(), DEFAULT_COLOUR)

        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

        # Build label text
        label = emotion.capitalize()
        if show_confidence:
            label += f"  {confidence * 100:.0f}%"

        # Draw filled rectangle behind text for readability
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame,
            (x, y - text_h - baseline - 6),
            (x + text_w + 4, y),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )


class EmotionDetector:
    """Captures frames from the camera and runs face emotion detection.

    Args:
        config: Parsed configuration dictionary (see ``config.yaml``).
        preview: Whether to display a live OpenCV preview window.
    """

    def __init__(self, config: dict, preview: bool = True) -> None:
        self.config = config
        self._preview = preview
        self._running = False

        hw = config.get("hardware", {})
        mdl = config.get("models", {})

        cam_cfg = hw.get("camera", {})
        self.camera = Camera(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 15),
        )

        self.recognizer = FaceEmotionRecognizer(
            model_path=mdl.get("face_emotion_model", "models/face_emotion.tflite"),
            labels=mdl.get("emotion_labels"),
            confidence_threshold=mdl.get("confidence_threshold", 0.25),
        )

        self._fps = cam_cfg.get("fps", 15)
        self._show_confidence = config.get("display", {}).get("show_confidence", True)
        self._window_name = "Raspi5cam – Emotion Detection  (press q to quit)"

    def start(self) -> None:
        """Open hardware and load models."""
        log.info("Opening camera …")
        self.camera.open()
        log.info("Loading face emotion model …")
        self.recognizer.load()
        self._running = True

    def stop(self) -> None:
        """Release resources."""
        self._running = False
        self.camera.close()
        if self._preview:
            try:
                import cv2  # type: ignore
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        log.info("Emotion detector stopped")

    def run(self) -> None:
        """Main capture-and-detect loop.  Blocks until stopped."""
        import cv2  # type: ignore

        frame_interval = 1.0 / max(self._fps, 1)
        last_results: list = []

        log.info("Starting capture loop (press Ctrl-C or 'q' to quit) …")

        while self._running:
            t0 = time.monotonic()

            frame = self.camera.capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            results = self.recognizer.predict(frame)
            last_results = results

            if results:
                emotions = [r["emotion"] for r in results]
                log.info("Detected: %s", ", ".join(emotions))
            else:
                log.debug("No face detected")

            if self._preview:
                display = frame.copy()
                draw_annotations(display, last_results, self._show_confidence)
                cv2.imshow(self._window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("'q' pressed – exiting")
                    self._running = False
                    break

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time facial emotion detection for Raspberry Pi 5 + IMX219"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window (useful for headless / SSH sessions)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    configure_logging(
        level=args.log_level or log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file") or None,
    )

    preview = not args.no_preview and cfg.get("hardware", {}).get("camera", {}).get(
        "preview_window", True
    )

    detector = EmotionDetector(cfg, preview=preview)

    def _handle_signal(signum: int, _frame: object) -> None:
        log.info("Signal %d received – stopping …", signum)
        detector.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    detector.start()
    try:
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
