"""Raspberry Pi 5 Camera interface using picamera2 (IMX219).

The IMX219 sensor (Camera Module v2) is enabled via::

    # /boot/firmware/config.txt  — under [all]
    dtoverlay=imx219,cam0

On non-Pi systems the module falls back to OpenCV ``VideoCapture`` so that
the rest of the pipeline can be developed and tested without real hardware.

Important note on colour order:
- OpenCV uses BGR.
- Although we request RGB888 from picamera2, libcamera may actually
  configure the stream as BGR888 (visible in libcamera logs).
- Therefore we treat picamera2's ``capture_array("main")`` output as BGR
  and do NOT blindly swap channels.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class Camera:
    """Thread-safe camera wrapper for Raspberry Pi 5 + IMX219.

    Tries ``picamera2`` first (Raspberry Pi with libcamera stack), then
    falls back to OpenCV ``VideoCapture`` for development machines.

    Args:
        device_index: Camera device index (0 = IMX219 on cam0 port).
        width: Capture width in pixels.
        height: Capture height in pixels.
        fps: Target frames per second.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cam = None
        self._lock = threading.Lock()
        self._use_picamera2 = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialise and start the camera."""
        try:
            from picamera2 import Picamera2  # type: ignore

            self._cam = Picamera2(self._device_index)
            config = self._cam.create_preview_configuration(
                main={
                    "size": (self._width, self._height),
                    # Request RGB888; libcamera may configure BGR888 in practice.
                    # Actual channel order is handled in capture_frame().
                    "format": "RGB888",
                }
            )
            self._cam.configure(config)
            self._cam.start()
            self._use_picamera2 = True
            log.info(
                "Camera opened via picamera2/IMX219 (device=%d, %dx%d @ %dfps)",
                self._device_index,
                self._width,
                self._height,
                self._fps,
            )
        except Exception:  # noqa: BLE001
            log.warning(
                "picamera2 not available; falling back to OpenCV VideoCapture. "
                "Ensure dtoverlay=imx219,cam0 is set in /boot/firmware/config.txt "
                "and picamera2 is installed."
            )
            import cv2  # type: ignore

            self._cam = cv2.VideoCapture(self._device_index)
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cam.set(cv2.CAP_PROP_FPS, self._fps)
            if not self._cam.isOpened():
                raise RuntimeError(
                    f"Cannot open camera device {self._device_index}. "
                    "Check that the IMX219 is connected and "
                    "dtoverlay=imx219,cam0 is in /boot/firmware/config.txt."
                )
            log.info(
                "Camera opened via OpenCV (device=%d, %dx%d @ %dfps)",
                self._device_index,
                self._width,
                self._height,
                self._fps,
            )

    def close(self) -> None:
        """Stop and release the camera."""
        with self._lock:
            if self._cam is None:
                return
            with contextlib.suppress(Exception):
                if self._use_picamera2:
                    self._cam.stop()
                    self._cam.close()
                else:
                    self._cam.release()
            self._cam = None
        log.info("Camera closed")

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single BGR frame.

        Returns:
            A ``(H, W, 3)`` uint8 NumPy array in BGR order, or *None* on
            failure.
        """
        if self._cam is None:
            log.error("Camera is not open")
            return None

        with self._lock:
            try:
                if self._use_picamera2:
                    # Treat picamera2 output as BGR.
                    # libcamera often configures the stream as BGR888 even when
                    # RGB888 is requested; treating it as BGR avoids incorrect
                    # colour swaps (e.g. skin tones appearing blue).
                    frame = self._cam.capture_array("main")
                else:
                    import cv2  # type: ignore

                    ret, frame = self._cam.read()
                    if not ret:
                        log.warning("Camera read failed")
                        return None
                return frame
            except Exception as exc:  # noqa: BLE001
                log.error("Camera capture error: %s", exc)
                return None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
