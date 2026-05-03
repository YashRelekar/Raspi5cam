"""Live camera preview for Raspberry Pi 5 (IMX219 / libcamera).

Run with::

    python3 -m raspi5cam

Press Ctrl+C in the terminal or 'q' in the preview window to exit.

Dependencies (install via apt, not pip)::

    sudo apt-get install -y python3-picamera2 python3-opencv
"""

import time


def main() -> int:
    try:
        from picamera2 import Picamera2  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Could not import Picamera2.")
        print("Fix: sudo apt-get install -y python3-picamera2")
        print(f"Details: {exc}")
        return 1

    try:
        import cv2  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Could not import cv2 (OpenCV).")
        print("Fix: sudo apt-get install -y python3-opencv")
        print(f"Details: {exc}")
        return 1

    picam2 = Picamera2()

    try:
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()

        print("Live preview started.  Press Ctrl+C to exit (or 'q' in the window).")

        while True:
            # Although RGB888 is requested, libcamera often configures the
            # stream as BGR888 in practice, so treat the array as BGR directly
            # (no cvtColor needed — matches the colour handling in camera.py).
            frame = picam2.capture_array()
            cv2.imshow("Raspi5cam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nExiting (Ctrl+C).")
    finally:
        try:
            picam2.stop()
        except Exception:  # noqa: BLE001
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
