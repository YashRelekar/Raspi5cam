"""Live camera feed using Picamera2 and OpenCV.

Run with:
    python -m raspi5cam

Press Ctrl+C to exit.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from picamera2 import Picamera2  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Could not import Picamera2.")
        print("Install it on Raspberry Pi OS / Debian with:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y python3-picamera2")
        print(f"\nDetails: {exc}")
        return 1

    try:
        import cv2  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        print("ERROR: Could not import OpenCV (cv2).")
        print("Install it on Raspberry Pi OS / Debian with:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y python3-opencv")
        print(f"\nDetails: {exc}")
        return 1

    picam2 = Picamera2()
    try:
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()

        print("Camera preview started. Press Ctrl+C to exit.")

        while True:
            # libcamera commonly delivers BGR888 even when RGB888 is requested,
            # so we pass the frame to OpenCV (which expects BGR) without
            # channel-swapping to avoid colour distortion.
            frame = picam2.capture_array()
            cv2.imshow("Raspi5cam", frame)

            # waitKey keeps the window responsive; 'q' is a secondary exit key.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        try:
            picam2.stop()
        except Exception:  # noqa: BLE001
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
