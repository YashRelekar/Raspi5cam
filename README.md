# Raspi5cam

Live camera preview for Raspberry Pi using Picamera2 and OpenCV.

---

## Requirements

* Raspberry Pi (any model with a camera port)
* Raspberry Pi Camera Module (e.g. IMX219 / Camera Module v2)
* Debian / Raspberry Pi OS (Bookworm or later recommended)

---

## Camera setup

Enable the camera overlay in `/boot/firmware/config.txt`:

```ini
[all]
dtoverlay=imx219,cam0
```

Then reboot:

```bash
sudo reboot
```

Verify the camera is detected:

```bash
libcamera-hello --list-cameras
```

---

## Install dependencies (apt — recommended for Debian/Pi OS)

> **Use `apt`, not `pip`, for these packages on modern Raspberry Pi OS /
> Debian.** Python 3.12+ wheels for `picamera2` and `opencv-python` are
> often unavailable on PyPI/piwheels for `aarch64`, while the `apt`
> packages work out of the box.

```bash
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv

# Optional – command-line camera tools for quick testing
sudo apt-get install -y libcamera-apps
```

---

## Run

### Option A – install then run (recommended)

```bash
# From the repo root (picamera2 and opencv come from apt, not pip):
pip install -e .
python -m raspi5cam
```

### Option B – run without installing

```bash
PYTHONPATH=src python -m raspi5cam
```

Press **Ctrl+C** (or **`q`** in the preview window) to exit.

---

## Project structure

```
Raspi5cam/
├── src/
│   └── raspi5cam/
│       ├── __init__.py   # package marker
│       └── __main__.py   # entry point: python -m raspi5cam
├── old/                  # prior code kept for reference
└── README.md
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not import Picamera2` | `sudo apt-get install -y python3-picamera2` |
| `Could not import OpenCV` | `sudo apt-get install -y python3-opencv` |
| `Cannot open camera device` | Check `dtoverlay=imx219,cam0` in `/boot/firmware/config.txt` and reboot |
| No preview window appears | Ensure you have a display connected (HDMI) or use `DISPLAY=:0` |
