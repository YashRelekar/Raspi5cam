# Raspi5cam – Real-time Facial Emotion Detection

Real-time facial emotion recognition for **Raspberry Pi 5** with the
**IMX219** (Camera Module v2) sensor, adapted from the
[tripletee](https://github.com/YashRelekar/tripletee) project.

---

## Step 1 – Live camera preview (start here)

This is the minimal entry point: pull the repo on your Pi and run a live
camera preview in one command.

### Clone & install dependencies

```bash
git clone https://github.com/YashRelekar/Raspi5cam.git
cd Raspi5cam
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv
```

### (Optional) install as an editable package

```bash
pip install -e .
```

### Run the preview

```bash
python3 -m raspi5cam
```

Press **Ctrl+C** in the terminal or **`q`** in the preview window to stop.

---

## What it does

* Captures frames from the IMX219 camera via `picamera2`
* Detects faces using OpenCV's Haar cascade
* Classifies each face into one of 7 emotions using a lightweight
  TFLite mini-XCEPTION model (trained on FER-2013):
  **angry · disgust · fear · happy · sad · surprise · neutral**
* Overlays colour-coded bounding boxes and emotion labels on a live
  preview window

---

## Hardware requirements

| Component | Details |
|-----------|---------|
| Raspberry Pi 5 | Any RAM variant |
| IMX219 camera | Camera Module v2 connected to the `cam0` port |
| Raspberry Pi OS | Bookworm 64-bit (recommended) |

---

## Camera setup

The IMX219 sensor needs a device-tree overlay to be loaded on boot.
Open `/boot/firmware/config.txt` and add the following under the
`[all]` section:

```ini
[all]
dtoverlay=imx219,cam0
```

Then **reboot**:

```bash
sudo reboot
```

Verify the camera is detected after rebooting:

```bash
libcamera-hello --list-cameras
```

You should see an entry for `imx219`.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YashRelekar/Raspi5cam.git
cd Raspi5cam
```

### 2. Run the installer (as root)

```bash
sudo bash scripts/install.sh
```

The installer will:
* Install system packages (`python3-picamera2`, `python3-opencv`, etc.)
* Create a Python virtual environment at `.venv/`
* Install Python dependencies from `requirements.txt`
* Download the pre-built `face_emotion.tflite` model

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Download the model manually (if the installer didn't)

```bash
bash scripts/download_models.sh
```

---

## Usage

```bash
# Live preview window (default)
python emotion_detection.py

# Custom config file
python emotion_detection.py --config config.yaml

# Headless mode (no display, e.g. over SSH)
python emotion_detection.py --no-preview

# Verbose debug output
python emotion_detection.py --log-level DEBUG
```

Press **`q`** in the preview window or **`Ctrl-C`** in the terminal to stop.

---

## Configuration

All settings are in `config.yaml`:

```yaml
hardware:
  camera:
    device_index: 0      # IMX219 on cam0 → index 0
    preview_window: true # set to false for headless use
    width: 640
    height: 480
    fps: 15

models:
  face_emotion_model: "models/face_emotion.tflite"
  confidence_threshold: 0.25   # raise to reduce false positives

display:
  show_confidence: true        # show % next to emotion label
```

---

## Project structure

```
Raspi5cam/
├── pyproject.toml         # package metadata (pip install -e .)
├── emotion_detection.py   # full emotion-detection entry point
├── config.yaml            # configuration
├── requirements.txt
├── models/
│   └── face_emotion.tflite  # TFLite model (downloaded)
├── scripts/
│   ├── install.sh           # one-shot installer for Raspberry Pi
│   ├── download_models.sh   # fetch the TFLite model
│   └── build_face_model.py  # (optional) rebuild from source
└── src/
    ├── raspi5cam/
    │   ├── __init__.py      # package marker
    │   └── __main__.py      # step-1 live preview (python3 -m raspi5cam)
    ├── hardware/
    │   └── camera.py        # picamera2 / OpenCV camera wrapper
    ├── emotion/
    │   └── face_emotion.py  # face detection + emotion classification
    └── utils/
        └── logger.py        # structured logging helper
```

---

## Building the model from source

If you are on a machine with TensorFlow installed (e.g. a development
laptop, not the Pi):

```bash
pip install tensorflow
python3 scripts/build_face_model.py
# Copy models/face_emotion.tflite to the Pi afterwards.
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Cannot open camera device 0` | Check `dtoverlay=imx219,cam0` in `/boot/firmware/config.txt` and reboot |
| `picamera2 not available` | Install with `sudo apt install python3-picamera2` |
| `Face emotion model not found` | Run `bash scripts/download_models.sh` |
| Blue skin tones in preview | Expected behaviour — libcamera configures BGR output internally |
| No faces detected | Improve lighting; move closer to camera; lower `confidence_threshold` |

---

## Acknowledgements

* [oarriaga/face_classification](https://github.com/oarriaga/face_classification) –
  the FER-2013 mini-XCEPTION model (MIT licence)
* [tripletee](https://github.com/YashRelekar/tripletee) – original project
  this is adapted from