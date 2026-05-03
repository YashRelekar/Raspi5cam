# models/

This directory holds the TFLite model files used for on-device inference.

## face_emotion.tflite

**Source:** [oarriaga/face_classification](https://github.com/oarriaga/face_classification) –
mini-XCEPTION model trained on FER-2013 (~65 % accuracy), MIT licence.

**Input:** `(1, 64, 64, 1)` float32, normalised to `[-1, 1]`  
**Output:** `(1, 7)` float32 softmax probabilities

**Label order (index → emotion):**

| Index | Emotion  |
|-------|----------|
| 0     | angry    |
| 1     | disgust  |
| 2     | fear     |
| 3     | happy    |
| 4     | sad      |
| 5     | surprise |
| 6     | neutral  |

### Obtaining the model

#### Option A – download the pre-built model (recommended for Raspberry Pi)

```bash
bash scripts/download_models.sh
```

The script downloads the pre-built `face_emotion.tflite` from this
repository's GitHub Releases. This requires no TensorFlow and is the
recommended method on the Raspberry Pi 5.

#### Option B – build locally (requires TensorFlow on a development machine)

```bash
# On a machine with TensorFlow installed (not required on the Pi):
pip install tensorflow
python3 scripts/build_face_model.py
# Then copy models/face_emotion.tflite to the Pi.
```

The build script downloads the original Keras HDF5 model from the
`oarriaga/face_classification` GitHub repository, converts it to
TFLite with dynamic-range quantisation, and saves it here.
