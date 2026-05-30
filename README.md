# GestureTalk — Wearable ASL Gesture Recognition (Hardware Pipeline)

End-to-end **embedded sensing + machine learning** stack for American Sign Language (ASL) recognition using a data-glove prototype. An Arduino streams flex-sensor bend and MPU6050 orientation data over serial; Python performs windowed inference with a **CNN + BiLSTM** model (Random Forest baseline included), optional **text-to-speech**, and bidirectional **serial write-back** for display or haptics.

| Metric (held-out real windows) | CNN + BiLSTM | Random Forest |
|---|---:|---:|
| Accuracy | 86.76% | 79.41% |
| Macro F1 | 0.8897 | 0.8161 |

Full methodology, hardware trade-offs, and results: [`docs/ieee_research_paper.md`](docs/ieee_research_paper.md). Feature notes: [`docs/added_features.md`](docs/added_features.md).

---

## Highlights for reviewers

- **Firmware:** I2C register-level MPU6050 reads, startup gyro/accel calibration, exponential smoothing on five flex channels, deterministic CSV serial protocol at 50 Hz.
- **ML pipeline:** Sliding windows (64 steps), engineered features (17-D), saved `StandardScaler`, Keras temporal model + sklearn baseline.
- **Real data:** 29 gesture classes, offline replay via `dataset2/` CSV recordings.
- **Product features:** Live inference, sentence composition, TTS, `PRED:` / `SENT:` Arduino feedback.

---

## Hardware stack

| Component | Role |
|---|---|
| Arduino Nano (or Uno + external ADC) | Sample sensors, stream CSV rows |
| 5× flex sensors | Finger bend (`f1`–`f5`: pinky, thumb, index, middle, ring) |
| MPU6050 (I2C) | Hand pitch and roll |
| Host PC | Training artifacts, live inference, optional speech |

**Pin note:** On Uno, I2C uses A4/A5, so a fifth analog flex channel typically needs **A6 on Nano**, an **ADS1115**, or a multiplexer—see the research paper for wiring options. This repository’s sketch targets **direct `analogRead`** on Nano-style pins (`A0`–`A3`, `A6`).

---

## Hardware Code Highlight

Low-level sensor path from [`arduino/gesture_glove_mpu6050/gesture_glove_mpu6050.ino`](arduino/gesture_glove_mpu6050/gesture_glove_mpu6050.ino): flex readings are low-pass filtered before transmission; MPU raw counts are offset-corrected and converted to orientation angles.

```cpp
// Exponential smoothing — stabilizes noisy flex ADC readings before ML ingest
for (uint8_t i = 0; i < 5; i++) {
  float reading = analogRead(FLEX_PINS[i]);
  flexFiltered[i] = FILTER_ALPHA * reading + (1.0f - FILTER_ALPHA) * flexFiltered[i];
  Serial.print(flexFiltered[i], 2);
  Serial.print(",");
}

// MPU6050: subtract calibrated offsets, scale to m/s², derive pitch/roll
float ax = ((float)accelRaw[0] - accelOffset[0]) / 16384.0f * 9.80665f;
float ay = ((float)accelRaw[1] - accelOffset[1]) / 16384.0f * 9.80665f;
float az = ((float)accelRaw[2] - accelOffset[2]) / 16384.0f * 9.80665f;
float pitch = atan2(ax, sqrt(ay * ay + az * az)) * 180.0f / PI;
float roll  = atan2(ay, sqrt(ax * ax + az * az)) * 180.0f / PI;
```

I2C burst read and startup calibration (250 samples, Z-axis gravity compensation):

```cpp
void readMpuRaw(int16_t *accel, int16_t *gyro) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14, (uint8_t)true);
  accel[0] = (Wire.read() << 8) | Wire.read();
  // ... axes 1–2, skip temperature, gyro 0–2
}
```

---

## System architecture

```text
[Glove: flex + MPU6050] --115200 serial CSV--> [Arduino firmware]
                                                      |
                                                      v
[Python: sensor_utils] --> window + normalize --> [CNN+BiLSTM | RF]
                                                      |
                        optional TTS, sentence mode, PRED:/SENT: write-back
```

**Serial row format** (one sample per line, no debug text during live inference):

```text
f1,f2,f3,f4,f5,pitch,roll
```

---

## Project structure

```text
hardware_testing_arduino/
├── arduino/gesture_glove_mpu6050/   # Firmware (flex + MPU6050)
├── dataset2/                        # Recorded gesture CSV samples
├── docs/                            # IEEE-style report, feature notes
├── inference.py                     # Live serial prediction + sentence/TTS
├── predict_csv.py                   # Offline CSV evaluation
├── data_collector.py                # Record labeled training CSVs
├── serial_reader.py                 # Serial stream sanity check
├── sensor_utils.py                  # Parsing, windows, preprocessing
├── hardware_config.py               # Ports, features, class labels
├── hardware_gesture_model.keras     # Primary temporal model
├── hardware_gesture_model.pkl         # Random Forest baseline (~18 MB)
├── hardware_preprocessor.pkl        # Fitted normalization
├── processed_dataset.npz            # Class label array for Keras
├── requirements-hardware.txt        # Minimal deps for this pipeline
├── requirements.txt                 # Full monorepo (camera, Jupyter, etc.)
└── commands.md                      # Full CLI cookbook
```

---

## Quick start

### Prerequisites

- Python 3.10+ (TensorFlow 2.17 per `requirements-hardware.txt`)
- Arduino IDE or `arduino-cli` — flash `arduino/gesture_glove_mpu6050/`
- USB serial drivers for your board

### Setup

```bash
cd hardware_testing_arduino
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-hardware.txt
```

### Offline test (no hardware)

```bash
python predict_csv.py dataset2/A.csv --preprocessor hardware_preprocessor.pkl --top-k 5
```

### Live glove inference

Replace the port with your board (`COM3` on Windows, `/dev/ttyUSB0` or `/dev/cu.usbmodem*` on Linux/macOS):

```bash
python inference.py --port COM3 --baud 115200
```

Sentence mode with speech and Arduino feedback:

```bash
python inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```

More commands: [`commands.md`](commands.md).

---

## Models and artifacts

| File | Description |
|---|---|
| `hardware_gesture_model.keras` | CNN + BiLSTM (primary) |
| `hardware_gesture_model.pkl` | Random Forest baseline |
| `hardware_preprocessor.pkl` | Training-time scaler (required for inference) |
| `processed_dataset.npz` | `classes` array aligned with Keras output |

---

## Arduino ↔ host protocol

- **Device → host:** CSV rows `f1,f2,f3,f4,f5,pitch,roll` only (no `Serial.println` debug during inference).
- **Host → device:** `PRED:<label>` for per-gesture feedback; `SENT:<sentence>` after sentence mode. Parsed in firmware for LCD/LED/haptic hooks.

---

## Tech stack

**Embedded:** C++ (Arduino), Wire/I2C, ADC  
**ML / backend:** Python, NumPy, TensorFlow/Keras, scikit-learn, pyserial  
**Optional:** gTTS / pyttsx3 for speech output  

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Nitya Prakash Reddy** — [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-USERNAME) · [Demo video](https://YOUR-DEMO-VIDEO-URL)
