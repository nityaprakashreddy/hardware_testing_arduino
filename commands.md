# GestureTalk — Command Reference

Cross-platform CLI examples for this repository. See [`README.md`](README.md) for setup, hardware notes, and architecture.

**Serial port:** use `COM3` on Windows, `/dev/ttyUSB0` on Linux, or `/dev/cu.usbmodem*` on macOS.

**Python:** run from the repo root with your virtual environment activated:

```bash
cd hardware_testing_arduino
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-hardware.txt
```

---

## 1. Single CSV prediction

```bash
python predict_csv.py dataset2/A.csv --preprocessor hardware_preprocessor.pkl --top-k 5
```

---

## 2. Single CSV prediction + speak

```bash
python predict_csv.py dataset2/A.csv --preprocessor hardware_preprocessor.pkl --top-k 5 --speak
```

---

## 3. Live base model prediction

```bash
python inference.py --port COM3 --baud 115200
```

---

## 4. Live prediction + speak

```bash
python inference.py --port COM3 --baud 115200 --speak
```

---

## 5. Live prediction + Arduino write-back

```bash
python inference.py --port COM3 --baud 115200 --write-back
```

---

## 6. Live prediction + speak + write-back

```bash
python inference.py --port COM3 --baud 115200 --speak --write-back
```

---

## 7. Continuous sentence mode

```bash
python inference.py --port COM3 --baud 115200 --sentence-mode
```

---

## 8. Sentence mode + speak

```bash
python inference.py --port COM3 --baud 115200 --sentence-mode --speak
```

---

## 9. Sentence mode + speak + Arduino write-back

```bash
python inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```

---

## 10. Random Forest baseline (offline)

```bash
python predict_csv.py dataset2/A.csv --model hardware_gesture_model.pkl --preprocessor hardware_preprocessor.pkl --top-k 5
```

---

## 11. Random Forest baseline (live)

```bash
python inference.py --port COM3 --baud 115200 --model hardware_gesture_model.pkl
```

---

## 12. Record new labeled samples

Prompts for a gesture label, then records to `dataset2/<label>/`:

```bash
python data_collector.py --port COM3 --baud 115200 --duration 2.5
```

---

## 13. Serial stream sanity check

Runs until interrupted (`Ctrl+C`):

```bash
python serial_reader.py --port COM3 --baud 115200
```

---

## Notes

| Item | Detail |
|---|---|
| Default baud | `115200` |
| `hardware_gesture_model.keras` | CNN + BiLSTM (primary) |
| `hardware_gesture_model.pkl` | Random Forest baseline |
| `--speak` | Text-to-speech (`pyttsx3`; Windows PowerShell fallback) |
| `--write-back` | Sends `PRED:` / `SENT:` lines to Arduino |
| `--sentence-mode` | Continuous sentence composition |

Sentence-mode gesture controls and tuning: [`docs/added_features.md`](docs/added_features.md).

---

## Windows (PowerShell) equivalents

If you prefer explicit paths on Windows:

```powershell
cd hardware_testing_arduino
.\.venv\Scripts\python.exe predict_csv.py dataset2\A.csv --preprocessor hardware_preprocessor.pkl --top-k 5
.\.venv\Scripts\python.exe inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```
