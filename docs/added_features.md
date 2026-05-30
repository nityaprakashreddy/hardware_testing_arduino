# GestureTalk Real-Time Sentence Prediction Mode

## Goal

Add a continuous real-time recording mode where:

```text
User performs gestures continuously
-> model predicts gestures in real time
-> repeated predictions are cleaned
-> gestures are combined into words/sentences
-> optional text-to-speech speaks the sentence
```

## Implemented Files

- `inference.py`
  - Added `--sentence-mode`
  - Added repeat-cleaning for held gestures
  - Added configurable space, backspace, speak, and clear gestures
  - Added automatic sentence speech after a quiet pause
  - Added `SENT:` serial write-back for full sentences

- `arduino/gesture_glove_mpu6050/gesture_glove_mpu6050.ino`
  - Receives `PRED:<label>` for the current predicted gesture
  - Receives `SENT:<sentence>` for the current completed sentence

## Command

Run live sentence mode:

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```

## Default Gesture Controls

| Gesture | Action |
|---|---|
| `A` to `Z` | Append letter |
| `OPEN` or `SPACE` | Insert space |
| `POINT`, `DELETE`, or `DEL` | Delete last character |
| `FIST` | Speak current sentence |
| `THUMBS_UP` or `CLEAR` | Clear sentence |

These can be changed from the command line:

```powershell
venv\Scripts\python.exe inference.py --sentence-mode --space-gestures OPEN --speak-gestures FIST --clear-gestures THUMBS_UP
```

## Repeat Cleaning

The same stable gesture is added only once while it is being held. To enter the same letter twice, relax or move through a low-confidence/neutral state, then perform the letter again.

Example:

```text
A A A A while held -> A
relax
A again -> AA
```

## Auto Speech

By default, if sentence mode is enabled and no new character is accepted for 3 seconds, the current sentence is spoken once.

Disable automatic sentence speech:

```powershell
venv\Scripts\python.exe inference.py --sentence-mode --speak --auto-speak-pause 0
```

Change the pause:

```powershell
venv\Scripts\python.exe inference.py --sentence-mode --speak --auto-speak-pause 5
```

## Arduino Write-Back Format

When `--write-back` is enabled, Python sends:

```text
PRED:A
SENT:HELLO WORLD
```

The Arduino sketch stores these as `lastPrediction` and `lastSentence`. They can be shown on an OLED/LCD, mapped to LEDs, or used with a buzzer/speaker module.
