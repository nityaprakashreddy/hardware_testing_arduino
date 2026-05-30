import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np

from hardware_config import (
    DEFAULT_BAUD_RATE,
    DEFAULT_MODEL_PATH,
    DEFAULT_PORT,
    DEFAULT_PREPROCESSOR_PATH,
    DEFAULT_WINDOW_SIZE,
)
from sensor_utils import PredictionSmoother, load_pickle, open_serial, parse_sensor_line
from speech_utils import create_tts_engine


def load_prediction_model(model_path, classes_path):
    if Path(model_path).suffix == ".pkl":
        payload = load_pickle(model_path)
        return "classical", payload["model"], payload["classes"]

    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True)["classes"]
    return "keras", model, classes


def predict_probabilities(model_type, model, X):
    if model_type == "classical":
        return model.predict_proba(X.reshape(1, -1))[0]
    return model.predict(X[None, ...], verbose=0)[0]


class SentenceAccumulator:
    def __init__(
        self,
        space_gestures=None,
        backspace_gestures=None,
        speak_gestures=None,
        clear_gestures=None,
        repeat_gap=0.8,
    ):
        self.tokens = []
        self.last_accepted_label = None
        self.last_accepted_at = 0.0
        self.last_activity_at = 0.0
        self.last_spoken_text = ""
        self.space_gestures = set(space_gestures or [])
        self.backspace_gestures = set(backspace_gestures or [])
        self.speak_gestures = set(speak_gestures or [])
        self.clear_gestures = set(clear_gestures or [])
        self.repeat_gap = repeat_gap

    def reset_repeat_lock(self):
        self.last_accepted_label = None

    def text(self):
        return "".join(self.tokens).strip()

    def _append_space(self):
        if self.tokens and self.tokens[-1] != " ":
            self.tokens.append(" ")

    def accept(self, label, now):
        label = str(label).upper()
        if label == self.last_accepted_label and now - self.last_accepted_at < self.repeat_gap:
            return None, self.text(), False
        if label == self.last_accepted_label:
            return None, self.text(), False

        self.last_accepted_label = label
        self.last_accepted_at = now
        self.last_activity_at = now

        if label in self.clear_gestures:
            self.tokens = []
            self.last_spoken_text = ""
            return "clear", self.text(), False
        if label in self.backspace_gestures:
            if self.tokens:
                self.tokens.pop()
            return "backspace", self.text(), False
        if label in self.space_gestures:
            self._append_space()
            return "space", self.text(), False
        if label in self.speak_gestures:
            return "speak", self.text(), True
        if len(label) == 1 and "A" <= label <= "Z":
            self.tokens.append(label)
            return "letter", self.text(), False

        return "ignored", self.text(), False

    def should_auto_speak(self, now, pause_seconds):
        text = self.text()
        if not text or pause_seconds <= 0:
            return False
        if text == self.last_spoken_text:
            return False
        return self.last_activity_at > 0 and now - self.last_activity_at >= pause_seconds

    def mark_spoken(self):
        self.last_spoken_text = self.text()


def parse_gesture_set(value):
    if not value:
        return set()
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def main():
    parser = argparse.ArgumentParser(description="Run real-time ASL gesture inference from serial data.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preprocessor", default=DEFAULT_PREPROCESSOR_PATH)
    parser.add_argument("--classes", default="processed_dataset.npz")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--vote-window", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--speak", action="store_true", help="Speak stable gesture predictions using text-to-speech.")
    parser.add_argument("--speak-cooldown", type=float, default=1.5, help="Minimum seconds between spoken predictions.")
    parser.add_argument("--write-back", action="store_true", help="Send stable predictions back to the Arduino over serial.")
    parser.add_argument("--write-prefix", default="PRED:", help="Prefix used when writing predictions back to Arduino.")
    parser.add_argument("--sentence-mode", action="store_true", help="Build a continuous sentence from stable live predictions.")
    parser.add_argument("--space-gestures", default="OPEN,SPACE", help="Comma-separated gestures that insert a word space.")
    parser.add_argument("--backspace-gestures", default="POINT,DELETE,DEL", help="Comma-separated gestures that delete one character.")
    parser.add_argument("--speak-gestures", default="FIST", help="Comma-separated gestures that speak the current sentence.")
    parser.add_argument("--clear-gestures", default="THUMBS_UP,CLEAR", help="Comma-separated gestures that clear the sentence buffer.")
    parser.add_argument("--repeat-gap", type=float, default=0.8, help="Minimum seconds before considering repeat cleanup logic.")
    parser.add_argument("--auto-speak-pause", type=float, default=3.0, help="Speak the sentence after this many quiet seconds. Use 0 to disable.")
    args = parser.parse_args()

    model_type, model, classes = load_prediction_model(args.model, args.classes)
    preprocessor = load_pickle(args.preprocessor)
    buffer = deque(maxlen=args.window_size)
    smoother = PredictionSmoother(size=args.vote_window)
    tts_engine = None
    if args.speak:
        tts_engine, error = create_tts_engine()
        if error:
            print(f"TTS warning: {error}")
            tts_engine = None
    last_output_label = None
    last_spoken_at = 0.0
    sentence = None
    if args.sentence_mode:
        sentence = SentenceAccumulator(
            space_gestures=parse_gesture_set(args.space_gestures),
            backspace_gestures=parse_gesture_set(args.backspace_gestures),
            speak_gestures=parse_gesture_set(args.speak_gestures),
            clear_gestures=parse_gesture_set(args.clear_gestures),
            repeat_gap=args.repeat_gap,
        )
        print("Sentence mode enabled.")
        print("Space gestures:", ", ".join(sorted(sentence.space_gestures)) or "none")
        print("Backspace gestures:", ", ".join(sorted(sentence.backspace_gestures)) or "none")
        print("Speak gestures:", ", ".join(sorted(sentence.speak_gestures)) or "none")
        print("Clear gestures:", ", ".join(sorted(sentence.clear_gestures)) or "none")

    with open_serial(args.port, args.baud) as ser:
        ser.reset_input_buffer()
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                buffer.append(parse_sensor_line(line))
            except ValueError:
                continue
            if len(buffer) < args.window_size:
                continue

            window = np.asarray(buffer, dtype=np.float32)
            X = preprocessor.transform_one(window)
            probabilities = predict_probabilities(model_type, model, X)
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            label = str(classes[index])
            smooth_label, smooth_confidence = smoother.update(label, confidence)
            if smooth_confidence >= args.threshold:
                print(f"Predicted Gesture: {smooth_label} ({smooth_confidence:.2f})")
                if args.write_back and smooth_label != last_output_label:
                    ser.write(f"{args.write_prefix}{smooth_label}\n".encode("utf-8"))
                now = time.monotonic()
                if sentence:
                    event, current_text, should_speak = sentence.accept(smooth_label, now)
                    if event:
                        print(f"Sentence: {current_text or '[empty]'}")
                    if should_speak and current_text:
                        if args.write_back:
                            ser.write(f"SENT:{current_text}\n".encode("utf-8"))
                        if tts_engine:
                            try:
                                tts_engine.say(current_text)
                                tts_engine.runAndWait()
                                sentence.mark_spoken()
                                last_spoken_at = now
                            except Exception as exc:
                                print(f"TTS warning: {exc}")
                elif tts_engine and smooth_label != last_output_label:
                    if now - last_spoken_at >= args.speak_cooldown:
                        try:
                            tts_engine.say(smooth_label.replace("_", " "))
                            tts_engine.runAndWait()
                            last_spoken_at = now
                        except Exception as exc:
                            print(f"TTS warning: {exc}")
                last_output_label = smooth_label
            else:
                last_output_label = None
                if sentence:
                    sentence.reset_repeat_lock()

            if sentence and sentence.should_auto_speak(time.monotonic(), args.auto_speak_pause):
                current_text = sentence.text()
                if args.write_back:
                    ser.write(f"SENT:{current_text}\n".encode("utf-8"))
                if tts_engine:
                    try:
                        tts_engine.say(current_text)
                        tts_engine.runAndWait()
                        sentence.mark_spoken()
                        last_spoken_at = time.monotonic()
                    except Exception as exc:
                        print(f"TTS warning: {exc}")
                else:
                    sentence.mark_spoken()


if __name__ == "__main__":
    main()
