import argparse
from pathlib import Path

import numpy as np

from hardware_config import DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH
from sensor_utils import _read_sensor_csv, load_pickle
from speech_utils import speak_text


def load_rows(csv_path):
    rows, malformed = _read_sensor_csv(csv_path)
    if malformed:
        print(f"Warning: skipped {len(malformed)} malformed rows from {csv_path}")
    if len(rows) < 2:
        raise ValueError(f"{csv_path} does not contain enough valid sensor rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Predict one gesture from a recorded Wokwi/sensor CSV file.")
    parser.add_argument("csv_path", help="CSV file with f1,f2,f3,f4,f5,pitch,roll columns.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preprocessor", default=DEFAULT_PREPROCESSOR_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--speak", action="store_true", help="Speak the best predicted gesture using text-to-speech.")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    preprocessor = load_pickle(args.preprocessor)
    X = preprocessor.transform_one(rows)
    if Path(args.model).suffix == ".pkl":
        payload = load_pickle(args.model)
        model = payload["model"]
        classes = payload["classes"]
        probabilities = model.predict_proba(X.reshape(1, -1))[0]
    else:
        from tensorflow.keras.models import load_model

        model = load_model(args.model)
        classes = np.load("processed_dataset.npz", allow_pickle=True)["classes"]
        probabilities = model.predict(X[None, ...], verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][: args.top_k]

    best_index = int(top_indices[0])
    print(f"File: {Path(args.csv_path)}")
    print(f"Predicted Gesture: {classes[best_index]} ({probabilities[best_index]:.2f})")
    print("Top predictions:")
    for index in top_indices:
        print(f"  {classes[int(index)]}: {probabilities[int(index)]:.2f}")

    if args.speak:
        try:
            speak_text(str(classes[best_index]).replace("_", " "))
        except RuntimeError as exc:
            print(f"TTS warning: {exc}")


if __name__ == "__main__":
    main()
