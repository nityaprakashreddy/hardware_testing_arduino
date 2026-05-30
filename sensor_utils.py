import csv
import json
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

from hardware_config import DEFAULT_STRIDE, FEATURE_NAMES, FLEX_SENSOR_MAPPING


def parse_sensor_line(line):
    parts = line.strip().split(",")
    if len(parts) != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} values, got {len(parts)}")
    values = np.asarray([float(part) for part in parts], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("line contains non-finite values")
    return values


def open_serial(port, baud_rate, timeout=1.0):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required. Install it with: pip install pyserial") from exc
    return serial.Serial(port, baud_rate, timeout=timeout)


def read_sensor_rows(port, baud_rate, duration=None, max_rows=None, warmup_seconds=2.0):
    rows = []
    deadline = None if duration is None else time.monotonic() + duration

    with open_serial(port, baud_rate) as ser:
        time.sleep(warmup_seconds)
        ser.reset_input_buffer()
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if max_rows is not None and len(rows) >= max_rows:
                break

            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue
            try:
                rows.append(parse_sensor_line(raw))
            except ValueError:
                continue

    return np.asarray(rows, dtype=np.float32)


def moving_average(data, window=3):
    if window <= 1 or len(data) == 0:
        return np.asarray(data, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    columns = [np.convolve(data[:, idx], kernel, mode="same") for idx in range(data.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def resample_sequence(sequence, target_steps, min_length=2):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"sequence must have shape (time_steps, {len(FEATURE_NAMES)})")
    if len(sequence) == target_steps:
        return sequence
    if len(sequence) < min_length:
        raise ValueError(f"sequence needs at least {min_length} rows for resampling")

    old_x = np.linspace(0.0, 1.0, num=len(sequence), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float32)
    resampled = [np.interp(new_x, old_x, sequence[:, idx]) for idx in range(sequence.shape[1])]
    return np.stack(resampled, axis=1).astype(np.float32)


def add_engineered_features(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    flex_mean = np.mean(sequence[:, :5], axis=1, keepdims=True)
    flex_spread = np.max(sequence[:, :5], axis=1, keepdims=True) - np.min(sequence[:, :5], axis=1, keepdims=True)
    orientation_mag = np.linalg.norm(sequence[:, 5:7], axis=1, keepdims=True)
    deltas = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, flex_mean, flex_spread, orientation_mag, deltas], axis=1).astype(np.float32)


class SequencePreprocessor:
    def __init__(self, window_size=64, filter_window=3, include_magnitudes=True):
        self.window_size = window_size
        self.filter_window = filter_window
        self.include_magnitudes = include_magnitudes
        self.mean_ = None
        self.scale_ = None
        self.feature_names = list(FEATURE_NAMES)

    def _prepare_one(self, sequence):
        prepared = moving_average(sequence, self.filter_window)
        prepared = resample_sequence(prepared, self.window_size)
        if self.include_magnitudes:
            prepared = add_engineered_features(prepared)
        return prepared

    def fit_transform(self, sequences):
        prepared = np.asarray([self._prepare_one(seq) for seq in sequences], dtype=np.float32)
        flat = prepared.reshape(-1, prepared.shape[-1])
        self.mean_ = flat.mean(axis=0).astype(np.float32)
        self.scale_ = flat.std(axis=0).astype(np.float32)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self.transform_prepared(prepared)

    def transform(self, sequences):
        prepared = np.asarray([self._prepare_one(seq) for seq in sequences], dtype=np.float32)
        return self.transform_prepared(prepared)

    def transform_one(self, sequence):
        return self.transform([sequence])[0]

    def transform_prepared(self, prepared):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("preprocessor has not been fitted")
        return ((prepared - self.mean_) / self.scale_).astype(np.float32)


def load_dataset(dataset_dir):
    samples, _ = load_sensor_dataset(dataset_dir)
    return [sample["rows"] for sample in samples], np.asarray([sample["label"] for sample in samples])


def _read_sensor_csv(csv_path):
    rows = []
    malformed = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        line_number = 1
        if first is None:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), [{"line": 1, "reason": "empty file"}]
        has_header = [cell.strip().lower() for cell in first] == FEATURE_NAMES
        if not has_header:
            reader = iter([first] + list(reader))
            line_number = 0
        for row in reader:
            line_number += 1
            if len(row) != len(FEATURE_NAMES):
                malformed.append({"line": line_number, "reason": f"expected {len(FEATURE_NAMES)} columns, got {len(row)}"})
                continue
            try:
                values = [float(value) for value in row]
            except ValueError:
                malformed.append({"line": line_number, "reason": "non-numeric value"})
                continue
            if not np.all(np.isfinite(values)):
                malformed.append({"line": line_number, "reason": "non-finite value"})
                continue
            rows.append(values)
    return np.asarray(rows, dtype=np.float32), malformed


def _normalise_label(label):
    label = label.strip()
    aliases = {"thumbsup": "THUMBSUP", "thumbs_up": "THUMBSUP", "openhand": "OPEN", "point": "POINT"}
    return aliases.get(label.lower(), label.upper())


def _window_recording(rows, label, source, window_size, stride):
    if len(rows) < 2:
        return []
    if len(rows) <= window_size:
        return [{"rows": rows, "label": label, "source": str(source), "start": 0, "end": int(len(rows))}]
    samples = []
    for start in range(0, len(rows) - window_size + 1, stride):
        samples.append(
            {
                "rows": rows[start : start + window_size],
                "label": label,
                "source": str(source),
                "start": int(start),
                "end": int(start + window_size),
            }
        )
    if samples and samples[-1]["end"] < len(rows):
        samples.append(
            {
                "rows": rows[-window_size:],
                "label": label,
                "source": str(source),
                "start": int(len(rows) - window_size),
                "end": int(len(rows)),
            }
        )
    return samples


def load_sensor_dataset(dataset_dir, window_size=64, stride=DEFAULT_STRIDE):
    dataset_path = Path(dataset_dir)
    samples = []
    summary = {
        "dataset_dir": str(dataset_path),
        "feature_names": list(FEATURE_NAMES),
        "flex_sensor_mapping": dict(FLEX_SENSOR_MAPPING),
        "files": [],
        "malformed_rows": [],
        "removed_corrupt_rows": 0,
    }
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset directory not found: {dataset_path}")

    flat_csvs = sorted(path for path in dataset_path.glob("*.csv") if path.is_file())
    for csv_path in flat_csvs:
        label = _normalise_label(csv_path.stem)
        rows, malformed = _read_sensor_csv(csv_path)
        file_samples = _window_recording(rows, label, csv_path, window_size, stride)
        samples.extend(file_samples)
        summary["files"].append(
            {
                "path": str(csv_path),
                "label": label,
                "valid_rows": int(len(rows)),
                "malformed_rows": int(len(malformed)),
                "samples_created": int(len(file_samples)),
            }
        )
        for row in malformed:
            summary["malformed_rows"].append({"file": str(csv_path), **row})

    for label_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        label = _normalise_label(label_dir.name)
        for csv_path in sorted(label_dir.glob("*.csv")):
            rows, malformed = _read_sensor_csv(csv_path)
            if len(rows) >= 2:
                samples.append({"rows": rows, "label": label, "source": str(csv_path), "start": 0, "end": int(len(rows))})
            summary["files"].append(
                {
                    "path": str(csv_path),
                    "label": label,
                    "valid_rows": int(len(rows)),
                    "malformed_rows": int(len(malformed)),
                    "samples_created": int(1 if len(rows) >= 2 else 0),
                }
            )
            for row in malformed:
                summary["malformed_rows"].append({"file": str(csv_path), **row})

    summary["removed_corrupt_rows"] = int(len(summary["malformed_rows"]))

    if not samples:
        raise RuntimeError(f"no CSV samples found in {dataset_path}")
    return samples, summary


def write_dataset_summary(path, samples, load_summary, preprocessed_shape=None):
    labels = [sample["label"] for sample in samples]
    classes = sorted(set(labels))
    lengths = [int(len(sample["rows"])) for sample in samples]
    all_rows = np.vstack([sample["rows"] for sample in samples])
    class_distribution = {label: int(labels.count(label)) for label in classes}
    summary = {
        **load_summary,
        "total_classes": int(len(classes)),
        "classes": classes,
        "total_samples": int(len(samples)),
        "timesteps_per_sample": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "median": float(np.median(lengths)),
            "preprocessed": None if preprocessed_shape is None else int(preprocessed_shape[1]),
        },
        "min_values": {name: float(all_rows[:, idx].min()) for idx, name in enumerate(FEATURE_NAMES)},
        "max_values": {name: float(all_rows[:, idx].max()) for idx, name in enumerate(FEATURE_NAMES)},
        "class_distribution": class_distribution,
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def save_csv_sample(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(np.asarray(rows, dtype=np.float32))


def save_pickle(path, value):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


class PredictionSmoother:
    def __init__(self, size=7):
        self.predictions = deque(maxlen=size)

    def update(self, label, confidence):
        self.predictions.append((label, float(confidence)))
        votes = {}
        confidences = {}
        for pred_label, pred_confidence in self.predictions:
            votes[pred_label] = votes.get(pred_label, 0) + 1
            confidences[pred_label] = confidences.get(pred_label, 0.0) + pred_confidence
        best_label = max(votes, key=lambda item: (votes[item], confidences[item]))
        return best_label, confidences[best_label] / votes[best_label]
