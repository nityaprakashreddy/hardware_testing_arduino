import argparse
from datetime import datetime
from pathlib import Path

from hardware_config import (
    DEFAULT_BAUD_RATE,
    DEFAULT_DATASET_DIR,
    DEFAULT_PORT,
    DEFAULT_SAMPLE_RATE_HZ,
    GESTURE_CLASSES,
)
from sensor_utils import read_sensor_rows, save_csv_sample


def main():
    parser = argparse.ArgumentParser(description="Record labeled hardware gesture samples.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    print(f"Classes: {', '.join(GESTURE_CLASSES)}")
    label = input("Enter gesture: ").strip().upper()
    if not label:
        raise SystemExit("Gesture label is required.")

    expected_rows = int(args.duration * DEFAULT_SAMPLE_RATE_HZ)
    for sample_idx in range(args.samples):
        input(f"Press Enter to record {label} sample {sample_idx + 1}/{args.samples}...")
        rows = read_sensor_rows(args.port, args.baud, duration=args.duration)
        if len(rows) < expected_rows * 0.75:
            print(f"Warning: collected only {len(rows)} rows, expected around {expected_rows}.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path(args.dataset_dir) / label / f"{label}_{timestamp}.csv"
        save_csv_sample(output_path, rows)
        print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
