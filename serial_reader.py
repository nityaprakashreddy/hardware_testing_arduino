import argparse

from hardware_config import DEFAULT_BAUD_RATE, DEFAULT_PORT, FEATURE_NAMES
from sensor_utils import open_serial, parse_sensor_line


def main():
    parser = argparse.ArgumentParser(description="Print live MPU6050 + flex sensor CSV rows.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE)
    args = parser.parse_args()

    print(",".join(FEATURE_NAMES))
    with open_serial(args.port, args.baud) as ser:
        ser.reset_input_buffer()
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                values = parse_sensor_line(line)
            except ValueError as exc:
                print(f"Skipping bad row: {line} ({exc})")
                continue
            print(",".join(f"{value:.6g}" for value in values))


if __name__ == "__main__":
    main()
