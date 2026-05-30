FEATURE_NAMES = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "pitch",
    "roll",
]

FLEX_SENSOR_MAPPING = {
    "f1": "pinky",
    "f2": "thumb",
    "f3": "index",
    "f4": "middle",
    "f5": "ring",
}

FLEX_FEATURE_NAMES = FEATURE_NAMES[:5]
FLEX_FINGER_ORDER = [FLEX_SENSOR_MAPPING[name] for name in FLEX_FEATURE_NAMES]

GESTURE_CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "FIST",
    "OPEN",
    "POINT",
    "OK",
    "PEACE",
    "THUMBS_UP",
]

DEFAULT_BAUD_RATE = 115200
DEFAULT_PORT = "COM3"
DEFAULT_SAMPLE_RATE_HZ = 50
DEFAULT_WINDOW_SIZE = 64
DEFAULT_STRIDE = 16
DEFAULT_DATASET_DIR = "dataset2"
DEFAULT_KERAS_MODEL_PATH = "hardware_gesture_model.keras"
DEFAULT_CLASSICAL_MODEL_PATH = "hardware_gesture_model.pkl"
DEFAULT_MODEL_PATH = DEFAULT_KERAS_MODEL_PATH
DEFAULT_PREPROCESSOR_PATH = "hardware_preprocessor.pkl"
