# GestureTalk: A Complete Camera and Wearable Sensor Based American Sign Language Recognition System

**An IEEE-Style Engineering Project Report**

**Project Title:** GestureTalk: Camera-Based and Arduino-Based ASL Gesture Recognition with Real-Time Text-to-Speech Feedback  
**Domain:** Assistive Technology, Human-Computer Interaction, Machine Learning, Embedded Systems  
**Input Modalities:** Webcam with MediaPipe, Arduino glove with flex sensors and MPU6050  
**Current Primary System:** Hardware glove using `f1,f2,f3,f4,f5,pitch,roll` time-series input  
**Model:** CNN + BiLSTM temporal deep learning model  
**Output:** Gesture label, sentence mode, text-to-speech, optional Arduino write-back  

\newpage

## Table of Contents

1. Abstract  
2. Keywords  
3. Acknowledgements  
4. List of Figures  
5. List of Tables  
6. Nomenclature  
7. Chapter 1: Introduction  
8. Chapter 2: Literature Review  
9. Chapter 3: Proposed System and System Design  
10. Chapter 4: Dataset, Preprocessing, and Feature Engineering  
11. Chapter 5: Model Architecture and Training Methodology  
12. Chapter 6: Implementation Details  
13. Chapter 7: Experimental Results and Analysis  
14. Chapter 8: Real-Time Inference, TTS, and Wireless Extension  
15. Chapter 9: Limitations and Risk Analysis  
16. Chapter 10: Conclusion and Future Work  
17. References  
18. Appendices  

\newpage

## Abstract

Sign language recognition is an important assistive technology problem because it can reduce communication barriers between sign language users and non-signing users. This report presents GestureTalk, a complete American Sign Language (ASL) gesture recognition project that evolved through two major stages. The first stage used a software-only camera system based on OpenCV, MediaPipe Hands, and a dense neural network trained on extracted hand landmark vectors. The second and current stage uses a wearable glove connected to an Arduino. The glove records five flex sensor values and two orientation angles derived from an MPU6050 inertial measurement unit. Each hardware row follows the format `f1,f2,f3,f4,f5,pitch,roll`, where the five flex values correspond to pinky, thumb, index, middle, and ring bending, while pitch and roll represent hand orientation.

The current hardware pipeline loads real recorded CSV files from `dataset2/`, validates malformed rows, dynamically detects labels, slices each recording into temporal windows, applies realistic augmentation only to the training split, and preserves a real unaugmented test split for evaluation. A sequence preprocessor converts variable recordings into normalized temporal tensors with shape `64 x 17`. The 17 features consist of seven raw sensor features, flex mean, flex spread, orientation magnitude, and seven temporal delta features. A CNN + BiLSTM model is trained for temporal classification and compared with a Random Forest baseline trained on flattened normalized windows.

The latest experiment detected 29 gesture classes, created 339 real source windows, used 271 real training windows, expanded the training set to 2168 samples through augmentation, and evaluated on 68 real unaugmented test windows. The CNN + BiLSTM model achieved 86.76 percent accuracy and 0.8897 macro F1, outperforming the Random Forest baseline accuracy of 79.41 percent and macro F1 of 0.8161. The strongest classes were recognized consistently, while the weakest classes were C, D, O, and P due to similar finger bend patterns and limited sensing of palm shape and thumb contact.

GestureTalk also includes real-time serial inference, CSV prediction, prediction smoothing, continuous sentence mode, optional text-to-speech, and optional serial write-back of predicted labels or sentences to the Arduino. The report documents the full engineering design, software architecture, hardware assumptions, preprocessing logic, model training workflow, results, limitations, and future deployment options, including Bluetooth and ESP32-based wireless communication.

\newpage

## Keywords

American Sign Language, ASL recognition, flex sensor, MPU6050, Arduino Uno, wearable glove, MediaPipe, hand landmarks, gesture recognition, CNN, BiLSTM, Random Forest, temporal sequence classification, assistive technology, text-to-speech, embedded systems, human-computer interaction.

## Acknowledgements

The authors express sincere gratitude to the project guide, faculty members, laboratory staff, and peers who supported the development and review of GestureTalk. The project required knowledge from machine learning, embedded systems, signal processing, human-computer interaction, and assistive technology. Support during dataset collection, debugging, model training, evaluation, and report preparation was valuable in transforming the initial camera-based prototype into a hardware-oriented gesture recognition system.

The authors also acknowledge the developers and maintainers of open-source tools used in this project, including Python, NumPy, scikit-learn, TensorFlow/Keras, OpenCV, MediaPipe, PySerial, Matplotlib, and Streamlit. These tools made it possible to build an end-to-end prototype within an academic project environment.

\newpage

## List of Figures

**Figure 1:** Overall GestureTalk system architecture.  
**Figure 2:** Camera-based MediaPipe recognition pipeline.  
**Figure 3:** Hardware glove data acquisition pipeline.  
**Figure 4:** Flex sensor placement on hand.  
**Figure 5:** MPU6050 pitch and roll orientation axes.  
**Figure 6:** Serial data flow from Arduino to Python.  
**Figure 7:** Dataset loading and validation workflow.  
**Figure 8:** Preprocessing pipeline.  
**Figure 9:** Augmentation pipeline.  
**Figure 10:** CNN + BiLSTM model architecture.  
**Figure 11:** Training accuracy plot.  
**Figure 12:** Training loss plot.  
**Figure 13:** Deep model confusion matrix.  
**Figure 14:** Random Forest baseline confusion matrix.  
**Figure 15:** Metrics comparison plot.  
**Figure 16:** Real-time inference and TTS output flow.  
**Figure 17:** Wireless Bluetooth serial extension.  

## List of Tables

**Table 1:** Comparison between camera-based and glove-based recognition.  
**Table 2:** Hardware sensor feature format.  
**Table 3:** Dataset class distribution.  
**Table 4:** Preprocessing feature expansion.  
**Table 5:** Model architecture summary.  
**Table 6:** Experimental dataset statistics.  
**Table 7:** Deep model and baseline comparison.  
**Table 8:** Weak-performing and confused classes.  
**Table 9:** Real-time command line modes.  
**Table 10:** Future work roadmap.  

\newpage

## Nomenclature

| Symbol or Term | Meaning |
|---|---|
| ASL | American Sign Language |
| HCI | Human-Computer Interaction |
| IMU | Inertial Measurement Unit |
| MPU6050 | Six-axis accelerometer and gyroscope module |
| CNN | Convolutional Neural Network |
| LSTM | Long Short-Term Memory network |
| BiLSTM | Bidirectional Long Short-Term Memory network |
| TTS | Text-to-Speech |
| `f1` | Thumb flex sensor value |
| `f2` | Index finger flex sensor value |
| `f3` | Middle finger flex sensor value |
| `f4` | Ring finger flex sensor value |
| `f5` | Pinky finger flex sensor value |
| `pitch` | Forward/backward hand orientation angle |
| `roll` | Side-to-side hand orientation angle |
| Window | Fixed-length temporal segment used for classification |
| Macro F1 | Mean F1 score computed equally across classes |

\newpage

# Chapter 1: Introduction

## 1.1 Background

Communication is a central part of daily life, education, health care, employment, and social participation. For deaf and hard-of-hearing individuals, sign language is a natural and expressive communication medium. However, communication barriers arise when sign language users interact with people who do not understand sign language. Automatic sign language recognition attempts to reduce this barrier by converting signs into text, speech, or machine-readable commands.

American Sign Language contains hand shapes, palm orientations, movements, locations, facial expressions, and body posture. A complete ASL translator is a complex research problem because ASL is a full natural language with grammar and semantics. A practical academic prototype often begins with fingerspelling or isolated gesture recognition. Fingerspelling is useful because many gestures correspond to alphabet letters, making the recognition task a discrete classification problem.

GestureTalk is designed as an academic project that explores this problem from both software and hardware perspectives. The early version used a camera and MediaPipe to detect hand landmarks. The current version uses a wearable glove with flex sensors and an MPU6050 connected to an Arduino. This progression is important because it demonstrates the strengths and weaknesses of two common recognition approaches.

## 1.2 Need for Assistive Gesture Recognition

A recognition system that converts signs into text or speech can support communication in classrooms, hospitals, public service counters, and personal devices. Such systems can also be used as learning tools for beginners practicing sign language. If the model can provide immediate feedback, users can test gestures and improve consistency. If the system can speak the recognized output, it becomes easier for non-signing listeners to understand the message.

Assistive gesture recognition is also relevant to human-computer interaction. A wearable glove can be used to control devices, type letters, trigger commands, or provide silent input in noisy environments. While the social and linguistic complexity of sign language must be treated respectfully, isolated recognition prototypes provide a foundation for more complete systems.

\newpage

## 1.3 Evolution of GestureTalk

The project began with a software-only camera approach. In that case, a webcam captured hand images, OpenCV processed the video stream, MediaPipe Hands extracted 21 hand landmarks, and a neural network classified the flattened 63-dimensional landmark vector. This method required only a laptop camera and was easy to demonstrate. It also provided a useful proof of concept: a compact feature representation can be more efficient than classifying raw image pixels.

The project later moved to a hardware glove approach. The hardware glove uses flex sensors to measure finger bending and an MPU6050 module to estimate hand pitch and roll. This change was made because camera-based systems can be affected by lighting, hand visibility, background clutter, skin tone variation, camera quality, occlusion, and viewing angle. A glove does not require the hand to be visible to a camera and directly measures physical finger movement.

The current project therefore covers both cases:

| Case | Input | Main Method | Output |
|---|---|---|---|
| Software case | Webcam | MediaPipe landmarks + dense neural network | Letter prediction and spelling |
| Current hardware case | Arduino glove | Flex + pitch/roll time series + CNN + BiLSTM | Gesture prediction, sentence mode, TTS |

## 1.4 Problem Statement

The problem addressed in this project is:

> To design and implement a real-time ASL-style gesture recognition system that can classify isolated hand gestures using either camera-based hand landmarks or wearable glove sensor data, with the current focus on an Arduino-based glove using temporal deep learning and text-to-speech output.

The hardware problem has several sub-problems. The system must acquire noisy sensor values, validate CSV rows, handle variable-length recordings, preserve temporal information, train a model on a small real dataset, compare model performance against a baseline, and provide usable real-time output.

\newpage

## 1.5 Objectives

The main objectives of GestureTalk are:

1. To study the earlier camera and MediaPipe-based recognition approach.
2. To design a hardware glove input format using five flex sensors and MPU6050-derived pitch and roll.
3. To collect and process real glove sensor recordings.
4. To dynamically detect gesture labels from dataset files.
5. To validate CSV data and remove malformed rows.
6. To convert variable recordings into fixed temporal windows.
7. To engineer meaningful temporal features from raw sensor data.
8. To apply realistic augmentation only to training data.
9. To train a CNN + BiLSTM model for time-series gesture recognition.
10. To compare the deep model with a Random Forest baseline.
11. To evaluate accuracy, precision, recall, F1 score, class-wise performance, and confusion patterns.
12. To support offline CSV prediction and live serial inference.
13. To add text-to-speech and real-time sentence prediction.
14. To discuss wireless communication and Arduino feedback.

## 1.6 Scope of the Project

The project focuses on isolated ASL-style gesture recognition and selected auxiliary gestures such as OPEN, POINT, and THUMBSUP. It does not claim to translate full ASL grammar. The current hardware model recognizes 29 classes based on real recorded glove data. The project demonstrates a working pipeline from dataset collection to model training, evaluation, reporting, live inference, and speech output.

The current system uses the laptop for machine learning inference. The Arduino collects and streams sensor data. This is a practical choice because an Arduino Uno cannot run TensorFlow or scikit-learn models due to limited RAM, flash memory, and processing capability.

\newpage

## 1.7 Report Organization

This report is organized into ten chapters. Chapter 1 introduces the problem, motivation, objectives, and scope. Chapter 2 reviews related work in camera-based recognition, wearable sensing, flex sensors, IMUs, and deep learning. Chapter 3 explains the proposed system design. Chapter 4 describes the dataset, preprocessing, and augmentation. Chapter 5 presents the model architecture and training methodology. Chapter 6 gives implementation details. Chapter 7 reports experimental results. Chapter 8 discusses real-time inference, TTS, sentence mode, and wireless extension. Chapter 9 presents limitations and risk analysis. Chapter 10 concludes the report and proposes future work.

\newpage

# Chapter 2: Literature Review

## 2.1 Introduction to Sign Language Recognition Research

Sign language recognition has been studied using camera systems, depth sensors, wearable gloves, electromyography, inertial sensors, and multimodal fusion. Vision-based systems attempt to infer hand shape and motion from images or video. Wearable systems measure the physical state of the hand using sensors attached to fingers or the wrist. Both approaches have value, and both contain limitations.

Vision-based systems are attractive because they require minimal user hardware. A webcam or phone camera can capture the hand, and computer vision models can extract useful features. However, vision systems are sensitive to the environment. They depend on lighting, background, camera position, and whether the hand remains visible. They can also struggle when signs require detailed finger contact or when the hand is partially occluded.

Wearable systems are more intrusive because the user must wear hardware, but they provide direct sensor data. Flex sensors can capture finger bending, and IMUs can capture hand orientation and movement. A glove can operate in poor lighting and does not require a clear camera view. This makes wearable sensing suitable for robust human-computer interaction prototypes.

## 2.2 Camera-Based Recognition

Camera-based ASL recognition commonly uses image processing or deep learning. Earlier systems extracted hand contours, skin color masks, or geometric descriptors. Modern systems often use convolutional neural networks or keypoint extraction frameworks. MediaPipe Hands is particularly useful because it converts an RGB image into 21 hand landmarks. Instead of training a large model on raw pixels, the system can train a smaller classifier on hand geometry.

GestureTalk's software case follows this idea. It uses MediaPipe to extract 21 landmarks. Each landmark has x, y, and z coordinates, giving 63 features per frame. A dense neural network then predicts the letter. This approach is computationally efficient and works well when the hand is clearly visible.

\newpage

## 2.3 Limitations of Camera-Based Systems

Despite their convenience, camera systems have several limitations:

1. Lighting variation can reduce hand detection accuracy.
2. Complex backgrounds can confuse detection models.
3. Occlusion can hide fingers.
4. Camera angle changes the observed shape.
5. Motion blur can reduce frame quality.
6. Users must remain inside the camera frame.
7. Privacy concerns may arise when cameras are always active.

These limitations motivated the transition from the camera case to the hardware glove case. The glove avoids some visual problems by measuring finger bend directly. However, it introduces new challenges such as sensor drift, wiring, calibration, and user comfort.

## 2.4 Wearable Glove Recognition

Wearable glove recognition systems usually include bend sensors, pressure sensors, IMUs, or conductive textiles. A flex sensor changes resistance when bent. By placing flex sensors along the fingers, the system can estimate whether each finger is straight, partially bent, or fully bent. This information is highly relevant for ASL fingerspelling because many letters are defined by finger configurations.

An IMU adds orientation and motion information. The MPU6050 contains a three-axis accelerometer and a three-axis gyroscope. In GestureTalk, the accelerometer readings are converted into pitch and roll angles. Pitch and roll help distinguish gestures that have similar finger bend states but different hand orientation.

## 2.5 Flex Sensors

A flex sensor is a resistive component whose resistance increases or changes as it bends. In a typical Arduino circuit, the flex sensor is used in a voltage divider. The Arduino reads an analog voltage that corresponds to the bend amount. Calibration is important because different sensors may have different base resistance, and users may mount sensors differently on the glove.

GestureTalk uses five flex values:

| Feature | Finger |
|---|---|
| `f1` | Thumb |
| `f2` | Index |
| `f3` | Middle |
| `f4` | Ring |
| `f5` | Pinky |

\newpage

## 2.6 MPU6050 and Orientation Estimation

The MPU6050 is a widely used IMU module. It provides accelerometer and gyroscope data through I2C communication. In GestureTalk, raw accelerometer values are converted into pitch and roll angles on the Arduino. The equations are:

```text
pitch = atan2(ax, sqrt(ay^2 + az^2)) * 180 / pi
roll  = atan2(ay, sqrt(ax^2 + az^2)) * 180 / pi
```

Pitch and roll are not a full orientation representation because yaw is not included. However, they provide useful information about hand tilt. For an academic prototype, pitch and roll offer a compact and understandable orientation feature set.

## 2.7 Deep Learning for Time-Series Gesture Recognition

Gesture recognition from sensor data is a time-series classification problem. A single sensor row can describe a moment, but gestures often include transitions and movement. For example, a letter may involve moving into a pose, holding it, or rotating the wrist. Temporal models can learn these patterns.

Convolutional neural networks can be applied to one-dimensional time-series data using Conv1D layers. These layers learn local patterns across nearby timesteps. LSTM networks learn longer temporal dependencies. A bidirectional LSTM processes the sequence in both forward and backward directions, which is useful when the model classifies a complete fixed window after it has been collected.

GestureTalk uses a CNN + BiLSTM architecture. The CNN layers learn local bend and orientation changes. The BiLSTM layer learns sequence-level context.

## 2.8 Classical Machine Learning Baselines

A baseline model is important because it helps determine whether deep learning is actually improving performance. GestureTalk uses a Random Forest baseline. The Random Forest receives the same normalized temporal window, but the `64 x 17` tensor is flattened into 1088 features. Random Forests are strong classical models for structured data and are fast to train.

The deep model achieved 86.76 percent accuracy, while the Random Forest achieved 79.41 percent accuracy. This improvement supports the decision to use temporal deep learning.

\newpage

## 2.9 Comparison of Camera and Hardware Approaches

| Criterion | Camera + MediaPipe | Flex + MPU6050 Glove |
|---|---|---|
| User hardware | Webcam only | Wearable glove |
| Lighting dependency | High | Low |
| Background dependency | Medium to high | Low |
| Finger bend sensing | Indirect | Direct |
| Orientation sensing | Visual estimate | IMU-derived pitch/roll |
| Occlusion sensitivity | High | Low |
| Calibration requirement | Camera/model setup | Sensor and glove calibration |
| Comfort | No wearable burden | Wearable device required |
| Deployment complexity | Software focused | Hardware and software |

The two approaches are complementary. Camera input is easy to access and useful for quick demonstrations. Hardware input can be more robust in visually difficult environments. A future system could combine both.

## 2.10 Research Gap Addressed

Many academic projects present either a camera system or a glove system. GestureTalk is valuable because it documents the transition from a camera prototype to a real hardware glove pipeline. It also emphasizes reproducibility: the dataset is validated, preprocessing is saved, models are exported, metrics are written to JSON, and report artifacts are generated from project outputs.

\newpage

# Chapter 3: Proposed System and System Design

## 3.1 Overview

GestureTalk is an end-to-end gesture recognition system. It includes data acquisition, preprocessing, model training, evaluation, offline prediction, live inference, TTS output, and Arduino feedback. The current primary flow is:

```text
Glove sensors
-> Arduino
-> serial CSV rows
-> Python parser
-> sliding temporal window
-> saved preprocessor
-> CNN + BiLSTM model
-> class probabilities
-> smoothed prediction
-> text/speech/Arduino output
```

The earlier camera flow is:

```text
Webcam
-> OpenCV frame
-> MediaPipe Hands
-> 21 landmarks
-> 63-feature vector
-> dense neural network
-> predicted letter
```

## 3.2 Hardware Architecture

The hardware glove consists of:

1. Five flex sensors mounted along the fingers.
2. One MPU6050 module mounted on the hand or wrist.
3. Arduino Uno or compatible microcontroller.
4. Voltage divider circuits for analog flex readings.
5. I2C wiring for MPU6050 communication.
6. USB serial or wireless serial link to the laptop.
7. Optional output device such as OLED, LCD, buzzer, or speaker module.

The Arduino reads flex sensor values and computes pitch and roll from accelerometer readings. It prints each sample as one comma-separated row.

## 3.3 Current Sensor Format

The active project format is:

```text
f1,f2,f3,f4,f5,pitch,roll
```

This format is used in `hardware_config.py`, `sensor_utils.py`, `data_collector.py`, `predict_csv.py`, `inference.py`, and the Arduino sketch.

\newpage

## 3.4 Flex Sensor Working

Each flex sensor is read through an analog pin. When the finger bends, the sensor resistance changes. The Arduino converts the resulting voltage into a numeric analog value. In the dataset summary, flex values are already normalized between approximately 0 and 1. The min and max values observed in the dataset are:

| Feature | Minimum | Maximum |
|---|---:|---:|
| `f1` | 0.000 | 1.000 |
| `f2` | 0.050 | 1.000 |
| `f3` | 0.000 | 0.984 |
| `f4` | 0.000 | 1.000 |
| `f5` | 0.000 | 1.000 |

These values indicate that the project pipeline expects normalized or scaled sensor readings rather than raw 0 to 1023 ADC values.

## 3.5 MPU6050 Working

The MPU6050 communicates with the Arduino through I2C. On an Arduino Uno, I2C uses A4 and A5. This creates an important hardware limitation: if A4 and A5 are used by the MPU6050, the Uno does not have enough free analog pins for five flex sensors unless an analog multiplexer or external ADC is used. A Nano with A6/A7, Arduino Mega, ADS1115 module, CD4051/CD4067 multiplexer, or ESP32 can solve this limitation.

GestureTalk computes pitch and roll from accelerometer data. The project does not require raw accelerometer and gyroscope values in the current model.

## 3.6 Arduino Data Acquisition

The Arduino performs the following steps:

1. Initialize serial communication.
2. Initialize I2C communication.
3. Wake the MPU6050.
4. Calibrate accelerometer offsets.
5. Read flex sensors.
6. Read MPU6050 accelerometer values.
7. Compute pitch and roll.
8. Print one CSV row.
9. Listen for `PRED:` and `SENT:` messages from Python.

\newpage

## 3.7 Serial Communication

The laptop receives rows through PySerial. Each row must contain exactly seven numeric values. Invalid rows are ignored during real-time inference and recorded during dataset loading. This design is important because serial communication may produce blank lines or malformed values during startup, reset, or cable disturbance.

Example row:

```text
0.82,0.15,0.20,0.74,0.91,12.35,-4.80
```

## 3.8 Camera-Based Software Architecture

The camera application uses Streamlit for the interface. OpenCV captures webcam frames. MediaPipe Hands estimates hand landmarks. The dense neural network predicts a class. A stability buffer prevents flickering output. The camera mode also supports spelling windows and text-to-speech.

Camera model input:

```text
21 landmarks x 3 coordinates = 63 features
```

Dense model:

```text
Input(63)
Dense(128)
Dropout(0.2)
Dense(256)
Dropout(0.3)
Dense(128)
Softmax
```

## 3.9 Hardware Software Architecture

The hardware software architecture is modular:

| File | Role |
|---|---|
| `hardware_config.py` | Feature names and default constants |
| `sensor_utils.py` | Parsing, preprocessing, dataset loading, smoothing |
| `data_collector.py` | Recording labeled samples |
| `train_augmented_deep.py` | Main training and evaluation pipeline |
| `predict_csv.py` | Offline CSV prediction |
| `inference.py` | Live serial inference and sentence mode |
| `speech_utils.py` | Text-to-speech abstraction |
| `generate_final_report.py` | Word report generation |

\newpage

## 3.10 Block Diagram

```text
+-------------------+       +-------------------+       +--------------------+
| Flex Sensors       |       | MPU6050           |       | Arduino            |
| f1..f5             |       | pitch, roll       |       | CSV serial output  |
+---------+---------+       +---------+---------+       +---------+----------+
          |                           |                           |
          +---------------------------+---------------------------+
                                                              |
                                                              v
                                                  +-----------+-----------+
                                                  | Python Serial Reader |
                                                  +-----------+-----------+
                                                              |
                                                              v
                                                  +-----------+-----------+
                                                  | Preprocessor          |
                                                  | 64 x 17 tensor        |
                                                  +-----------+-----------+
                                                              |
                                                              v
                                                  +-----------+-----------+
                                                  | CNN + BiLSTM Model   |
                                                  +-----------+-----------+
                                                              |
                          +-----------------------------------+----------------------------------+
                          v                                   v                                  v
                    Text prediction                  Text-to-speech                      Arduino write-back
```

## 3.11 Design Justification

The design separates data acquisition from machine learning. The Arduino is responsible for simple sensor collection, while Python handles preprocessing and model inference. This is appropriate because the Arduino Uno is not powerful enough for the trained model. It also makes debugging easier: raw data can be saved, replayed, predicted offline, and used for retraining.

\newpage

# Chapter 4: Dataset, Preprocessing, and Feature Engineering

## 4.1 Dataset Overview

The current dataset is stored in:

```text
dataset2/
```

The dataset uses a flat CSV layout. Example files include:

```text
dataset2/A.csv
dataset2/B.csv
dataset2/C.csv
dataset2/openhand.csv
dataset2/point.csv
dataset2/thumbsup.csv
```

Labels are detected dynamically from filenames. The loader normalizes labels:

```text
openhand.csv -> OPEN
point.csv -> POINT
thumbsup.csv -> THUMBSUP
k.csv -> K
```

The dataset contains 29 classes and 339 real source temporal windows after windowing.

## 4.2 Dataset Classes

The detected classes are:

```text
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
OPEN, P, POINT, Q, R, S, T, THUMBSUP, U, V, W, X, Y, Z
```

This includes alphabet gestures and additional command-like gestures. OPEN and POINT are useful for interaction modes, while THUMBSUP can be mapped to confirmation or clearing in sentence mode.

## 4.3 Class Distribution

| Class | Windows |
|---|---:|
| A | 20 |
| B | 21 |
| C | 17 |
| D | 17 |
| E | 12 |
| F | 7 |
| G | 18 |
| H | 13 |
| I | 9 |
| J | 8 |
| K | 7 |
| L | 10 |
| M | 10 |
| N | 12 |
| O | 8 |
| OPEN | 19 |
| P | 8 |
| POINT | 16 |
| Q | 7 |
| R | 10 |
| S | 10 |
| T | 11 |
| THUMBSUP | 11 |
| U | 9 |
| V | 7 |
| W | 9 |
| X | 12 |
| Y | 12 |
| Z | 9 |

\newpage

## 4.4 Data Validation

Every valid row must contain exactly seven numeric values. The loader checks:

1. Number of columns.
2. Numeric conversion.
3. Finite values.
4. Correct feature order.

The dataset summary reported four malformed blank rows in `dataset2/A.csv`. These rows were skipped and recorded in `dataset_summary.json`. The rest of the dataset was valid.

Malformed row handling is important because real serial recordings often contain incomplete lines, blank rows, or startup noise. A robust loader should not fail completely because of a small number of bad rows.

## 4.5 Windowing

The model expects fixed-length temporal input. The default window size is 64 timesteps, and the stride is 16. Long recordings are sliced into overlapping windows. This increases the number of training examples while preserving temporal continuity.

Windowing can be represented as:

```text
recording rows -> [0:64], [16:80], [32:96], ...
```

This design allows each recording stream to produce multiple training samples.

## 4.6 Preprocessing Steps

The preprocessor applies:

1. Moving-average smoothing.
2. Resampling to fixed length.
3. Feature engineering.
4. Standardization.

The standardization equation is:

```text
z = (x - mean) / standard_deviation
```

The fitted mean and scale are stored in:

```text
hardware_preprocessor.pkl
```

The same preprocessor must be used during inference.

\newpage

## 4.7 Feature Engineering

The raw input has seven features:

```text
f1,f2,f3,f4,f5,pitch,roll
```

The preprocessor adds:

1. Flex mean.
2. Flex spread.
3. Orientation magnitude.
4. Delta of all seven raw features.

The feature expansion is:

```text
7 raw features
+ 1 flex mean
+ 1 flex spread
+ 1 orientation magnitude
+ 7 deltas
= 17 total features
```

Final input shape:

```text
64 timesteps x 17 features
```

## 4.8 Flex Mean

Flex mean summarizes overall hand closure:

```text
flex_mean = mean(f1, f2, f3, f4, f5)
```

This helps distinguish open-hand and closed-hand gestures.

## 4.9 Flex Spread

Flex spread measures the difference between the most bent and least bent finger:

```text
flex_spread = max(f1..f5) - min(f1..f5)
```

This feature helps distinguish gestures where some fingers are extended and others are bent.

## 4.10 Orientation Magnitude

Orientation magnitude combines pitch and roll:

```text
orientation_mag = sqrt(pitch^2 + roll^2)
```

This gives a compact representation of hand tilt intensity.

\newpage

## 4.11 Temporal Delta Features

Delta features measure change between consecutive timesteps:

```text
delta_t = x_t - x_(t-1)
```

Delta features help the model understand motion, not just static pose. This is useful for dynamic letters and for transitions into a gesture.

## 4.12 Augmentation Strategy

The project does not generate random synthetic gestures. Instead, every augmented sample begins from a real training window. Augmentation methods include:

1. Gaussian sensor noise.
2. Temporal stretching and compression.
3. Small flex drift.
4. Pitch and roll perturbation.
5. Sequence jitter.
6. Frame dropout.
7. Smoothing variation.

This approach reflects real hardware conditions: sensors are noisy, users move at different speeds, flex sensors drift, serial timing can vary, and small orientation changes occur naturally.

## 4.13 Why Test Data Is Not Augmented

Only the training split is augmented. The test split remains real and unaugmented. This is important because evaluation should measure real recognition performance, not recognition of modified copies of training examples.

Current split:

| Item | Count |
|---|---:|
| Real source windows | 339 |
| Real training windows | 271 |
| Augmented training samples | 2168 |
| Real unaugmented test samples | 68 |
| Processed total samples | 2236 |
| Augmentation factor | 7 |

\newpage

## 4.14 Dataset Assumptions

The current dataset appears to contain one primary recording stream per class. Train and test windows may therefore come from the same recording session, even though test windows are real and unaugmented. This is acceptable for a prototype demonstration but not enough for a production-level claim. A stronger evaluation would collect separate sessions and split by session or user.

Important assumptions:

1. The same glove configuration is used during training and inference.
2. Sensor order remains fixed.
3. Flex values are normalized consistently.
4. Pitch and roll are computed before reaching Python.
5. The saved preprocessor is reused during inference.
6. Users perform gestures similarly to the training recordings.

## 4.15 Dataset Artifacts

The preprocessing and training pipeline generates:

```text
processed_dataset.npz
hardware_preprocessor.pkl
X_test.npy
y_test.npy
dataset_summary.json
```

`processed_dataset.npz` stores arrays, labels, and split information. `hardware_preprocessor.pkl` stores the fitted normalization parameters.

\newpage

# Chapter 5: Model Architecture and Training Methodology

## 5.1 Model Selection

GestureTalk uses a CNN + BiLSTM model for the current hardware system. This choice is based on the nature of the input. The glove data is temporal: each sample is a sequence of sensor readings. Conv1D layers are useful for extracting local temporal patterns, while LSTM layers are useful for learning sequence context.

The Random Forest baseline is retained to provide a fair comparison. If a classical model performs similarly or better, a deep model may not be justified. In this project, the deep model outperformed the baseline.

## 5.2 CNN + BiLSTM Architecture

The final deep learning architecture is:

```text
Input: 64 x 17
Conv1D: 32 filters, kernel size 5
BatchNormalization
MaxPooling1D
Dropout
Conv1D: 48 filters, kernel size 3
BatchNormalization
MaxPooling1D
Dropout
Bidirectional LSTM: 32 units
Dense: 64 ReLU units
Dropout
Softmax: 29 classes
```

## 5.3 Role of Conv1D Layers

Conv1D layers scan across time. They can learn patterns such as:

1. A finger bending quickly.
2. A hand rotating slightly.
3. A stable hold after movement.
4. A sensor drift pattern.
5. A short transition between neutral and gesture.

These patterns are local, so convolution is appropriate.

## 5.4 Role of BiLSTM

The BiLSTM reads the sequence in both directions. Since the model classifies a complete window rather than predicting at each timestep, it can benefit from both past and future context inside that window. A BiLSTM can learn whether a transition leads into a stable pose or whether the entire window represents a different movement pattern.

\newpage

## 5.5 Softmax Output

The final layer uses softmax. For 29 classes, the output is a probability vector:

```text
p = [p1, p2, ..., p29]
```

The predicted class is:

```text
class = argmax(p)
```

The confidence is the maximum probability.

## 5.6 Training Workflow

The main training workflow is:

```text
Load dataset2
-> validate rows
-> window recordings
-> split train/test
-> augment training windows
-> fit preprocessor on training data
-> transform train/test
-> train Random Forest baseline
-> train CNN + BiLSTM
-> evaluate on real test windows
-> save metrics and plots
```

The main command is:

```powershell
venv\Scripts\python.exe train_augmented_deep.py --dataset-dir dataset2 --augmentation-factor 7 --epochs 12 --batch-size 32 --window-size 64 --stride 16
```

## 5.7 Loss Function

The model uses categorical cross-entropy:

```text
L = - sum(y_i * log(p_i))
```

where `y_i` is the true one-hot label and `p_i` is the predicted probability.

## 5.8 Optimization

The model uses an adaptive optimizer such as Adam. Adam is commonly used because it adjusts learning rates for each parameter and works well across many deep learning tasks.

\newpage

## 5.9 Baseline Model

The Random Forest baseline receives flattened normalized windows:

```text
64 x 17 = 1088 features
```

Random Forest is useful because it:

1. Trains quickly.
2. Handles structured features well.
3. Requires less tuning.
4. Provides a strong non-neural comparison.

## 5.10 Evaluation Metrics

The project reports:

1. Accuracy.
2. Macro precision.
3. Macro recall.
4. Macro F1.
5. Weighted metrics.
6. Per-class accuracy.
7. Confusion matrix.
8. Mean confidence.
9. Weak-performing classes.
10. Most confused gesture pairs.

Macro F1 is especially important because the dataset is class-imbalanced. A model could achieve high accuracy by performing well on larger classes, but macro F1 treats each class equally.

## 5.11 Model Artifacts

The training process generates:

```text
hardware_gesture_model.keras
hardware_gesture_model.pkl
training_history.json
evaluation_metrics.json
accuracy_plot.png
loss_plot.png
confusion_matrix.png
baseline_confusion_matrix.png
metrics_comparison.png
conclusion.txt
```

\newpage

# Chapter 6: Implementation Details

## 6.1 Programming Environment

The project is implemented primarily in Python and Arduino C/C++. The Python environment uses:

1. NumPy for numerical arrays.
2. pandas where tabular handling is needed.
3. TensorFlow/Keras for deep learning.
4. scikit-learn for the Random Forest baseline and metrics.
5. Matplotlib for plots.
6. PySerial for serial communication.
7. OpenCV and MediaPipe for the camera system.
8. pyttsx3 and PowerShell speech fallback for TTS.
9. Streamlit for the camera application.

## 6.2 Key Files

| File | Purpose |
|---|---|
| `app.py` | Camera-based Streamlit spelling app |
| `extract_keypoints_from_images.py` | Extracts MediaPipe hand keypoints from images |
| `train_keypoint_model.py` | Trains the camera keypoint classifier |
| `hardware_config.py` | Stores hardware constants |
| `sensor_utils.py` | Shared parsing, preprocessing, dataset, and smoothing utilities |
| `data_collector.py` | Records labeled glove samples |
| `train_augmented_deep.py` | Main hardware training pipeline |
| `predict_csv.py` | Predicts one CSV recording |
| `inference.py` | Live serial prediction, sentence mode, TTS, write-back |
| `speech_utils.py` | Text-to-speech helper |
| `generate_final_report.py` | Generates the Word report |

## 6.3 `hardware_config.py`

This file defines:

```python
FEATURE_NAMES = ["f1", "f2", "f3", "f4", "f5", "pitch", "roll"]
DEFAULT_WINDOW_SIZE = 64
DEFAULT_STRIDE = 16
DEFAULT_DATASET_DIR = "dataset2"
DEFAULT_MODEL_PATH = "hardware_gesture_model.keras"
DEFAULT_PREPROCESSOR_PATH = "hardware_preprocessor.pkl"
```

\newpage

## 6.4 `sensor_utils.py`

`sensor_utils.py` is one of the most important files. It contains:

1. `parse_sensor_line()`
2. `open_serial()`
3. `read_sensor_rows()`
4. `moving_average()`
5. `resample_sequence()`
6. `add_engineered_features()`
7. `SequencePreprocessor`
8. `load_sensor_dataset()`
9. `write_dataset_summary()`
10. `PredictionSmoother`

This file ensures that training and inference use the same parsing and preprocessing logic.

## 6.5 `SequencePreprocessor`

The preprocessor performs:

```text
raw sequence
-> moving average
-> resampling
-> feature engineering
-> standardization
```

It stores mean and scale values after fitting. During inference, it transforms one sliding window using those same values.

## 6.6 `train_augmented_deep.py`

This script is the final training pipeline. It handles the full process from dataset loading to report artifact generation. It is preferred over older scripts because it includes augmentation, baseline comparison, and deep model evaluation in one workflow.

## 6.7 `predict_csv.py`

This script predicts one recorded CSV file. Example:

```powershell
venv\Scripts\python.exe predict_csv.py dataset2\A.csv --preprocessor hardware_preprocessor.pkl --top-k 5
```

Verified output:

```text
Predicted Gesture: A (1.00)
```

It now also supports:

```powershell
--speak
```

\newpage

## 6.8 `inference.py`

`inference.py` performs live serial inference. It:

1. Opens the serial port.
2. Reads sensor rows.
3. Parses valid rows.
4. Maintains a sliding window.
5. Applies the saved preprocessor.
6. Predicts probabilities.
7. Smooths predictions.
8. Prints stable labels.
9. Optionally speaks labels.
10. Optionally sends labels back to the Arduino.
11. Optionally builds continuous sentences.

Live command:

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200
```

Sentence mode:

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```

## 6.9 Text-to-Speech Implementation

Text-to-speech is handled through `speech_utils.py`. The primary engine is `pyttsx3`. On Windows, if pyttsx3 fails because SAPI is not registered, the system can fall back to PowerShell's `System.Speech.Synthesis.SpeechSynthesizer`.

This design makes TTS more robust across local Windows environments.

## 6.10 Arduino Write-Back

When write-back is enabled, Python sends:

```text
PRED:A
SENT:HELLO WORLD
```

The Arduino sketch stores these values as:

```cpp
String lastPrediction;
String lastSentence;
```

These can later be displayed on an OLED or LCD.

\newpage

# Chapter 7: Experimental Results and Analysis

## 7.1 Experimental Setup

The experiment used real recorded glove data from `dataset2/`. The data was windowed into 64-timestep samples. The train split was augmented with a factor of 7, while the test split remained real and unaugmented.

Dataset summary:

| Metric | Value |
|---|---:|
| Detected classes | 29 |
| Real source windows | 339 |
| Real training windows | 271 |
| Augmented training samples | 2168 |
| Real test samples | 68 |
| Input shape | 64 x 17 |

## 7.2 Deep Learning Results

The CNN + BiLSTM model achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.8676 |
| Macro precision | 0.8851 |
| Macro recall | 0.8966 |
| Macro F1 | 0.8897 |
| Mean confidence | 0.9068 |
| Keras evaluated accuracy | 0.8676 |
| Loss | 0.1433 |

## 7.3 Baseline Results

The Random Forest baseline achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.7941 |
| Macro precision | 0.8276 |
| Macro recall | 0.8276 |
| Macro F1 | 0.8161 |
| Mean confidence | 0.8868 |

\newpage

## 7.4 Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest baseline | 79.41% | 0.8161 |
| CNN + BiLSTM | 86.76% | 0.8897 |

Improvement:

```text
Accuracy improvement = +0.0735
Macro F1 improvement = +0.0736
```

The deep model improves both accuracy and macro F1. This suggests that temporal learning helps the system recognize glove gestures more effectively than a flattened classical baseline.

## 7.5 Per-Class Accuracy

The deep model achieved perfect accuracy on many classes, including A, B, E, F, G, H, I, J, K, L, M, N, OPEN, POINT, Q, R, S, T, THUMBSUP, U, V, W, X, Y, and Z. Weak classes were:

```text
C, D, O, P
```

These classes are difficult because they can have similar flex sensor patterns. For example, C and D may share similar partial finger bending, while O and P can have similar curved or oriented hand patterns.

## 7.6 Most Confused Gestures

| True Class | Predicted Class | Count |
|---|---|---:|
| D | C | 4 |
| C | D | 2 |
| O | P | 2 |
| P | O | 1 |

The confusion pairs are meaningful from a sensing perspective. Flex sensors cannot directly detect exact fingertip contact, palm shape, or thumb placement. Adding more sensors or fusing camera data could improve these classes.

\newpage

## 7.7 Discussion of Results

The result is strong for a small academic dataset. The deep model achieves high accuracy and macro F1, and it improves over the baseline. However, the result should be interpreted carefully. The test set is real and unaugmented, but train and test windows may originate from the same recording sessions. This means the evaluation may not fully represent new users, new sessions, or different glove mounting conditions.

The weak classes reveal the limits of the current sensor design. Some ASL signs require detailed geometry that five flex sensors and pitch/roll cannot fully capture. The system can sense bend and tilt, but not exact finger separation, palm orientation yaw, or thumb contact location.

## 7.8 Confusion Matrix Interpretation

The confusion matrix should be read as a map of class-level errors. Strong diagonal values indicate correct predictions. Off-diagonal values indicate confusions. The major off-diagonal values occur in C/D and O/P. This supports the need for:

1. More samples for weak classes.
2. Better calibration.
3. More orientation features.
4. Additional sensing for thumb and fingertip contact.
5. Possible camera-glove fusion.

## 7.9 Training Plots

The project generates:

```text
accuracy_plot.png
loss_plot.png
```

The accuracy plot shows how training and validation accuracy change across epochs. The loss plot shows whether the model is still learning or beginning to overfit. These graphs are useful for project demonstration because they show the training process rather than only final metrics.

## 7.10 Generated Report Figures

The project includes:

```text
system_architecture.png
preprocessing_pipeline.png
augmentation_pipeline.png
model_architecture.png
metrics_comparison.png
confusion_matrix.png
baseline_confusion_matrix.png
```

These figures can be inserted into a final Word or PDF report.

\newpage

# Chapter 8: Real-Time Inference, TTS, and Wireless Extension

## 8.1 Real-Time Inference Flow

The live inference system reads continuous serial data from the Arduino. It stores recent rows in a fixed-size buffer. Once the buffer contains 64 rows, the window is preprocessed and passed to the model. Predictions are smoothed using a vote window and confidence threshold.

Flow:

```text
Serial row
-> parse
-> append to buffer
-> if buffer full, preprocess
-> model prediction
-> smoother
-> print/speak/write back
```

## 8.2 Prediction Smoothing

Real-time predictions can flicker due to sensor noise. The `PredictionSmoother` stores recent predictions and selects the most stable label. A confidence threshold avoids accepting weak predictions.

Default parameters include:

```text
window size = 64
vote window = 7
threshold = 0.55
```

These can be changed from the command line.

## 8.3 Sentence Mode

The project now includes continuous sentence mode. In this mode:

```text
User performs gestures continuously
-> stable letters are accepted once
-> repeated predictions are cleaned
-> command gestures insert space, backspace, speak, or clear
-> current sentence can be spoken
```

Command:

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```

\newpage

## 8.4 Default Sentence Controls

| Gesture | Action |
|---|---|
| A to Z | Append letter |
| OPEN or SPACE | Insert space |
| POINT, DELETE, or DEL | Backspace |
| FIST | Speak sentence |
| THUMBS_UP or CLEAR | Clear sentence |

These controls are configurable:

```powershell
venv\Scripts\python.exe inference.py --sentence-mode --space-gestures OPEN --speak-gestures FIST --clear-gestures THUMBS_UP
```

## 8.5 Repeat Cleaning

When a user holds a gesture, the model may predict the same class many times. Sentence mode accepts the same stable gesture only once while it is held. To type the same letter again, the user relaxes or moves through a low-confidence state and then repeats the gesture.

Example:

```text
A A A A while held -> A
relax
A again -> AA
```

## 8.6 Text-to-Speech

Text-to-speech improves accessibility because the predicted output becomes audible. The project supports:

1. Speaking one CSV prediction.
2. Speaking stable live predictions.
3. Speaking a complete sentence in sentence mode.

CSV speech:

```powershell
venv\Scripts\python.exe predict_csv.py dataset2\A.csv --preprocessor hardware_preprocessor.pkl --top-k 5 --speak
```

Live speech:

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200 --speak
```

\newpage

## 8.7 Arduino Write-Back

The live inference script can send predicted labels and sentences back to the Arduino:

```text
PRED:A
SENT:HELLO WORLD
```

The Arduino can display these on an OLED, LCD, or serial-connected module. This makes the glove more interactive because the prediction can return to the wearable device instead of remaining only on the laptop screen.

## 8.8 Wireless Design

The user asked whether the laptop must remain connected to the Arduino Uno. The answer is that the system can be made wireless, but the Uno should still not run the ML model. The practical design is:

```text
Arduino Uno glove
-> Bluetooth serial or Wi-Fi serial
-> laptop Python model
-> prediction
-> wireless serial back to Arduino
-> display or speaker
```

## 8.9 Bluetooth Option

The easiest wireless option for Arduino Uno is an HC-05 or HC-06 Bluetooth module. It acts like a serial cable replacement. The laptop sees it as a COM port. The existing Python code can still use PySerial.

Advantages:

1. Simple.
2. Low cost.
3. Works with existing serial code.
4. Good for demonstrations.

Limitations:

1. Pairing setup is required.
2. Range is limited.
3. Bandwidth is lower than USB, though enough for this data rate.

## 8.10 ESP32 Option

The best long-term option is replacing the Uno with an ESP32. ESP32 has built-in Bluetooth and Wi-Fi, more memory, more processing power, and more flexible I/O. It still may not run the full Keras model directly, but it is better for wireless streaming and future TensorFlow Lite Micro experiments.

\newpage

## 8.11 Edge AI Possibility

Running the model directly on the glove would require converting the model to a small embedded format. The current CNN + BiLSTM model is not designed for Arduino Uno. Future edge deployment could use:

1. TensorFlow Lite.
2. TensorFlow Lite Micro.
3. Edge Impulse.
4. ESP32-S3.
5. Raspberry Pi Zero 2 W.
6. Arduino Nano 33 BLE Sense.

For the current system, laptop inference is the correct engineering choice.

\newpage

# Chapter 9: Limitations and Risk Analysis

## 9.1 Dataset Limitations

The dataset is small and may contain one recording stream per class. This means train and test windows may come from the same recording session. The test set is real and unaugmented, but it may not be session-independent.

Future work should collect:

1. Multiple recordings per class.
2. Multiple users.
3. Multiple days or sessions.
4. Different glove placements.
5. Different signing speeds.

## 9.2 Sensor Limitations

Five flex sensors and pitch/roll cannot fully describe hand shape. The system does not directly sense:

1. Fingertip contact.
2. Palm shape.
3. Exact thumb position.
4. Finger separation.
5. Yaw orientation.
6. Hand location in space.

This explains confusion between C/D and O/P.

## 9.3 Hardware Limitations

Arduino Uno has limited analog input availability. A4 and A5 are used for I2C with MPU6050. Therefore, five flex sensors plus MPU6050 may require an analog multiplexer, an external ADC, or a different board.

Recommended hardware improvements:

1. Arduino Mega for more analog pins.
2. ADS1115 external ADC.
3. CD4051 or CD4067 analog multiplexer.
4. ESP32 with wireless support.
5. Better connectors and strain relief.

\newpage

## 9.4 Model Limitations

The model recognizes isolated gestures, not complete ASL grammar. Sentence mode combines predicted letters and command gestures, but it does not understand ASL syntax. It is closer to fingerspelling plus command recognition than full language translation.

The model also lacks unknown gesture rejection. If the user performs a gesture outside the training set, the softmax layer may still assign a known class with high confidence. A production system should include rejection thresholds, anomaly detection, or an "unknown" class.

## 9.5 Real-Time Risks

Real-time use introduces:

1. Serial noise.
2. Timing delays.
3. Repeated predictions.
4. Hand transition errors.
5. TTS blocking delays.
6. Wireless latency.

The project addresses some of these through smoothing, repeat-cleaning, and confidence thresholds. More advanced systems could use asynchronous inference and non-blocking speech.

## 9.6 Ethical and Accessibility Considerations

Gesture recognition systems should not be presented as replacing human interpreters or complete sign language understanding. ASL is a natural language with cultural and grammatical depth. This project is an assistive prototype focused on isolated gestures and fingerspelling-style communication.

The system should be framed as a learning, demonstration, and accessibility support tool, not a complete translator.

\newpage

# Chapter 10: Conclusion and Future Work

## 10.1 Conclusion

GestureTalk demonstrates a complete end-to-end ASL-style gesture recognition project. The project began with a camera and MediaPipe system that extracted 63 hand landmark features and classified them using a dense neural network. It then evolved into a hardware glove system using five flex sensors and MPU6050-derived pitch and roll angles. The current hardware system processes real recorded CSV data, validates rows, dynamically detects labels, slices temporal windows, augments training data, preserves a real test set, and trains a CNN + BiLSTM model.

The latest hardware model achieved 86.76 percent test accuracy and 0.8897 macro F1 across 29 classes. It outperformed the Random Forest baseline, which achieved 79.41 percent accuracy and 0.8161 macro F1. The results show that temporal deep learning is useful for glove-based gesture recognition. The main weak classes were C, D, O, and P, which are difficult because they have similar bend and orientation patterns.

The project also includes practical deployment features: offline CSV prediction, live serial inference, prediction smoothing, text-to-speech, sentence mode, and Arduino write-back. Wireless operation is possible using Bluetooth serial or ESP32-based communication, while keeping model inference on the laptop.

## 10.2 Achievements

The project achieved:

1. Camera-based software recognition.
2. Hardware glove data format.
3. Real dataset loading and validation.
4. Temporal preprocessing.
5. Realistic training augmentation.
6. CNN + BiLSTM classification.
7. Baseline model comparison.
8. Detailed metrics and confusion analysis.
9. Real-time serial inference.
10. Text-to-speech output.
11. Continuous sentence mode.
12. Arduino feedback support.
13. IEEE-style documentation.

\newpage

## 10.3 Future Work

Future improvements should include:

1. Collecting more recordings per class.
2. Collecting data from multiple users.
3. Splitting train and test by session.
4. Adding yaw or magnetometer features.
5. Improving thumb and fingertip sensing.
6. Adding calibration mode.
7. Adding unknown gesture rejection.
8. Using an OLED display for glove feedback.
9. Testing Bluetooth wireless streaming.
10. Migrating to ESP32.
11. Converting the model to TensorFlow Lite.
12. Comparing CNN, LSTM, BiLSTM, TCN, and Transformer models.
13. Combining camera and glove features.
14. Adding language-model correction for spelled sentences.
15. Building a mobile or web dashboard.

## 10.4 Final Statement

GestureTalk is a strong academic prototype because it does not stop at model training. It includes data collection, preprocessing, augmentation, training, evaluation, reporting, offline prediction, live inference, speech output, and hardware feedback. It provides a practical foundation for future wearable sign recognition research and embedded assistive communication systems.

\newpage

# References

[1] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann, "MediaPipe Hands: On-device Real-time Hand Tracking," Workshop on Computer Vision for Augmented and Virtual Reality, 2020.  

[2] A. Nagaraj, "ASL Alphabet," Kaggle Dataset, 2018.  

[3] DataMunge, "Sign Language MNIST," Kaggle Dataset.  

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.  

[5] M. Abadi et al., "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems," arXiv:1603.04467, 2016.  

[6] G. Bradski, "The OpenCV Library," Dr. Dobb's Journal of Software Tools, 2000.  

[7] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.  

[8] L. Breiman, "Random Forests," Machine Learning, vol. 45, pp. 5-32, 2001.  

[9] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.  

[10] InvenSense, "MPU-6000 and MPU-6050 Product Specification," TDK InvenSense.  

[11] SparkFun Electronics, "Flex Sensor Hookup Guide."  

[12] Arduino, "Arduino Uno Rev3 Documentation."  

[13] Wokwi, "Wokwi Documentation."  

[14] Python Software Foundation, "Python Documentation."  

[15] Keras Team, "Keras Documentation."  

\newpage

# Appendix A: Reproduction Commands

## Train the Current Hardware Model

```powershell
venv\Scripts\python.exe train_augmented_deep.py --dataset-dir dataset2 --augmentation-factor 7 --epochs 12 --batch-size 32 --window-size 64 --stride 16
```

## Predict One CSV File

```powershell
venv\Scripts\python.exe predict_csv.py dataset2\A.csv --preprocessor hardware_preprocessor.pkl --top-k 5
```

## Predict and Speak One CSV File

```powershell
venv\Scripts\python.exe predict_csv.py dataset2\A.csv --preprocessor hardware_preprocessor.pkl --top-k 5 --speak
```

## Run Live Inference

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200
```

## Run Live Sentence Mode

```powershell
venv\Scripts\python.exe inference.py --port COM3 --baud 115200 --sentence-mode --speak --write-back
```

\newpage

# Appendix B: Important Output Files

```text
dataset2/                    real glove CSV dataset
hardware_config.py            feature names and constants
sensor_utils.py               preprocessing and parsing utilities
train_augmented_deep.py       main training pipeline
predict_csv.py                offline CSV prediction
inference.py                  real-time serial inference
speech_utils.py               text-to-speech helper
hardware_preprocessor.pkl     fitted preprocessor
hardware_gesture_model.keras  trained CNN + BiLSTM model
hardware_gesture_model.pkl    Random Forest baseline
processed_dataset.npz         processed arrays and labels
evaluation_metrics.json       final metrics
dataset_summary.json          dataset statistics
training_history.json         training history
confusion_matrix.png          deep model confusion matrix
baseline_confusion_matrix.png baseline confusion matrix
accuracy_plot.png             training accuracy graph
loss_plot.png                 training loss graph
final_report.docx             generated formal report
```

\newpage

# Appendix C: Current Dataset Summary

```text
Total classes: 29
Total real source windows: 339
Real training windows: 271
Augmented training samples: 2168
Real test windows: 68
Processed total samples: 2236
Window size: 64
Feature count per timestep: 17
Malformed rows removed: 4
Malformed source: dataset2/A.csv
```

\newpage

# Appendix D: Current Performance Summary

```text
CNN + BiLSTM accuracy: 0.8676
CNN + BiLSTM macro precision: 0.8851
CNN + BiLSTM macro recall: 0.8966
CNN + BiLSTM macro F1: 0.8897

Random Forest accuracy: 0.7941
Random Forest macro precision: 0.8276
Random Forest macro recall: 0.8276
Random Forest macro F1: 0.8161

Accuracy improvement: +0.0735
Macro F1 improvement: +0.0736
```

\newpage

# Appendix E: Page Expansion Notes for Final Formatting

This Markdown document is structured for conversion into a 40+ page formal report. When rendered to PDF or Word with normal academic formatting, use:

1. A title page.
2. Certificate page if required by the institution.
3. Declaration page if required.
4. Acknowledgements page.
5. Abstract page.
6. Table of contents.
7. List of figures.
8. List of tables.
9. Ten main chapters.
10. References.
11. Appendices.
12. Insert the generated diagrams and plots at the figure placeholders.
13. Keep page breaks marked by `\newpage`.
14. Use 12 pt Times New Roman, 1.5 line spacing, and standard margins for a full-length report.

The report text, tables, equations, command examples, figures, and appendices together form a complete 40+ page academic project paper when formatted with standard engineering project report spacing and images.
