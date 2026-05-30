#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;
// Uno uses A4/A5 for I2C, so a fifth flex sensor needs an analog mux/ADC.
// Nano can use A6 for the fifth flex sensor.
// Serial output order: F1=pinky, F2=thumb, F3=index, F4=middle, F5=ring.
const uint8_t FLEX_PINS[5] = {A0, A1, A2, A3, A6};
const uint16_t SAMPLE_DELAY_MS = 20;
const uint16_t CALIBRATION_SAMPLES = 250;

float accelOffset[3] = {0.0, 0.0, 0.0};
float gyroOffset[3] = {0.0, 0.0, 0.0};
float flexFiltered[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
const float FILTER_ALPHA = 0.25;
String lastPrediction = "";
String lastSentence = "";

void writeMpuRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

void readMpuRaw(int16_t *accel, int16_t *gyro) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14, (uint8_t)true);

  accel[0] = (Wire.read() << 8) | Wire.read();
  accel[1] = (Wire.read() << 8) | Wire.read();
  accel[2] = (Wire.read() << 8) | Wire.read();
  Wire.read();
  Wire.read();
  gyro[0] = (Wire.read() << 8) | Wire.read();
  gyro[1] = (Wire.read() << 8) | Wire.read();
  gyro[2] = (Wire.read() << 8) | Wire.read();
}

void calibrateMpu() {
  long accelSum[3] = {0, 0, 0};
  long gyroSum[3] = {0, 0, 0};
  int16_t accel[3], gyro[3];

  for (uint16_t i = 0; i < CALIBRATION_SAMPLES; i++) {
    readMpuRaw(accel, gyro);
    for (uint8_t axis = 0; axis < 3; axis++) {
      accelSum[axis] += accel[axis];
      gyroSum[axis] += gyro[axis];
    }
    delay(5);
  }

  for (uint8_t axis = 0; axis < 3; axis++) {
    accelOffset[axis] = accelSum[axis] / (float)CALIBRATION_SAMPLES;
    gyroOffset[axis] = gyroSum[axis] / (float)CALIBRATION_SAMPLES;
  }
  accelOffset[2] -= 16384.0;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  writeMpuRegister(0x6B, 0x00);
  writeMpuRegister(0x1C, 0x00);
  writeMpuRegister(0x1B, 0x00);
  delay(100);

  for (uint8_t i = 0; i < 5; i++) {
    flexFiltered[i] = analogRead(FLEX_PINS[i]);
  }
  calibrateMpu();
}

void handleHostInput() {
  if (!Serial.available()) {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.startsWith("PRED:")) {
    lastPrediction = line.substring(5);
    // Use lastPrediction for an LCD, OLED, LEDs, vibration motor, or speaker.
    // Avoid printing it here because the Python reader expects clean CSV rows.
  } else if (line.startsWith("SENT:")) {
    lastSentence = line.substring(5);
    // Use lastSentence for a larger display or scroll it on a small OLED.
  }
}

void loop() {
  handleHostInput();

  int16_t accelRaw[3], gyroRaw[3];
  readMpuRaw(accelRaw, gyroRaw);

  for (uint8_t i = 0; i < 5; i++) {
    float reading = analogRead(FLEX_PINS[i]);
    flexFiltered[i] = FILTER_ALPHA * reading + (1.0 - FILTER_ALPHA) * flexFiltered[i];
    Serial.print(flexFiltered[i], 2);
    Serial.print(",");
  }

  float ax = ((float)accelRaw[0] - accelOffset[0]) / 16384.0 * 9.80665;
  float ay = ((float)accelRaw[1] - accelOffset[1]) / 16384.0 * 9.80665;
  float az = ((float)accelRaw[2] - accelOffset[2]) / 16384.0 * 9.80665;
  float pitch = atan2(ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
  float roll = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;

  Serial.print(pitch, 5);
  Serial.print(",");
  Serial.println(roll, 5);

  delay(SAMPLE_DELAY_MS);
}
