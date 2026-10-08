/*
  TinyML Fan Guardian - ESP32
  -----------------------------------------
  Hardware:
    DHT22       -> GPIO 4
    MPU6050     -> SDA 21, SCL 22
    ACS712 5A   -> GPIO 34 (ADC)
    KY-038      -> GPIO 35 (ADC)
    Relay       -> GPIO 26
    Buzzer      -> GPIO 27
    Traffic LED -> GPIO 25 (GREEN), GPIO 32 (YELLOW), GPIO 33 (RED)

  TinyML:
    model/model_data.h
    model/normalization_data.h

  IMPORTANT:
    The trained model uses these feature units/order:
      temperature, humidity, vibration, current, sound

    Normalization:
      normalized = (value - scaler_mean[i]) / scaler_scale[i]

    The INT8 TFLite model produced by your Python pipeline currently
    reports FLOAT32 input/output. That is normal for your exported model:
    the internal weights/operations are quantized, while the public
    interpreter interface accepts float32 values.

  Libraries:
    - DHT sensor library by Adafruit
    - Adafruit Unified Sensor
    - Adafruit MPU6050
    - ESP32 board package
    - Espressif ESP-TFLite-Micro bundled with recent ESP32 core

  Copy these two generated files into this sketch folder:
    model_data.h
    normalization_data.h
*/

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <DHT.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "model_data.h"
#include "normalization_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------- Wi-Fi ----------------
const char* WIFI_SSID = "iPhone";
const char* WIFI_PASSWORD = "Andre123";

// ---------------- Pins -----------------
#define DHT_PIN       4
#define DHT_TYPE      DHT22

#define I2C_SDA       21
#define I2C_SCL       22

#define ACS712_PIN     34
#define SOUND_PIN      35

#define RELAY_PIN      26
#define BUZZER_PIN     27

#define TRAFFIC_GREEN  25
#define TRAFFIC_YELLOW 32
#define TRAFFIC_RED    33

// Change these if your relay/buzzer module is active LOW.
#define RELAY_ACTIVE_HIGH  true
#define BUZZER_ACTIVE_HIGH true

// ---------------- Objects ----------------
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_MPU6050 mpu;
WebServer server(80);

// ---------------- TinyML -----------------
// 16 KB is deliberately generous for this 3 KB model.
constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Recent sensor state
float temperature = NAN;
float humidity = NAN;
float vibration = NAN;
float currentRms = NAN;
float soundAdc = NAN;

int prediction = 1;
float confidence = 0.0f;
float healthScore = 78.0f;

bool relayState = false;
bool buzzerState = false;
String trafficState = "YELLOW";

unsigned long lastSampleMs = 0;
unsigned long lastInferenceMs = 0;
unsigned long bootMs = 0;

// ---------------- ACS712 ------------------
// ACS712 5A sensitivity is nominally 185 mV/A.
// The exact zero-current voltage varies by module, ADC reference,
// supply and installation, so calibrate ZERO_CURRENT_VOLTAGE below.
const float ACS_SENSITIVITY = 0.185f;       // V/A for 5A version
float ACS_ZERO_VOLTAGE = 2.50f;             // calibrate at 0 A

// ESP32 ADC conversion. ADC calibration differs between boards.
// This is an initial approximation; calibrate ACS712 for your hardware.
const float ADC_VREF = 3.30f;
const int ADC_MAX = 4095;

// Number of RMS samples.
const int ACS_SAMPLES = 300;

// ---------------- Utility ----------------
float adcToVoltage(int raw) {
  return (raw * ADC_VREF) / ADC_MAX;
}

void setRelay(bool on) {
  relayState = on;
  digitalWrite(RELAY_PIN,
               RELAY_ACTIVE_HIGH ? (on ? HIGH : LOW)
                                  : (on ? LOW : HIGH));
}

void setBuzzer(bool on) {
  buzzerState = on;
  digitalWrite(BUZZER_PIN,
               BUZZER_ACTIVE_HIGH ? (on ? HIGH : LOW)
                                   : (on ? LOW : HIGH));
}

void setTraffic(const String& state) {
  trafficState = state;

  digitalWrite(TRAFFIC_GREEN, LOW);
  digitalWrite(TRAFFIC_YELLOW, LOW);
  digitalWrite(TRAFFIC_RED, LOW);

  if (state == "GREEN") digitalWrite(TRAFFIC_GREEN, HIGH);
  else if (state == "YELLOW") digitalWrite(TRAFFIC_YELLOW, HIGH);
  else if (state == "RED") digitalWrite(TRAFFIC_RED, HIGH);
}

// ---------------- Sensor reading ----------------
float readVibration() {
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  // Magnitude of acceleration.
  // At rest this is approximately 9.81 m/s^2.
  // For the trained model, this is the value being supplied as vibration.
  float ax = accel.acceleration.x;
  float ay = accel.acceleration.y;
  float az = accel.acceleration.z;

  return sqrtf(ax * ax + ay * ay + az * az);
}

float readSound() {
  // KY-038 analog output. This is an ADC value, matching the
  // "sound" feature scale used by the current dataset.
  return (float)analogRead(SOUND_PIN);
}

float readCurrentRMS() {
  // Remove DC offset and calculate RMS current.
  // For a 5A ACS712, sensitivity is nominally 185 mV/A.
  double sumSq = 0.0;

  // Average voltage around the module's actual zero point.
  // Use the configured zero voltage as the baseline.
  for (int i = 0; i < ACS_SAMPLES; i++) {
    int raw = analogRead(ACS712_PIN);
    float voltage = adcToVoltage(raw);
    float centered = voltage - ACS_ZERO_VOLTAGE;
    sumSq += centered * centered;
    delayMicroseconds(250);
  }

  float rmsVoltage = sqrtf((float)(sumSq / ACS_SAMPLES));
  float amps = rmsVoltage / ACS_SENSITIVITY;

  if (amps < 0.05f) amps = 0.0f;
  return amps;
}

// ---------------- TinyML ------------------
bool initTinyML() {
  model = tflite::GetModel(model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR: model schema %d != supported %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  /*
    Your ESP32 core does not contain all_ops_resolver.h.
    We therefore use MicroMutableOpResolver and register only
    operators needed by this small Dense network.
  */
  static tflite::MicroMutableOpResolver<8> resolver;

  if (resolver.AddFullyConnected() != kTfLiteOk) return false;
  if (resolver.AddRelu() != kTfLiteOk) return false;
  if (resolver.AddSoftmax() != kTfLiteOk) return false;

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TinyML model initialized.");
  Serial.printf("Input type: %d, shape: ", input->type);
  for (int i = 0; i < input->dims->size; i++) {
    Serial.printf("%d ", input->dims->data[i]);
  }
  Serial.println();

  Serial.printf("Output type: %d, shape: ", output->type);
  for (int i = 0; i < output->dims->size; i++) {
    Serial.printf("%d ", output->dims->data[i]);
  }
  Serial.println();

  // Your tested model should expose FLOAT32 input/output.
  if (input->type != kTfLiteFloat32) {
    Serial.println("WARNING: This sketch expects FLOAT32 model input.");
  }

  return true;
}

void fillModelInput() {
  float features[5] = {
    temperature,
    humidity,
    vibration,
    currentRms,
    soundAdc
  };

  for (int i = 0; i < 5; i++) {
    float normalized =
      (features[i] - scaler_mean[i]) / scaler_scale[i];

    input->data.f[i] = normalized;
  }
}

bool runInference() {
  if (!interpreter || !input || !output) return false;

  if (!isfinite(temperature) ||
      !isfinite(humidity) ||
      !isfinite(vibration) ||
      !isfinite(currentRms) ||
      !isfinite(soundAdc)) {
    return false;
  }

  fillModelInput();

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: TinyML Invoke() failed.");
    return false;
  }

  float p0 = output->data.f[0];
  float p1 = output->data.f[1];
  float p2 = output->data.f[2];

  int best = 0;
  float bestP = p0;

  if (p1 > bestP) {
    best = 1;
    bestP = p1;
  }

  if (p2 > bestP) {
    best = 2;
    bestP = p2;
  }

  prediction = best;
  confidence = bestP;

  /*
    Health score:
      NORMAL  -> high
      WARNING -> medium
      FAULT   -> low

    This is a dashboard/actuator score, not a second ML model.
  */
  if (prediction == 0) {
    healthScore = 90.0f + 10.0f * confidence;
    if (healthScore > 100) healthScore = 100;
  } else if (prediction == 1) {
    healthScore = 55.0f + 25.0f * (1.0f - confidence);
  } else {
    healthScore = 20.0f + 25.0f * (1.0f - confidence);
  }

  // ---------------- Actuator policy ----------------
  if (prediction == 0) {
    setRelay(false);
    setBuzzer(false);
    setTraffic("GREEN");
  }
  else if (prediction == 1) {
    // WARNING: keep relay off and alert operator.
    setRelay(false);
    setBuzzer(true);
    setTraffic("YELLOW");
  }
  else {
    // FAULT: disconnect the fan and alarm.
    setRelay(true);
    setBuzzer(true);
    setTraffic("RED");
  }

  return true;
}

// ---------------- Serial monitor ----------------
void printStatus() {
  const char* labels[] = {"NORMAL", "WARNING", "FAULT"};

  Serial.println("----------------------------------------------");
  Serial.printf("Temperature: %.2f °C\n", temperature);
  Serial.printf("Humidity:    %.2f %%\n", humidity);
  Serial.printf("Vibration:   %.2f m/s²\n", vibration);
  Serial.printf("Current:     %.2f A RMS\n", currentRms);
  Serial.printf("Sound:       %.2f ADC\n", soundAdc);
  Serial.println("----------------------------------------------");
  Serial.printf("Prediction:  %s\n", labels[prediction]);
  Serial.printf("Confidence:  %.2f %%\n", confidence * 100.0f);
  Serial.printf("Health:      %.0f / 100\n", healthScore);
  Serial.printf("Relay:       %s\n", relayState ? "ON" : "OFF");
  Serial.printf("Buzzer:      %s\n", buzzerState ? "ON" : "OFF");
  Serial.printf("Traffic:     %s\n", trafficState.c_str());
}

// ---------------- HTTP API ----------------
String jsonFloat(float x, int digits = 2) {
  if (!isfinite(x)) return "null";
  return String(x, digits);
}

void handleStatus() {
  String json = "{";
  json += "\"temperature\":" + jsonFloat(temperature) + ",";
  json += "\"humidity\":" + jsonFloat(humidity) + ",";
  json += "\"vibration\":" + jsonFloat(vibration) + ",";
  json += "\"current\":" + jsonFloat(currentRms) + ",";
  json += "\"sound\":" + jsonFloat(soundAdc) + ",";
  json += "\"prediction\":" + String(prediction) + ",";
  json += "\"confidence\":" + jsonFloat(confidence, 4) + ",";
  json += "\"health\":" + jsonFloat(healthScore, 1) + ",";
  json += "\"relay\":" + String(relayState ? "true" : "false") + ",";
  json += "\"buzzer\":" + String(buzzerState ? "true" : "false") + ",";
  json += "\"traffic\":\"" + trafficState + "\",";
  json += "\"uptime\":" + String((millis() - bootMs) / 1000UL);
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void handleRoot() {
  String page =
    "<!doctype html><html><head><meta name='viewport' content='width=device-width'>"
    "<title>TinyML Fan Guardian</title></head><body>"
    "<h2>TinyML Fan Guardian ESP32</h2>"
    "<p>API: <a href='/api/status'>/api/status</a></p>"
    "<pre id='x'>Loading...</pre>"
    "<script>setInterval(()=>fetch('/api/status').then(r=>r.json()).then(x=>"
    "document.getElementById('x').textContent=JSON.stringify(x,null,2)),1000)</script>"
    "</body></html>";

  server.send(200, "text/html", page);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// ---------------- Setup -------------------
void setup() {
  Serial.begin(115200);
  delay(1000);

  bootMs = millis();

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(TRAFFIC_GREEN, OUTPUT);
  pinMode(TRAFFIC_YELLOW, OUTPUT);
  pinMode(TRAFFIC_RED, OUTPUT);

  setRelay(false);
  setBuzzer(false);
  setTraffic("GREEN");

  analogReadResolution(12);

  // ADC attenuation for approximately 0-3.3 V sensor outputs.
  analogSetPinAttenuation(ACS712_PIN, ADC_11db);
  analogSetPinAttenuation(SOUND_PIN, ADC_11db);

  Serial.println();
  Serial.println("==============================================");
  Serial.println(" TINYML FAN GUARDIAN - ESP32");
  Serial.println("==============================================");

  dht.begin();

  Wire.begin(I2C_SDA, I2C_SCL);

  if (!mpu.begin()) {
    Serial.println("ERROR: MPU6050 not found.");
    while (true) {
      setTraffic("RED");
      setBuzzer(true);
      delay(500);
      setBuzzer(false);
      delay(500);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("DHT22: OK");
  Serial.println("MPU6050: OK");
  Serial.println("ACS712: configured");
  Serial.println("KY-038: configured");

  if (!initTinyML()) {
    Serial.println("ERROR: TinyML initialization failed.");
    setTraffic("RED");
    setBuzzer(true);
    while (true) delay(1000);
  }

  // ---------------- Wi-Fi ----------------
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  unsigned long wifiStart = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - wifiStart < 20000) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi connected. ESP32 IP: ");
    Serial.println(WiFi.localIP());

    server.on("/", HTTP_GET, handleRoot);
    server.on("/api/status", HTTP_GET, handleStatus);
    server.onNotFound(handleNotFound);
    server.begin();

    Serial.println("HTTP server started.");
    Serial.print("Dashboard API: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/api/status");
  } else {
    Serial.println("Wi-Fi unavailable. TinyML still runs locally.");
  }

  Serial.println("System ready.");
}

// ---------------- Loop -------------------
void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    server.handleClient();
  }

  if (millis() - lastSampleMs >= 1000) {
    lastSampleMs = millis();

    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (isfinite(t)) temperature = t;
    if (isfinite(h)) humidity = h;

    vibration = readVibration();
    currentRms = readCurrentRMS();
    soundAdc = readSound();

    if (millis() - lastInferenceMs >= 1000) {
      lastInferenceMs = millis();

      if (runInference()) {
        printStatus();
      }
    }
  }
}
