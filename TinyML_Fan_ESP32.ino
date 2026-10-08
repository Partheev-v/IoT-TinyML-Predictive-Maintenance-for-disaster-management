/*
   ============================================================
   TinyML FAN PREDICTIVE MAINTENANCE - ESP32
   ============================================================

   Sensors:
     DHT22       -> Temperature + Humidity
     MPU6050     -> Vibration
     ACS712 5A   -> Current
     KY-038      -> Sound

   Outputs:
     Relay       -> GPIO 26
     Buzzer      -> GPIO 27
     Traffic RED -> GPIO 25
     Traffic YELLOW -> GPIO 33
     Traffic GREEN  -> GPIO 32

   TinyML:
     INT8 TFLite model
     FLOAT32 input/output interface

   Feature order:
     0 = temperature
     1 = humidity
     2 = vibration
     3 = current
     4 = sound

   Classes:
     0 = NORMAL
     1 = WARNING
     2 = FAULT
*/

#include <Arduino.h>
#include <Wire.h>
#include <DHT.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ============================================================
// TinyML
// ============================================================

#include "model_data.h"
#include "normalization_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"


// ============================================================
// PIN DEFINITIONS
// ============================================================

// DHT22
#define DHT_PIN 4
#define DHT_TYPE DHT22

// I2C
#define I2C_SDA 21
#define I2C_SCL 22

// Analog sensors
#define ACS712_PIN 34
#define SOUND_PIN 35

// Outputs
#define RELAY_PIN 26
#define BUZZER_PIN 27

// Traffic light
#define RED_PIN 25
#define YELLOW_PIN 33
#define GREEN_PIN 32


// ============================================================
// OUTPUT LOGIC
// ============================================================

// Most relay modules are ACTIVE LOW.
// If your relay works opposite, change this to false.
#define RELAY_ACTIVE_LOW true

// Most buzzer modules are active HIGH.
#define BUZZER_ACTIVE_HIGH true

// Most 3-color LED modules are active HIGH.
// Change to false if your module is active LOW.
#define TRAFFIC_ACTIVE_HIGH true


// ============================================================
// SENSOR OBJECTS
// ============================================================

DHT dht(DHT_PIN, DHT_TYPE);

Adafruit_MPU6050 mpu;


// ============================================================
// ACS712 SETTINGS
// ============================================================

// ACS712 5A version:
// typical sensitivity = 185 mV/A
const float ACS712_SENSITIVITY = 0.185f;

// Nominal zero-current output.
// Actual zero point is measured during startup.
float acsZeroVoltage = 2.500f;

// IMPORTANT:
// If you use a resistor divider between ACS712 OUT and GPIO34,
// set this to the ratio.
//
// Example:
// 10k top + 20k bottom:
// ESP32 sees 20/(10+20) = 0.6667
//
// If no voltage divider is used:
// 1.0
//
// IMPORTANT: GPIO34 must never receive an unsafe voltage.
const float ACS_VOLTAGE_DIVIDER_RATIO = 1.0f;


// ============================================================
// ADC SETTINGS
// ============================================================

const int ADC_SAMPLES = 400;


// ============================================================
// MODEL SETTINGS
// ============================================================

const int NUM_FEATURES = 5;
const int NUM_CLASSES = 3;

// Tensor arena.
// 24 KB gives the tiny model plenty of working RAM.
constexpr int TENSOR_ARENA_SIZE = 24 * 1024;

alignas(16)
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// Resolver with only the operations used by our Dense network.
//
// Keras model:
// Dense(16, relu)
// Dense(12, relu)
// Dense(3, softmax)
//
// TFLite operations required:
// FULLY_CONNECTED
// RELU
// SOFTMAX

tflite::MicroMutableOpResolver<3> resolver;


// ============================================================
// TIMING
// ============================================================

unsigned long lastPrediction = 0;

const unsigned long PREDICTION_INTERVAL = 2000;


// ============================================================
// FAULT CONFIRMATION
// ============================================================

// Prevent one noisy prediction from immediately switching off
// the fan.
//
// 3 consecutive FAULT predictions are required.
const int FAULT_CONFIRMATIONS_REQUIRED = 3;

int consecutiveFaults = 0;


// ============================================================
// CLASS NAMES
// ============================================================

const char* className(int classIndex)
{
  switch (classIndex)
  {
    case 0:
      return "NORMAL";

    case 1:
      return "WARNING";

    case 2:
      return "FAULT";

    default:
      return "UNKNOWN";
  }
}


// ============================================================
// OUTPUT HELPERS
// ============================================================

void setRelay(bool on)
{
  if (RELAY_ACTIVE_LOW)
  {
    digitalWrite(RELAY_PIN, on ? LOW : HIGH);
  }
  else
  {
    digitalWrite(RELAY_PIN, on ? HIGH : LOW);
  }
}


void setBuzzer(bool on)
{
  if (BUZZER_ACTIVE_HIGH)
  {
    digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
  }
  else
  {
    digitalWrite(BUZZER_PIN, on ? LOW : HIGH);
  }
}


void trafficOff()
{
  if (TRAFFIC_ACTIVE_HIGH)
  {
    digitalWrite(RED_PIN, LOW);
    digitalWrite(YELLOW_PIN, LOW);
    digitalWrite(GREEN_PIN, LOW);
  }
  else
  {
    digitalWrite(RED_PIN, HIGH);
    digitalWrite(YELLOW_PIN, HIGH);
    digitalWrite(GREEN_PIN, HIGH);
  }
}


void trafficNormal()
{
  trafficOff();

  if (TRAFFIC_ACTIVE_HIGH)
  {
    digitalWrite(GREEN_PIN, HIGH);
  }
  else
  {
    digitalWrite(GREEN_PIN, LOW);
  }
}


void trafficWarning()
{
  trafficOff();

  if (TRAFFIC_ACTIVE_HIGH)
  {
    digitalWrite(YELLOW_PIN, HIGH);
  }
  else
  {
    digitalWrite(YELLOW_PIN, LOW);
  }
}


void trafficFault()
{
  trafficOff();

  if (TRAFFIC_ACTIVE_HIGH)
  {
    digitalWrite(RED_PIN, HIGH);
  }
  else
  {
    digitalWrite(RED_PIN, LOW);
  }
}


// ============================================================
// ACS712 ZERO CALIBRATION
// ============================================================

float readACSVoltage()
{
  uint32_t voltageSum = 0;

  const int samples = 50;

  for (int i = 0; i < samples; i++)
  {
    voltageSum += analogReadMilliVolts(ACS712_PIN);

    delayMicroseconds(500);
  }

  float voltage = (voltageSum / (float)samples) / 1000.0f;

  // If a divider is used, reconstruct original ACS712 voltage.
  voltage = voltage / ACS_VOLTAGE_DIVIDER_RATIO;

  return voltage;
}


void calibrateACS712()
{
  Serial.println();
  Serial.println("==============================================");
  Serial.println(" ACS712 ZERO CALIBRATION");
  Serial.println("==============================================");

  Serial.println("Make sure NO current is flowing through ACS712.");
  Serial.println("Calibrating...");

  delay(2000);

  float sum = 0;

  const int samples = 100;

  for (int i = 0; i < samples; i++)
  {
    sum += readACSVoltage();

    delay(5);
  }

  acsZeroVoltage = sum / samples;

  Serial.print("ACS712 zero voltage: ");
  Serial.print(acsZeroVoltage, 4);
  Serial.println(" V");

  Serial.println("ACS712 calibration complete.");
}


// ============================================================
// ACS712 CURRENT RMS
// ============================================================

float readACSCurrentRMS()
{
  /*
     For AC current:

       Vout = Vzero + I * sensitivity

     Therefore:

       I = (Vout - Vzero) / sensitivity

     We sample many points and calculate RMS current.
  */

  double sumSquares = 0.0;

  for (int i = 0; i < ADC_SAMPLES; i++)
  {
    float voltage = analogReadMilliVolts(ACS712_PIN) / 1000.0f;

    // Restore original ACS712 output voltage if divider exists.
    voltage = voltage / ACS_VOLTAGE_DIVIDER_RATIO;

    float current = (voltage - acsZeroVoltage)
                    / ACS712_SENSITIVITY;

    sumSquares += current * current;

    delayMicroseconds(200);
  }

  float rms = sqrt(sumSquares / ADC_SAMPLES);

  // Small noise suppression.
  if (rms < 0.05f)
  {
    rms = 0.0f;
  }

  return rms;
}


// ============================================================
// KY-038 SOUND
// ============================================================

float readSound()
{
  uint32_t sum = 0;

  const int samples = 50;

  for (int i = 0; i < samples; i++)
  {
    sum += analogRead(SOUND_PIN);

    delayMicroseconds(200);
  }

  return sum / (float)samples;
}


// ============================================================
// MPU6050 VIBRATION
// ============================================================

float readVibration()
{
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;

  mpu.getEvent(&accel, &gyro, &temp);

  float ax = accel.acceleration.x;
  float ay = accel.acceleration.y;
  float az = accel.acceleration.z;

  /*
     Total acceleration magnitude.

     At rest, this is approximately 9.81 m/s²
     because of gravity.
  */

  float magnitude =
      sqrt(
        ax * ax +
        ay * ay +
        az * az
      );

  /*
     IMPORTANT:

     The training data has normal vibration values around
     0.5 rather than 9.8.

     Therefore remove the gravity component.

       vibration = |magnitude - 9.80665|
  */

  float vibration = fabs(magnitude - 9.80665f);

  return vibration;
}


// ============================================================
// DHT22
// ============================================================

bool readDHT(float &temperature, float &humidity)
{
  temperature = dht.readTemperature();
  humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity))
  {
    Serial.println("ERROR: DHT22 reading failed.");

    return false;
  }

  return true;
}


// ============================================================
// NORMALIZATION
// ============================================================

float normalizeFeature(float value, int index)
{
  return (value - scaler_mean[index])
         / scaler_scale[index];
}


// ============================================================
// PRINT FEATURES
// ============================================================

void printRawFeatures(
  float temperature,
  float humidity,
  float vibration,
  float current,
  float sound
)
{
  Serial.println();
  Serial.println("----------------------------------------------");

  Serial.print("Temperature: ");
  Serial.print(temperature, 2);
  Serial.println(" °C");

  Serial.print("Humidity:    ");
  Serial.print(humidity, 2);
  Serial.println(" %");

  Serial.print("Vibration:   ");
  Serial.print(vibration, 3);
  Serial.println(" m/s²");

  Serial.print("Current:     ");
  Serial.print(current, 3);
  Serial.println(" A RMS");

  Serial.print("Sound:       ");
  Serial.print(sound, 2);
  Serial.println(" ADC");

  Serial.println("----------------------------------------------");
}


// ============================================================
// PRINT NORMALIZED FEATURES
// ============================================================

void printNormalizedFeatures(float features[])
{
  Serial.println("Normalized features:");

  for (int i = 0; i < NUM_FEATURES; i++)
  {
    Serial.print("  [");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(features[i], 5);
  }
}


// ============================================================
// INITIALIZE TFLITE
// ============================================================

bool initializeModel()
{
  Serial.println();
  Serial.println("==============================================");
  Serial.println(" INITIALIZING TFLITE MICRO");
  Serial.println("==============================================");

  // Get model from generated model_data.h
  model = tflite::GetModel(model_data);

  if (model == nullptr)
  {
    Serial.println("ERROR: Model pointer is NULL.");

    return false;
  }

  Serial.print("Model size: ");
  Serial.print(model_data_len);
  Serial.println(" bytes");

  Serial.print("Model schema version: ");
  Serial.println(model->version());

  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("ERROR: TFLite schema version mismatch.");

    return false;
  }


  // ----------------------------------------------------------
  // Register required operators
  // ----------------------------------------------------------

  if (resolver.AddFullyConnected() != kTfLiteOk)
  {
    Serial.println("ERROR: AddFullyConnected failed.");

    return false;
  }

  if (resolver.AddRelu() != kTfLiteOk)
  {
    Serial.println("ERROR: AddRelu failed.");

    return false;
  }

  if (resolver.AddSoftmax() != kTfLiteOk)
  {
    Serial.println("ERROR: AddSoftmax failed.");

    return false;
  }


  // ----------------------------------------------------------
  // Create interpreter
  // ----------------------------------------------------------

  static tflite::MicroInterpreter staticInterpreter(
    model,
    resolver,
    tensor_arena,
    TENSOR_ARENA_SIZE
  );

  interpreter = &staticInterpreter;


  // ----------------------------------------------------------
  // Allocate tensors
  // ----------------------------------------------------------

  Serial.println("Allocating tensors...");

  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk)
  {
    Serial.println("ERROR: AllocateTensors() failed.");

    return false;
  }

  Serial.println("Tensor allocation successful.");


  // ----------------------------------------------------------
  // Get input/output
  // ----------------------------------------------------------

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr)
  {
    Serial.println("ERROR: Input/output tensor unavailable.");

    return false;
  }


  // ----------------------------------------------------------
  // Print tensor information
  // ----------------------------------------------------------

  Serial.println();
  Serial.println("TFLite input:");

  Serial.print("Type: ");

  if (input->type == kTfLiteFloat32)
  {
    Serial.println("FLOAT32");
  }
  else
  {
    Serial.println(input->type);
  }

  Serial.print("Dimensions: ");

  for (int i = 0; i < input->dims->size; i++)
  {
    Serial.print(input->dims->data[i]);

    if (i < input->dims->size - 1)
    {
      Serial.print(" x ");
    }
  }

  Serial.println();


  Serial.println();
  Serial.println("TFLite output:");

  Serial.print("Type: ");

  if (output->type == kTfLiteFloat32)
  {
    Serial.println("FLOAT32");
  }
  else
  {
    Serial.println(output->type);
  }

  Serial.print("Dimensions: ");

  for (int i = 0; i < output->dims->size; i++)
  {
    Serial.print(output->dims->data[i]);

    if (i < output->dims->size - 1)
    {
      Serial.print(" x ");
    }
  }

  Serial.println();

  return true;
}


// ============================================================
// RUN MODEL
// ============================================================

int runPrediction(float temperature,
                  float humidity,
                  float vibration,
                  float current,
                  float sound)
{
  /*
     ----------------------------------------------------------
     RAW SENSOR VALUES
     ----------------------------------------------------------

     These MUST be in the same units as training data.
  */

  float rawFeatures[NUM_FEATURES];

  rawFeatures[0] = temperature;
  rawFeatures[1] = humidity;
  rawFeatures[2] = vibration;
  rawFeatures[3] = current;
  rawFeatures[4] = sound;


  /*
     ----------------------------------------------------------
     NORMALIZATION
     ----------------------------------------------------------

     x_normalized = (x - training_mean) / training_scale
  */

  float normalized[NUM_FEATURES];

  for (int i = 0; i < NUM_FEATURES; i++)
  {
    normalized[i] =
      normalizeFeature(rawFeatures[i], i);
  }


  printNormalizedFeatures(normalized);


  /*
     ----------------------------------------------------------
     MODEL INPUT
     ----------------------------------------------------------

     Your INT8 TFLite model has FLOAT32 input.

     Therefore DO NOT manually convert these values to int8.
  */

  if (input->type != kTfLiteFloat32)
  {
    Serial.println("ERROR: Model input is not FLOAT32.");

    return -1;
  }

  for (int i = 0; i < NUM_FEATURES; i++)
  {
    input->data.f[i] = normalized[i];
  }


  /*
     ----------------------------------------------------------
     INFERENCE
     ----------------------------------------------------------
  */

  TfLiteStatus status = interpreter->Invoke();

  if (status != kTfLiteOk)
  {
    Serial.println("ERROR: TFLite Invoke() failed.");

    return -1;
  }


  /*
     ----------------------------------------------------------
     OUTPUT
     ----------------------------------------------------------
  */

  float normalProbability = output->data.f[0];
  float warningProbability = output->data.f[1];
  float faultProbability = output->data.f[2];

  Serial.println();
  Serial.println("Model probabilities:");

  Serial.print("NORMAL : ");
  Serial.print(normalProbability * 100.0f, 2);
  Serial.println(" %");

  Serial.print("WARNING: ");
  Serial.print(warningProbability * 100.0f, 2);
  Serial.println(" %");

  Serial.print("FAULT  : ");
  Serial.print(faultProbability * 100.0f, 2);
  Serial.println(" %");


  /*
     Find highest probability.
  */

  int predictedClass = 0;

  float highest = normalProbability;

  if (warningProbability > highest)
  {
    highest = warningProbability;

    predictedClass = 1;
  }

  if (faultProbability > highest)
  {
    highest = faultProbability;

    predictedClass = 2;
  }


  Serial.println();
  Serial.println("==============================================");
  Serial.println(" TINYML PREDICTION");
  Serial.println("==============================================");

  Serial.print("Prediction: ");
  Serial.println(className(predictedClass));

  Serial.print("Confidence: ");
  Serial.print(highest * 100.0f, 2);
  Serial.println(" %");


  return predictedClass;
}


// ============================================================
// APPLY SYSTEM RESPONSE
// ============================================================

void applyPrediction(int predictedClass)
{
  if (predictedClass == 0)
  {
    // NORMAL
    consecutiveFaults = 0;

    Serial.println();
    Serial.println("SYSTEM STATUS: NORMAL");

    // Fan allowed to run
    setRelay(true);

    // No alarm
    setBuzzer(false);

    // Green
    trafficNormal();
  }


  else if (predictedClass == 1)
  {
    // WARNING
    consecutiveFaults = 0;

    Serial.println();
    Serial.println("SYSTEM STATUS: WARNING");

    // Keep fan running
    setRelay(true);

    // Warning does not activate buzzer
    setBuzzer(false);

    // Yellow
    trafficWarning();
  }


  else if (predictedClass == 2)
  {
    // FAULT
    consecutiveFaults++;

    Serial.println();
    Serial.println("SYSTEM STATUS: FAULT");

    Serial.print("Fault confirmation: ");
    Serial.print(consecutiveFaults);
    Serial.print("/");
    Serial.println(FAULT_CONFIRMATIONS_REQUIRED);


    // Red warning immediately
    trafficFault();


    /*
       Buzzer on for FAULT.
    */
    setBuzzer(true);


    /*
       Do not shut down immediately from one prediction.
       Require consecutive fault predictions.
    */

    if (consecutiveFaults >= FAULT_CONFIRMATIONS_REQUIRED)
    {
      Serial.println("!!! CONFIRMED FAULT !!!");
      Serial.println("Turning fan OFF.");

      setRelay(false);
    }
    else
    {
      Serial.println("Fault not yet confirmed.");
      Serial.println("Fan remains ON temporarily.");

      setRelay(true);
    }
  }
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("==============================================");
  Serial.println(" TINYML FAN PREDICTIVE MAINTENANCE");
  Serial.println(" ESP32 + SENSOR + INT8 TFLITE");
  Serial.println("==============================================");


  // ----------------------------------------------------------
  // Outputs
  // ----------------------------------------------------------

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);


  // Safe startup state
  setRelay(false);
  setBuzzer(false);
  trafficOff();


  // ----------------------------------------------------------
  // ADC
  // ----------------------------------------------------------

  analogReadResolution(12);

  analogSetPinAttenuation(
    ACS712_PIN,
    ADC_11db
  );

  analogSetPinAttenuation(
    SOUND_PIN,
    ADC_11db
  );


  // ----------------------------------------------------------
  // DHT22
  // ----------------------------------------------------------

  Serial.println();
  Serial.println("Initializing DHT22...");

  dht.begin();

  delay(1500);


  // ----------------------------------------------------------
  // MPU6050
  // ----------------------------------------------------------

  Serial.println("Initializing MPU6050...");

  Wire.begin(
    I2C_SDA,
    I2C_SCL
  );

  if (!mpu.begin())
  {
    Serial.println("ERROR: MPU6050 not found.");

    while (true)
    {
      setBuzzer(true);

      delay(300);

      setBuzzer(false);

      delay(300);
    }
  }

  Serial.println("MPU6050 OK.");


  // Configure MPU6050
  mpu.setAccelerometerRange(
    MPU6050_RANGE_8_G
  );

  mpu.setGyroRange(
    MPU6050_RANGE_500_DEG
  );

  mpu.setFilterBandwidth(
    MPU6050_BAND_21_HZ
  );


  // ----------------------------------------------------------
  // ACS712
  // ----------------------------------------------------------

  calibrateACS712();


  // ----------------------------------------------------------
  // TFLite
  // ----------------------------------------------------------

  if (!initializeModel())
  {
    Serial.println();
    Serial.println("==============================================");
    Serial.println(" FATAL: TFLITE INITIALIZATION FAILED");
    Serial.println("==============================================");

    while (true)
    {
      trafficFault();

      setBuzzer(true);

      delay(500);

      setBuzzer(false);

      delay(500);
    }
  }


  // ----------------------------------------------------------
  // Startup complete
  // ----------------------------------------------------------

  Serial.println();
  Serial.println("==============================================");
  Serial.println(" SYSTEM READY");
  Serial.println("==============================================");

  Serial.println("Features:");
  Serial.println("0 = Temperature");
  Serial.println("1 = Humidity");
  Serial.println("2 = Vibration");
  Serial.println("3 = Current");
  Serial.println("4 = Sound");

  Serial.println();
  Serial.println("Classes:");
  Serial.println("0 = NORMAL");
  Serial.println("1 = WARNING");
  Serial.println("2 = FAULT");

  Serial.println();
  Serial.println("Starting monitoring...");


  // Green startup
  trafficNormal();

  lastPrediction = millis();
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  if (millis() - lastPrediction < PREDICTION_INTERVAL)
  {
    delay(10);

    return;
  }

  lastPrediction = millis();


  // ==========================================================
  // READ DHT22
  // ==========================================================

  float temperature;
  float humidity;

  if (!readDHT(temperature, humidity))
  {
    Serial.println("Skipping prediction because DHT22 failed.");

    return;
  }


  // ==========================================================
  // READ MPU6050
  // ==========================================================

  float vibration = readVibration();


  // ==========================================================
  // READ ACS712
  // ==========================================================

  float current = readACSCurrentRMS();


  // ==========================================================
  // READ KY-038
  // ==========================================================

  float sound = readSound();


  // ==========================================================
  // DISPLAY RAW VALUES
  // ==========================================================

  printRawFeatures(
    temperature,
    humidity,
    vibration,
    current,
    sound
  );


  // ==========================================================
  // RUN TINYML
  // ==========================================================

  int prediction =
    runPrediction(
      temperature,
      humidity,
      vibration,
      current,
      sound
    );


  if (prediction < 0)
  {
    Serial.println("Prediction failed.");

    return;
  }


  // ==========================================================
  // CONTROL OUTPUTS
  // ==========================================================

  applyPrediction(prediction);


  Serial.println();
  Serial.println("==============================================");
  Serial.println();
}
