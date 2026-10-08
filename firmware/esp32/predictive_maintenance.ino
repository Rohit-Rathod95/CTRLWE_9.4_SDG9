/*
 * ============================================================================
 * Industrial Predictive Maintenance Platform - Edge Sensor Acquisition Module
 * ============================================================================
 * 
 * Target Board: ESP32 Dev Module (NodeMCU ESP32, ESP-WROOM-32, etc.)
 * Framework:    Arduino IDE
 * 
 * Purpose:
 *   Basic sensor data acquisition layer for industrial machinery monitoring.
 *   Acquires telemetry from physical sensors:
 *     - MPU6050 (I2C): Tri-axial acceleration & vibration magnitude
 *     - DS18B20 (1-Wire): Machine surface / bearing temperature
 *     - ACS712 (ADC): Motor current draw / electrical load
 *     - Hall-Effect (Interrupt): Shaft rotational speed (RPM)
 * 
 * Output:
 *   Prints human-readable diagnostics and standardized single-line JSON
 *   telemetry packets to the Serial Monitor at a fixed sampling interval.
 * 
 * Note:
 *   This is strictly the sensor acquisition layer. Edge ML inference, Wi-Fi,
 *   MQTT, and backend telemetry dispatch are scheduled for subsequent stages.
 * ============================================================================
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================================
// 1. HARDWARE PIN DEFINITIONS (Configurable)
// ============================================================================
// I2C Pins for MPU6050
#define PIN_I2C_SDA         21    // ESP32 default I2C Data line
#define PIN_I2C_SCL         22    // ESP32 default I2C Clock line

// 1-Wire Pin for DS18B20 Temperature Sensor
// Note: Requires a 4.7k Ohm pull-up resistor between Data and 3.3V
#define PIN_DS18B20         4     

// Analog Pin for ACS712 Current Sensor
// Note: GPIO 34 is an ADC1 input-only pin on the ESP32 (safe from pull-up conflicts)
#define PIN_ACS712          34    

// Digital Interrupt Pin for Hall-Effect Sensor
// Note: GPIO 18 supports hardware interrupts for pulse detection
#define PIN_HALL_SENSOR     18    

// ============================================================================
// 2. CONFIGURATION PARAMETERS
// ============================================================================
// Identification & Acquisition Rate
const char* DEVICE_ID                 = "MACHINE_01";
const unsigned long SAMPLING_INTERVAL = 1000; // Sensor sampling period in milliseconds (1 Hz)

// Hall-Effect Sensor Settings
const int PULSES_PER_REV              = 1;    // Number of magnet pulses per full shaft revolution

// ACS712 Current Sensor Calibration Settings
// Common ACS712 sensor sensitivities:
//   - 5A Module:  185 mV/A (0.185 V/A)
//   - 20A Module: 100 mV/A (0.100 V/A)
//   - 30A Module:  66 mV/A (0.066 V/A)
const float ACS_SENSITIVITY           = 0.185; // Sensitivity in Volts/Amp (Change according to module variant)
const float ADC_REF_VOLTAGE           = 3.3;   // ESP32 ADC Reference Voltage (typically 3.3V)
const int   ADC_RESOLUTION            = 4095;  // ESP32 12-bit ADC (range: 0 - 4095)

// ACS712 quiescent zero-current voltage offset:
// When using a 5V sensor with a voltage divider or 3.3V-scaled circuit, calibrate
// this value by observing the voltage output when no current is flowing.
// Theoretical midpoint on a 3.3V scale is ~1.65V (or ~2.5V without divider if scaled).
const float ACS_ZERO_OFFSET_V         = 1.65;  // Baseline voltage at 0 Amperes load

// ============================================================================
// 3. SENSOR DRIVERS & GLOBAL STATE
// ============================================================================
// MPU6050 IMU instance
Adafruit_MPU6050 mpu;
bool mpuInitialized = false;

// DS18B20 1-Wire Temperature instance
OneWire oneWire(PIN_DS18B20);
DallasTemperature tempSensor(&oneWire);
bool tempSensorInitialized = false;

// Hall-Effect Sensor pulse accumulator (must be volatile for ISR modification)
volatile unsigned long hallPulseCount = 0;

// Non-blocking timer tracking
unsigned long previousMillis = 0;

// ============================================================================
// 4. INTERRUPT SERVICE ROUTINE (ISR)
// ============================================================================
// Triggered on falling edge when the Hall sensor detects the magnet pass
void IRAM_ATTR onHallPulse() {
  hallPulseCount++;
}

// ============================================================================
// 5. HELPER FUNCTIONS
// ============================================================================

/**
 * Reads the ACS712 analog pin, averages multiple samples to attenuate ADC noise,
 * and converts the reading into current (Amperes).
 */
float readCurrent() {
  const int NUM_SAMPLES = 20;
  int rawSum = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    rawSum += analogRead(PIN_ACS712);
    delayMicroseconds(50); // Minimal settling pause
  }

  float avgRaw = (float)rawSum / (float)NUM_SAMPLES;
  float measuredVoltage = (avgRaw / (float)ADC_RESOLUTION) * ADC_REF_VOLTAGE;
  
  // Calculate current: I = (V_measured - V_zero) / Sensitivity
  float currentAmps = (measuredVoltage - ACS_ZERO_OFFSET_V) / ACS_SENSITIVITY;

  // Filter out microscopic ADC noise around true zero
  if (abs(currentAmps) < 0.05) {
    currentAmps = 0.0;
  }

  return currentAmps;
}

// ============================================================================
// 6. SETUP ROUTINE
// ============================================================================
void setup() {
  // 1. Initialize Serial Console
  Serial.begin(115200);
  delay(1000); // Short settle time for USB Serial monitor connection

  Serial.println("\n==================================================");
  Serial.println("  ESP32 Predictive Maintenance Sensor Acquisition ");
  Serial.println("==================================================");
  Serial.print("Device ID: ");
  Serial.println(DEVICE_ID);

  // 2. Initialize I2C Bus & MPU6050 Vibration Sensor
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  if (!mpu.begin()) {
    Serial.println("[ERROR] MPU6050 initialization failed! Check wiring (SDA/SCL).");
    mpuInitialized = false;
  } else {
    Serial.println("[OK]    MPU6050 initialized successfully.");
    mpuInitialized = true;
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // 3. Initialize DS18B20 1-Wire Temperature Sensor
  tempSensor.begin();
  if (tempSensor.getDeviceCount() == 0) {
    Serial.println("[ERROR] DS18B20 initialization failed! No sensors found on 1-Wire bus.");
    tempSensorInitialized = false;
  } else {
    Serial.print("[OK]    DS18B20 initialized successfully. Found devices: ");
    Serial.println(tempSensor.getDeviceCount());
    tempSensorInitialized = true;
  }

  // 4. Configure ACS712 Current Sensor ADC
  pinMode(PIN_ACS712, INPUT);
  analogReadResolution(12); // Ensure 12-bit resolution (0 - 4095)
  Serial.println("[OK]    ACS712 ADC channel configured on GPIO 34.");

  // 5. Configure Hall-Effect Sensor Interrupt
  pinMode(PIN_HALL_SENSOR, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_HALL_SENSOR), onHallPulse, FALLING);
  Serial.println("[OK]    Hall-Effect interrupt attached on GPIO 18.");

  Serial.println("--------------------------------------------------");
  Serial.println("Telemetry streaming started...\n");
}

// ============================================================================
// 7. MAIN EXECUTION LOOP
// ============================================================================
void loop() {
  unsigned long currentMillis = millis();

  // Execute sensor acquisition at non-blocking fixed interval
  if (currentMillis - previousMillis >= SAMPLING_INTERVAL) {
    unsigned long intervalDuration = currentMillis - previousMillis;
    previousMillis = currentMillis;

    // --- A. Read Hall Sensor & Calculate RPM ---
    // Protect shared pulse count variable during read & reset
    noInterrupts();
    unsigned long pulses = hallPulseCount;
    hallPulseCount = 0;
    interrupts();

    // RPM Formula: (pulses / pulses_per_rev) * (60,000 ms / elapsed_time_ms)
    float rpm = ((float)pulses / (float)PULSES_PER_REV) * (60000.0 / (float)intervalDuration);

    // --- B. Read MPU6050 Vibration / Acceleration ---
    float vibX = 0.0;
    float vibY = 0.0;
    float vibZ = 0.0;
    float vibMagnitude = 0.0;

    if (mpuInitialized) {
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      vibX = a.acceleration.x;
      vibY = a.acceleration.y;
      vibZ = a.acceleration.z;
      // Vibration magnitude: Euclidean norm sqrt(x^2 + y^2 + z^2)
      vibMagnitude = sqrt((vibX * vibX) + (vibY * vibY) + (vibZ * vibZ));
    }

    // --- C. Read DS18B20 Machine Temperature ---
    float temperature = 0.0;
    if (tempSensorInitialized) {
      tempSensor.requestTemperatures();
      float tempReading = tempSensor.getTempCByIndex(0);
      if (tempReading == DEVICE_DISCONNECTED_C) {
        Serial.println("[WARNING] DS18B20 sensor disconnected during read.");
      } else {
        temperature = tempReading;
      }
    }

    // --- D. Read ACS712 Current ---
    float current = readCurrent();

    // --- E. Human-Readable Serial Output ---
    Serial.println("--------------------------------------------------");
    Serial.print("Temperature: ");
    Serial.print(temperature, 1);
    Serial.println(" C");

    Serial.print("Acceleration: X=");
    Serial.print(vibX, 2);
    Serial.print(" Y=");
    Serial.print(vibY, 2);
    Serial.print(" Z=");
    Serial.println(vibZ, 2);

    Serial.print("Vibration Magnitude: ");
    Serial.println(vibMagnitude, 2);

    Serial.print("Current: ");
    Serial.print(current, 2);
    Serial.println(" A");

    Serial.print("RPM: ");
    Serial.println((int)rpm);

    // --- F. Standardized Single-Line JSON Telemetry Packet ---
    Serial.print("{\"device_id\":\"");
    Serial.print(DEVICE_ID);
    Serial.print("\",\"temperature\":");
    Serial.print(temperature, 1);
    Serial.print(",\"vibration_x\":");
    Serial.print(vibX, 2);
    Serial.print(",\"vibration_y\":");
    Serial.print(vibY, 2);
    Serial.print(",\"vibration_z\":");
    Serial.print(vibZ, 2);
    Serial.print(",\"vibration_magnitude\":");
    Serial.print(vibMagnitude, 2);
    Serial.print(",\"current\":");
    Serial.print(current, 2);
    Serial.print(",\"rpm\":");
    Serial.print((int)rpm);
    Serial.println("}");
  }
}
