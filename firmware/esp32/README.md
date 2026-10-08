# ESP32 Edge Sensor Acquisition Module

## Purpose

This firmware provides the basic **hardware sensor acquisition layer** for the Industrial Predictive Maintenance Platform. It runs on an ESP32 microcontroller to collect real-time physical telemetry from key industrial sensors:

- Tri-axial vibration and mechanical shock (accelerometer)
- Machine surface / bearing temperature
- Motor current draw / electrical load
- Rotational shaft velocity (RPM)

The firmware periodically outputs diagnostic logs and formatted single-line JSON data packets over USB Serial, serving as the foundational physical data layer.

> **Current Status:** **Basic ESP32 sensor acquisition implemented; integration with the predictive-maintenance software is a future step.**  
> *(No edge ML inference, cloud APIs, MQTT broker, or database communication are executed on the ESP32 in this stage).*

---

## Hardware Components

1. **ESP32 Development Board** (NodeMCU ESP-32S, ESP-WROOM-32, or equivalent)
2. **MPU6050** 6-Axis Accelerometer / Gyroscope (I2C interface)
3. **DS18B20** Digital Temperature Sensor (1-Wire interface, waterproof probe recommended for industrial casing)
4. **ACS712** Current Sensor Module (Analog Hall-effect current module; 5A, 20A, or 30A version)
5. **Hall-Effect Sensor** (e.g., A3144 or digital magnetic switch module) + neodymium shaft magnet
6. **4.7kΩ Resistor** (pull-up for DS18B20 data line)
7. Breadboard and jumper wires

---

## Wiring & Pinout Guide

| Sensor Module | Sensor Pin | ESP32 Pin | Notes / Recommended Circuit |
| :--- | :--- | :--- | :--- |
| **MPU6050** | VCC | 3.3V / 5V | Power pin (most breakouts include a 3.3V LDO) |
| | GND | GND | Common ground |
| | SCL | **GPIO 22** | Default hardware I2C Clock (`PIN_I2C_SCL`) |
| | SDA | **GPIO 21** | Default hardware I2C Data (`PIN_I2C_SDA`) |
| **DS18B20** | VDD | 3.3V | Power supply |
| | GND | GND | Common ground |
| | DQ (Data) | **GPIO 4** | 1-Wire Data (`PIN_DS18B20`), requires **4.7kΩ pull-up to 3.3V** |
| **ACS712** | VCC | 5V | ACS712 requires 5V supply for rated Hall-effect operation |
| | GND | GND | Common ground |
| | OUT | **GPIO 34** | ADC1 input (`PIN_ACS712`). *See electrical note below* |
| **Hall Sensor**| VCC | 3.3V | Power supply |
| | GND | GND | Common ground |
| | DO (Signal)| **GPIO 18** | Digital interrupt pin (`PIN_HALL_SENSOR`), active-LOW pulse |

> **Electrical Note for ACS712:**  
> The ACS712 module is powered at 5V, producing a quiescent center voltage of $V_{CC} / 2 \approx 2.5\text{V}$ at 0A. Because ESP32 ADC inputs must stay within **0V to 3.3V**, connect the ACS712 OUT pin through a simple resistor divider (or 3.3V scaling circuit) if measuring large bidirectional currents, and update `ACS_ZERO_OFFSET_V` and `ACS_SENSITIVITY` in the code accordingly.

---

## Required Arduino Libraries

Install the following libraries using the **Arduino IDE Library Manager** (`Sketch` -> `Include Library` -> `Manage Libraries...`):

1. **Adafruit MPU6050** (by Adafruit)
2. **Adafruit Unified Sensor** (dependency required by Adafruit MPU6050)
3. **OneWire** (by Paul Stoffregen)
4. **DallasTemperature** (by Miles Burton)

The standard `Wire` library for I2C communication is included by default with the ESP32 Arduino Core.

---

## How to Configure & Upload Using Arduino IDE

1. **Install ESP32 Board Support:**
   - In Arduino IDE, open **File > Preferences**.
   - In *Additional Board Manager URLs*, add:
     ```text
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Go to **Tools > Board > Boards Manager...**, search for `esp32` by *Espressif Systems*, and click **Install**.

2. **Open the Project:**
   - Open `firmware/esp32/predictive_maintenance.ino` in Arduino IDE.

3. **Adjust Configurations (if needed):**
   - Check the top section of `predictive_maintenance.ino`:
     ```cpp
     const char* DEVICE_ID = "MACHINE_01";
     const unsigned long SAMPLING_INTERVAL = 1000; // ms
     const float ACS_SENSITIVITY = 0.185;          // 0.185 for 5A, 0.100 for 20A, 0.066 for 30A
     const int PULSES_PER_REV = 1;                 // Number of magnets on rotating shaft
     ```

4. **Select Board & Port:**
   - Board: **Tools > Board > esp32 > ESP32 Dev Module**
   - Port: **Tools > Port > [Select your ESP32 COM Port]**

5. **Upload:**
   - Click the **Upload** button (arrow icon) in Arduino IDE.
   - If required by your ESP32 board, hold down the `BOOT` button during connection until the upload begins.

---

## How to Open Serial Monitor

1. In Arduino IDE, navigate to **Tools > Serial Monitor** (or press `Ctrl + Shift + M`).
2. Set the baud rate in the bottom right corner of the Serial Monitor to **`115200 baud`**.
3. Press the `EN` / `RST` button on the ESP32 board to observe the boot diagnostics.

---

## Expected Serial Output

Upon boot, the firmware tests the initialization of the I2C and 1-Wire sensors, then streams readings at 1-second intervals:

```text
==================================================
  ESP32 Predictive Maintenance Sensor Acquisition 
==================================================
Device ID: MACHINE_01
[OK]    MPU6050 initialized successfully.
[OK]    DS18B20 initialized successfully. Found devices: 1
[OK]    ACS712 ADC channel configured on GPIO 34.
[OK]    Hall-Effect interrupt attached on GPIO 18.
--------------------------------------------------
Telemetry streaming started...

--------------------------------------------------
Temperature: 42.5 C
Acceleration: X=0.12 Y=0.18 Z=1.04
Vibration Magnitude: 1.06
Current: 1.82 A
RPM: 1450
{"device_id":"MACHINE_01","temperature":42.5,"vibration_x":0.12,"vibration_y":0.18,"vibration_z":1.04,"vibration_magnitude":1.06,"current":1.82,"rpm":1450}
```

---

## Next Steps / Roadmap

- [ ] Connect ESP32 to Wi-Fi network using non-blocking connection logic.
- [ ] Publish telemetry payload to an MQTT broker or REST ingest endpoint.
- [ ] Ingest streaming edge data into the Python backend predictive maintenance pipeline.
