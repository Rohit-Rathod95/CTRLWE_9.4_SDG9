# 🏭 Industrial Predictive Maintenance Intelligence Platform  
**PS ID: 9.4 – Central India Hackathon (CIH 3.0)**

## 📌 Overview
This project presents an **AI-powered predictive maintenance system** designed to detect early signs of machine degradation, identify root causes, and recommend actionable maintenance steps.

The platform spans the full industrial maintenance lifecycle:
- **Physical Edge Sensor Layer:** ESP32-based hardware acquisition for vibration, temperature, current, and RPM
- **Custom Industrial Datasets:** Modeled from real-world turbofan and manufacturing telemetry
- **Ensemble Machine Learning:** Robust multi-model prediction of machine degradation
- **AI-Driven Diagnostics:** Root-cause analysis and actionable repair guidance via LLM reasoning
- **Real-Time Alerting:** Instant SMS & WhatsApp notifications for operational emergencies

Moving maintenance from **reactive → predictive → intelligent**.

---

## 🎯 Problem Statement (PS ID: 9.4)
Unexpected machine failures in industrial environments lead to:
- Production downtime
- High repair costs
- Safety risks
- Efficiency loss

Traditional maintenance approaches fail to detect *early degradation patterns*.

**Objective:**  
Build an end-to-end scalable predictive maintenance solution that acquires physical machine telemetry, predicts machine health degradation, diagnoses failure causes, and assists plant maintenance engineers with timely decisions.

---

## ⚡ Edge Hardware & Sensor Acquisition Layer

The platform includes a dedicated **ESP32 Arduino firmware module** (`firmware/esp32/`) responsible for acquiring real-time physical telemetry from machinery:

- **MPU6050 (I2C):** Tri-axial acceleration ($X, Y, Z$) and composite vibration magnitude calculation
- **DS18B20 (1-Wire):** Machinery surface and bearing temperature monitoring
- **ACS712 (ADC):** Motor current draw and electrical load sensing
- **Hall-Effect Sensor (Interrupt):** Rotational velocity (RPM) measurement via hardware pulse counting

The edge module outputs formatted diagnostics and standardized JSON telemetry packets over Serial (115200 baud):
```json
{"device_id":"MACHINE_01","temperature":42.5,"vibration_x":0.12,"vibration_y":0.18,"vibration_z":1.04,"vibration_magnitude":1.06,"current":1.82,"rpm":1450}
```

> **Note:** Basic ESP32 sensor acquisition is implemented. Integration with the streaming analytics pipeline (Wi-Fi/MQTT/REST) is scheduled for the next development phase. See [firmware/esp32/README.md](firmware/esp32/README.md) for complete pinout and wiring guides.

---

## 📁 Repository Structure

```text
.
├── firmware/
│   └── esp32/
│       ├── predictive_maintenance.ino   # ESP32 Arduino sensor acquisition firmware
│       └── README.md                    # Hardware wiring, pinout & setup guide
├── predictive-maintenance/
│   ├── alerts.py                        # Twilio SMS / WhatsApp alerting service
│   ├── app.py                           # Streamlit analytics & diagnostics dashboard
│   ├── features.py                      # Feature engineering & transformation utilities
│   ├── train_model.py                   # Ensemble model training script
│   ├── data/                            # Processed dataset files
│   └── models/                          # Serialized ML models & scalers
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## 🧩 Dataset Strategy

### 🔹 Custom Industrial Dataset
A custom dataset was created by taking reference from:
- **NASA Turbofan Engine Degradation Dataset**
- **UCI AI4I 2020 Predictive Maintenance Dataset**

The dataset simulates real industrial conditions with features such as:
- Vibration index  
- Thermal index  
- Efficiency metrics  
- Operational load patterns  
- Environmental context (simulated)

---

## 🔄 Data Processing Pipeline
1. Data cleaning & normalization  
2. Feature engineering:
   - Mechanical stress indicators  
   - Thermal degradation patterns  
   - Efficiency decay trends  
3. Dataset splitting with leakage prevention  
4. Model-specific preprocessing  

---

## 🧠 Machine Learning Architecture

### Models Implemented
- **Model 1:** XGBoost (optimized hyperparameters)
- **Model 2:** Random Forest
- **Model 3:** Histogram Gradient Boosting
- **Model 4:** Ridge Regression

### 🔗 Ensemble Strategy
A **weighted ensemble** approach combines all models to:
- Improve prediction stability
- Reduce variance
- Increase robustness across machine types

---

## 🤖 AI Intelligence Layer
An AI reasoning layer powered by **Gemini 2.5 Pro** generates:
- Root cause diagnosis  
- Maintenance recommendations  
- Risk classification  
- Maintenance timelines (Immediate / Short / Long term)

This bridges the gap between **ML predictions and human decision-making**.

---

## 🚨 Smart Notification System
- Integrated **Twilio** for real-time alerts
- Sends **SMS & WhatsApp notifications** on critical failures
- Includes:
  - Asset ID
  - Health metrics
  - AI diagnosis
  - Immediate action steps

---

## 📊 System Outputs
- Vibration Index & Acceleration Vector
- Thermal Index / Bearing Temperature
- Efficiency Score
- Motor Current & Electrical Load
- Rotational Speed (RPM)
- Asset Risk Level
- AI-generated maintenance actions & timelines

---

## 🛠️ Tech Stack
- **Edge / Embedded:** C++, Arduino IDE, ESP32, I2C, 1-Wire, ADC
- **Core Analytics & ML:** Python, Pandas, NumPy, Scikit-learn, XGBoost
- **Web Dashboard:** Streamlit
- **Generative AI Diagnostics:** Gemini 2.5 Pro
- **Alerting & Communications:** Twilio (SMS & WhatsApp)

---

## 🚀 Future Enhancements
- [x] Dedicated ESP32 physical sensor acquisition firmware module
- [ ] Wireless telemetry ingestion (ESP32 Wi-Fi / MQTT / HTTP to backend)
- [ ] Edge-based lightweight anomaly detection (TinyML on ESP32)
- [ ] Digital twin modeling
- [ ] Remaining Useful Life (RUL) regression forecasting

---

## 👥 Team
- Rohit Rathod  
- Chitransh Damhedhar  
- Ujwal Prakash Hiwase  
- Prachit Mankar  

---

## 📄 License
Academic & educational use only.

