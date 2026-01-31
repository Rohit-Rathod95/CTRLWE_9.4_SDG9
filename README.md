# 🏭 Predictive Maintenance System (PS ID: 9.4)

## 📌 Overview
This project implements an **AI‑based Predictive Maintenance System** that predicts machine failures in advance using **machine sensor data** enriched with **environmental (weather) context**.  
The solution is designed to be **versatile, scalable, and applicable across multiple machine types** in industrial environments.

---

## 🎯 Problem Statement (PS ID: 9.4)
Industries face frequent **unexpected machine failures** leading to downtime, high maintenance costs, reduced efficiency, and safety risks.  
Traditional maintenance approaches are either **reactive** or **schedule‑based**, which are inefficient and costly.

The goal is to design a **machine learning–based predictive maintenance system** that:
- Predicts machine failures before they occur  
- Identifies the **type of failure**  
- Considers **environmental conditions**  
- Supports proactive and data‑driven maintenance decisions  

---

## 💡 Proposed Solution
We propose a **context‑aware predictive maintenance platform** that:
- Uses **machine sensor data** to detect early degradation
- Integrates **weather data** (temperature, humidity, rainfall) as contextual features
- Predicts:
  - Machine Failure (Yes / No)
  - Failure Type (Tool Wear, Heat Dissipation, Power Failure, etc.)
- Generates actionable insights for maintenance teams

---

## 📂 Datasets Used

### 1️⃣ UCI AI4I 2020 Predictive Maintenance Dataset
- Machine sensor readings:
  - Air temperature
  - Process temperature
  - Rotational speed
  - Torque
  - Tool wear
- Failure labels:
  - Machine failure (binary)
  - Failure types:
    - TWF – Tool Wear Failure
    - HDF – Heat Dissipation Failure
    - PWF – Power Failure
    - OSF – Overstrain Failure
    - RNF – Random Failure

### 2️⃣ Synthetic Industrial Weather Dataset
- Ambient temperature
- Humidity
- Rainfall
- Used **only as contextual features**, not labels

---

## 🔄 Data Processing Pipeline
1. Load and clean AI4I machine sensor data  
2. Generate / load synthetic weather data  
3. Simulate timestamps for AI4I dataset  
4. Perform **time‑aware feature‑level merge** with weather data  
5. Engineer additional features:
   - Thermal gap
   - Environmental stress
   - Mechanical stress (vibration proximity)

---

## 🧠 Machine Learning Approach

### Model Design
- **Stage 1:** Binary Classification  
  → Predict Machine Failure (Yes / No)

- **Stage 2:** Multi‑Class Classification  
  → Predict Failure Type (TWF, HDF, PWF, OSF, RNF)

### Models Used
- Random Forest
- XGBoost (optional)

### Key ML Considerations
- Handles **class imbalance** using class weighting
- Evaluated using **F1‑score and Recall**
- Avoids data leakage during training

---

## 📊 System Outputs
- Failure probability
- Failure type prediction
- Weather‑adjusted risk score
- Machine health insights

---

## 🚀 Key Advantages
- Early failure prediction → reduced downtime  
- Lower maintenance and repair costs  
- Adaptable to **multiple machine types**  
- Improved explainability using contextual data  
- Industry‑ready and scalable design  

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit‑learn
- XGBoost (optional)
- Jupyter Notebook / Python scripts

---

## 🏁 Conclusion
This system shifts industrial maintenance from **reactive** to **predictive**, enabling smarter decision‑making, improved machine health, and cost‑efficient operations.

---

## 👥 Team
- Rohit Rathod  
- Chengiskhan  
- Ujwal Prakash Hiwase  
- Prachit Mankar  

---

## 📄 License
This project is for academic and educational purposes.
