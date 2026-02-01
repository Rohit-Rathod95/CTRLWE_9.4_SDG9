# 🏭 Industrial Predictive Maintenance Intelligence Platform  
**PS ID: 9.4 – Central India Hackathon (CIH 3.0)**

## 📌 Overview
This project presents an **AI-powered predictive maintenance system** designed to detect early signs of machine degradation, identify root causes, and recommend actionable maintenance steps.

The system combines:
- **Custom industrial datasets**
- **Ensemble machine learning**
- **AI-driven diagnostics**
- **Real-time alerting**

to move maintenance from **reactive → predictive → intelligent**.

---

## 🎯 Problem Statement (PS ID: 9.4)
Unexpected machine failures in industrial environments lead to:
- Production downtime
- High repair costs
- Safety risks
- Efficiency loss

Traditional maintenance approaches fail to detect *early degradation patterns*.

**Objective:**  
Build a scalable ML system that predicts machine health, diagnoses failure causes, and assists maintenance teams with timely decisions.

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
- Vibration Index
- Thermal Index
- Efficiency Score
- Asset Risk Level
- AI-generated maintenance actions & timelines

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit (Dashboard)
- Gemini 2.5 Pro (AI reasoning)
- Twilio (Alerts)

---

## 🚀 Future Enhancements
- IoT sensor integration (real-time data)
- Edge deployment for factories
- Digital twin modeling
- Remaining Useful Life (RUL) prediction

---

## 👥 Team
- Rohit Rathod  
- Chitransh Damhedhar  
- Ujwal Prakash Hiwase  
- Prachit Mankar  

---

## 📄 License
Academic & educational use only.
