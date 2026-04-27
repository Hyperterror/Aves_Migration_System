# 🐦 Aves Migration Prediction System
### Predicting Bird Migration Success using Machine Learning

---

## 🌍 Overview

This project builds an end-to-end Machine Learning pipeline to predict whether a bird successfully completes its migration using GPS telemetry data.

Migration success is defined as whether the bird returns to its home region after traveling long distances.

---

## 🎯 Problem Statement

Predicting migration success from raw GPS data is challenging due to:

- Massive data size (~4.8 million records)
- Highly imbalanced dataset (≈92% success, ≈8% failure)
- Data dominated by a single bird (bias issue)
- Raw data being time-series (not directly usable for ML)

---

## 📊 Dataset

- **Source:** Audubon Canyon Ranch Telemetry Project  
- **Species:** Great Egret (*Ardea alba*)  
- **Time Span:** 2017 – 2026 (~9 years)  
- **Total Records:** 4.8 million GPS fixes  
- **Birds Tracked:** 22  
- **Sampling Interval:** Every 30 minutes  

Each record contains:
- Latitude, Longitude  
- Timestamp  
- Speed, Direction  
- Environmental sensor data  

⚠️ Note: Each row represents a single GPS observation, not a full migration.

---

## 🔄 Data Transformation

Raw GPS data was converted into **migration events** using:

- Distance ≥ 50 km  
- Duration ≥ 2 days  
- ≥ 20 GPS points  

➡️ Final dataset: **29 migration events**

---

## 🧠 Feature Engineering

Each migration event was transformed into **42 features** across multiple categories:

- 🌍 Geospatial (distance, displacement)
- 📅 Temporal (duration, time patterns)
- 🚀 Movement (speed, direction)
- 🌡️ Sensor (temperature, battery)
- 🏔️ Altitude
- 🦅 Behaviour

Example feature:
> **Tortuosity = Path Length / Displacement**  
Measures efficiency of movement.

---

## ⚠️ Key Challenges

### 1. Class Imbalance
- 27 success vs 2 failure
- Model learns biased patterns

### 2. Bird Dominance
- One bird contributes ~50% of data

### 3. Misleading Accuracy
- Initial models showed **100% accuracy** (incorrect)

---

## 🔧 Bias Correction Techniques

To ensure reliable learning:

- **SMOTE** → Synthetic minority samples
- **Stratified Cross-Validation** → Balanced testing
- **Class Weights** → Penalize bias
- **Per-bird Weighting** → Reduce dominance

---

## 🤖 Models Used

| Model | Purpose |
|------|--------|
| Logistic Regression | Baseline linear model |
| SVM | Handles complex boundaries |
| KNN | Distance-based similarity |
| Decision Tree | Interpretable rules |
| Random Forest ⭐ | Best performer |
| XGBoost | Advanced boosting |

---

## 📈 Results

- **Best Model:** Random Forest  
- **Accuracy:** ~94–97%  
- **F1 Score:** Highest among all models  

📌 Note:
> F1 Score is prioritized over accuracy due to class imbalance.

---

## 🌐 Web Visualization

The project includes an interactive web dashboard:

- 🗺️ Bird migration map (Leaflet.js)
- 🔥 Heatmap visualization
- 📊 Model comparison charts (Chart.js)
- 📈 Bias correction visualizations

Files used:
- `bird_tracks_all.json`
- `website_data.json`

---

## 📁 Project Structure

```
Aves_Migration_System/
│
├── telemetry_preprocess.py
├── telemetry_bucket_splitter.py
├── telemetry_model_training.py
├── telemetry_bias_fix.py
│
├── telemetry_migration_events.csv
├── telemetry_bucket_data/
│
├── bird_tracks_all.json
├── website_data.json
│
├── index.html
└── README.md
```

---

## 💡 Key Insight

> In real-world machine learning, data quality, preprocessing, and bias handling are more important than the choice of model.

---

## 🚀 Future Work

- Increase dataset size (more migration events)
- Apply deep learning (LSTM for time-series)
- Real-time migration prediction
- Deploy as web application

