# 🦢 Aves Migration System

![Live ML Project](https://img.shields.io/badge/Live_ML_Project-Active-success)
![Subject](https://img.shields.io/badge/Species-Great_Egret_(Ardea_alba)-blue)
![Dataset](https://img.shields.io/badge/Dataset-4.8M_GPS_Fixes-orange)

The **Aves Migration System** is an end-to-end Machine Learning pipeline designed to predict bird migration success using massive volumes of raw GPS telemetry data. Focused on the **Great Egret (*Ardea alba*)** populations around the Audubon Canyon Ranch (Bolinas Lagoon, California), the project processes millions of sensor points into a comprehensive predictive system.

## 📊 Dataset & Scope
* **Kaggle Dataset:** [Audubon Canyon Ranch Egret Telemetry Project](https://www.kaggle.com/datasets/ishitagarwal/audubon-canyon-ranch-egret-telemetry-project)
* **Data Volume:** 4,808,057 raw GPS fixes
* **Tracking Duration:** 2017 – 2026
* **Subjects:** 22 tagged Great Egrets
* **Total Migration Events:** 38 clean, ML-ready events
* **Features:** 48 raw sensor columns engineered into **42 predictive features**

## ⚙️ Processing Pipeline
The project pipeline transforms granular telemetry data into semantic biological metrics across three core Python modules:
1. **`telemetry_preprocess.py`**: Conducts chunked data loading using pandas (processing 200k-row blocks to handle memory efficiently). It cleans invalid coordinates, parses timestamps, removes outlier speeds (>50 m/s), computes personal breeding sites (home bases), and extracts continuous migration events via a state machine.
2. **`telemetry_bucket_splitter.py`**: Organizes the engineered features into 6 distinct biological "buckets" allowing isolating studies on different ecological metrics.
3. **`telemetry_bias_fix.py`**: A specialized data-correction pipeline that corrects severe class imbalances and applies strict cross-validation methodologies.

### 🧪 Feature Engineering (The 6 Buckets)
* **🌍 Geospatial:** Route tortuosity, displacement, bearing statistics, and maximum distance reached.
* **📅 Temporal:** Migration duration, modal movement hour (diurnal/nocturnal), start and end DOY.
* **🚀 Movement:** Speed dynamics, heading consistency, fix rates.
* **🌡️ Sensor:** Onboard tag health proxies (battery decay, temperature metrics).
* **🏔️ Altitude:** Flight envelope metrics derived from height-above-ellipsoid.
* **🦅 Behaviour:** Rest stop durations, movement consistency ratios, long flight segments.

## ⚖️ Imbalance & Bias Correction
Initially, the data exhibited a **13.5:1 class imbalance** (93% Migration Success vs 7% Failed) which resulted in heavily skewed, artificially high accuracy. The `telemetry_bias_fix.py` pipeline resolved this through a 4-layer methodology:

1. **Stricter Success Labeling:** Tightened the successful home-return threshold from 100 km to **60 km**, reclassifying borderline trips as "Failed".
2. **SMOTE Oversampling:** Synthetically generated "Failed" minority samples exclusively within the training fold to prevent mathematical data leakage.
3. **Per-Bird Sample Weights:** Eliminated bias from high-frequency tagged birds (e.g., GREG_8 or GREG_5) by ensuring each bird contributed equally to the model's pattern recognition.
4. **Stratified K-Fold Cross-Validation:** Ensured every fold contained both Successful and Failed samples for honest evaluation.

## 🤖 Model Performance
Six distinct classification models were benchmarked using the fully corrected dataset:
* **Logistic Regression**
* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**
* **Random Forest** (Best F1 Stability)
* **XGBoost** (Highly capable native imbalance handling)

**Peak Accuracy Achieved:** ~**94.7%** (with robust precision/recall splits)

## 🌐 Visualization & Dashboard
The findings, feature maps, and model analytics are visualized in two interactive frontends:
* `index.html`: A fully interactive presentation dashboard including a Leaflet-powered hotspot heatmap of all telemetry data.
* `research_poster.html`: A printable, structured overview formatting the findings and charts for academic presentation.

## 🚀 Running the Project
*(Depending on final directory structure, you can run the files as follows)*
```bash
# 1. Generate core engineered features from raw data
python telemetry_preprocess.py

# 2. Split features into the 6 operational buckets
python telemetry_bucket_splitter.py

# 3. Train models and generate bias-corrected visualizations
python telemetry_bias_fix.py
```
> Note: Visualizations and model evaluation outputs will be dynamically saved to the `telemetry_bias_corrected_results/` directory.

---
*Developed for the Audubon Canyon Ranch Egret Telemetry Project.*
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

