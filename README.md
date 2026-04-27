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
