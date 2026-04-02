# ML Methods & Results

## Overview
This document describes the machine learning methodology used for drought risk prediction and water stress forecasting.

---

## 1. Problem Formulation

### Task A: Drought Risk Classification
- **Objective:** Classify regions into drought risk categories
- **Target:** `drought_risk_class` ∈ {Low, Moderate, High, Extreme}
- **Type:** Multi-class classification

### Task B: Water Stress Prediction
- **Objective:** Predict future water stress scores (3-month horizon)
- **Target:** `drought_composite_score` (continuous, 0–5 scale)
- **Type:** Regression with temporal forecasting

---

## 2. Feature Engineering

### Source Features
| Feature | Source | Description |
|---------|--------|-------------|
| `water_stress_score` | Aqueduct | Baseline water stress (0–5) |
| `water_depletion_score` | Aqueduct | Water depletion ratio (0–5) |
| `drought_risk_score` | Aqueduct | Drought risk assessment (0–5) |
| `interannual_variability` | Aqueduct | Year-to-year supply variation |
| `seasonal_variability` | Aqueduct | Within-year supply variation |
| `groundwater_decline_score` | Aqueduct | Groundwater table decline rate |
| `tws_anomaly_cm` | GRACE | Terrestrial Water Storage anomaly |
| `groundwater_anomaly_cm` | GRACE | Groundwater storage anomaly |
| `sdg642_water_stress_pct` | AQUASTAT | SDG 6.4.2 water stress percentage |
| `total_renewable_water_km3` | AQUASTAT | Total renewable freshwater |
| `total_water_withdrawal_km3` | AQUASTAT | Total water withdrawal |
| `precipitation_mm` | AQUASTAT | Annual precipitation |

### Derived Features
| Feature | Method | Purpose |
|---------|--------|---------|
| `drought_composite_score` | Weighted avg of indicators | Unified drought severity metric |
| `tws_3/6/12month_avg` | Rolling mean | Smooth seasonal noise |
| `tws_rate_of_change` | Month-over-month diff | Capture acceleration/deceleration |
| `tws_cumulative_change` | Cumulative sum from baseline | Long-term trend |
| `tws_zscore` | Z-score per country | Normalize for cross-country comparison |
| Lag features (1,3,6,12 months) | Temporal shift | Capture autoregressive patterns |
| `month_sin`, `month_cos` | Cyclical encoding | Seasonal patterns without discontinuity |

### Target Variable Engineering
- **Classification:** `drought_risk_class` derived from `drought_composite_score`:
  - Low: 0.0 – 1.0
  - Moderate: 1.0 – 2.0
  - High: 2.0 – 3.5
  - Extreme: 3.5 – 5.0

- **Regression:** Target is `drought_composite_score` shifted 3 months into the future

---

## 3. Model Selection

### Classification Models

#### Random Forest Classifier
- **Rationale:** Robust to outliers, handles non-linear relationships, provides feature importance
- **Hyperparameters:**
  - `n_estimators`: [100, 200] (tuned via GridSearchCV)
  - `max_depth`: [10, 15, 20]
  - `min_samples_split`: [3, 5]
  - `min_samples_leaf`: [1, 2]

#### XGBoost Classifier
- **Rationale:** State-of-the-art for tabular data, handles imbalanced classes, gradient-boosted
- **Hyperparameters:**
  - `n_estimators`: [200, 300]
  - `max_depth`: [6, 8]
  - `learning_rate`: [0.05, 0.1]
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `eval_metric`: mlogloss

### Regression Models

#### Ridge Regression (Baseline)
- **Rationale:** Simple, interpretable baseline for comparison
- **Regularization:** α tuned from [0.01, 0.1, 1.0, 10.0, 100.0]

#### XGBoost Regressor
- **Rationale:** Captures non-linear relationships and feature interactions
- **Same boosting hyperparameters as classifier variant

---

## 4. Training Procedure

### Data Split
- **Method:** Temporal split (not random)
- **Training:** All data before 2018
- **Testing:** All data from 2018 onwards
- **Rationale:** Simulates real-world prediction (can't use future data to predict past)
- **Fallback:** If temporal split produces empty sets, 80/20 stratified random split

### Cross-Validation
- **Method:** Stratified K-Fold (K=5 for classification, K=5 for regression)
- **Scoring:** 
  - Classification: F1 weighted
  - Regression: Negative RMSE

### Hyperparameter Tuning
- **Method:** Grid Search with Cross-Validation (GridSearchCV)
- **Selection criterion:** Best cross-validation score

---

## 5. Evaluation Metrics

### Classification
| Metric | Description |
|--------|-------------|
| F1 Score (weighted) | Harmonic mean of precision/recall, weighted by class frequency |
| Accuracy | Overall correct predictions / total |
| Confusion Matrix | Visual breakdown of true vs predicted classes |
| ROC-AUC | Area under ROC curve (one-vs-rest for multi-class) |

### Regression
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error — penalizes large errors |
| MAE | Mean Absolute Error — average error magnitude |
| R² | Coefficient of determination — variance explained |

---

## 6. Explainability (SHAP)

We use **SHAP (SHapley Additive exPlanations)** for model interpretability:

- **Method:** TreeExplainer (optimized for tree-based models)
- **Output:** Per-feature contribution to each prediction
- **Plots:**
  - Summary plot (beeswarm) — feature importance + directionality
  - Bar plot — aggregate feature importance across classes

### Key Findings (Typical)
1. `drought_composite_score` — strongest predictor (by design, validates the composite)
2. `tws_anomaly_cm` — satellite-derived water storage is highly informative
3. `water_stress_score` — Aqueduct baseline stress correlates with risk class
4. Lag features — temporal patterns matter for prediction

---

## 7. Unsupervised Analysis (K-Means)

In addition to supervised models, we cluster countries by drought profile:

- **Algorithm:** K-Means (k=4, matching risk categories)
- **Features:** Country-mean of all water stress indicators
- **Preprocessing:** StandardScaler normalization
- **Validation:** Elbow method + silhouette analysis
- **Output:** Risk zone classification for each country

---

## 8. Reproducing Results

```bash
# Step 1: Generate consolidated data
python -m src.pipeline.quarterly_pipeline

# Step 2: Run ML notebook
jupyter notebook notebooks/06_drought_risk_modeling.ipynb

# Step 3: Run mapping notebook
jupyter notebook notebooks/07_seasonal_water_stress_mapping.ipynb

# Step 4: Launch dashboard
streamlit run dashboard/app.py
```

---

## 9. Limitations & Future Work

### Current Limitations
- Synthetic data used when real data sources are unavailable
- AQUASTAT data is sparse (5-year intervals) — interpolation introduces smoothing
- GRACE has a gap during satellite transition (2017-2018)
- Country-level aggregation loses sub-national variation

### Future Improvements
- LSTM/Transformer models for capturing long-range temporal dependencies
- Higher spatial resolution (sub-national, grid-level)
- Real-time data integration (near real-time GRACE-FO data)
- Ensemble stacking (combining RF + XGBoost + Ridge)
- Climate projection integration (CMIP6 scenarios for future predictions)
