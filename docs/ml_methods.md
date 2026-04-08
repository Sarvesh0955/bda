# ML Methods & Results

## Overview
This document describes the machine learning methodology used for drought risk prediction and water stress forecasting, trained on **real-world data** from WRI Aqueduct 4.0, FAO AQUASTAT, and NASA GRACE/GRACE-FO.

---

## 1. Problem Formulation

### Task A: Drought Risk Classification
- **Objective:** Classify regions into drought risk categories
- **Target:** `drought_risk_class` ∈ {Low, Moderate, High, Extreme}
- **Type:** Multi-class classification

### Task B: Water Stress Prediction
- **Objective:** Predict future drought composite scores (3-month horizon)
- **Target:** `drought_composite_score` (continuous, 0–5 scale)
- **Type:** Regression with temporal forecasting

---

## 2. Feature Engineering

### Source Features (from real datasets)
| Feature | Source | Description |
|---------|--------|-------------|
| `water_stress_score` | WRI Aqueduct 4.0 | Baseline water stress (0–5) |
| `water_depletion_score` | WRI Aqueduct 4.0 | Water depletion ratio (0–5) |
| `drought_risk_score` | WRI Aqueduct 4.0 | Drought risk assessment (0–5) |
| `interannual_variability` | WRI Aqueduct 4.0 | Year-to-year supply variation |
| `seasonal_variability` | WRI Aqueduct 4.0 | Within-year supply variation |
| `groundwater_decline_score` | WRI Aqueduct 4.0 | Groundwater table decline rate |
| `tws_anomaly_cm` | NASA GRACE | Terrestrial Water Storage anomaly (cm) |
| `groundwater_anomaly_cm` | NASA GRACE | Estimated groundwater component (cm) |
| `uncertainty_cm` | NASA GRACE | GRACE measurement uncertainty (cm) |
| `sdg642_water_stress_pct` | FAO AQUASTAT | SDG 6.4.2 water stress percentage |
| `total_renewable_water_km3` | FAO AQUASTAT | Total renewable freshwater (km³/yr) |
| `total_water_withdrawal_km3` | FAO AQUASTAT | Total water withdrawal (km³/yr) |
| `precipitation_mm` | FAO AQUASTAT | Annual precipitation (mm) |

### Derived Features (computed during processing & training)
| Feature | Method | Purpose |
|---------|--------|---------|
| `drought_composite_score` | Weighted avg of 4 indicators | Unified drought severity metric |
| `tws_3/6/12month_avg` | Rolling mean | Smooth seasonal noise |
| `tws_rate_of_change` | Month-over-month diff | Capture acceleration/deceleration |
| `tws_cumulative_change` | Cumulative sum from baseline | Long-term trend |
| `tws_zscore` | Z-score per country | Normalize for cross-country comparison |
| `tws_trend_cm_yr` | Linear regression slope | Annual trend direction |
| Lag features (1,3,6 months) | Temporal shift | Capture autoregressive patterns |
| `month_sin`, `month_cos` | Cyclical encoding | Seasonal patterns without discontinuity |

### Target Variable
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
- **Hyperparameters (tuned via GridSearchCV):**
  - `n_estimators`: [100, 200]
  - `max_depth`: [10, 15, 20]
  - `min_samples_split`: [3, 5]
  - `min_samples_leaf`: [1, 2]

#### XGBoost Classifier
- **Rationale:** State-of-the-art for tabular data, gradient-boosted decision trees
- **Hyperparameters:**
  - `n_estimators`: [200, 300]
  - `max_depth`: [6, 8]
  - `learning_rate`: [0.05, 0.1]
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8

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
- **Method:** Temporal split (not random) — prevents data leakage
- **Training:** All data before 2018
- **Testing:** All data from 2018 onwards
- **Rationale:** Simulates real-world use — can't use future data to predict past

### Cross-Validation
- **Method:** Stratified K-Fold (K=3)
- **Scoring:**
  - Classification: F1 weighted
  - Regression: Negative RMSE

### Hyperparameter Tuning
- **Method:** Grid Search with Cross-Validation (GridSearchCV)
- **Selection:** Best cross-validation score

---

## 5. Evaluation Metrics

### Classification
| Metric | Description |
|--------|-------------|
| F1 Score (weighted) | Harmonic mean of precision/recall, weighted by class frequency |
| Accuracy | Overall correct predictions / total |
| Confusion Matrix | Breakdown of true vs predicted classes |

### Regression
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |

---

## 6. Explainability (SHAP)

We use **SHAP (SHapley Additive exPlanations)** for model interpretability:

- **Method:** TreeExplainer (optimized for tree-based models)
- **Output:** Per-feature contribution to each prediction

---

## 7. Reproducing Results

```bash
# Step 1: Process raw data
source venv/bin/activate
python process_real_data.py

# Step 2: Train models
python train_models.py

# Step 3: Launch dashboard
streamlit run dashboard/app.py
```

---

## 8. Limitations & Future Work

### Current Limitations
- AQUASTAT data is sparse (5-year intervals) — interpolation introduces smoothing
- GRACE has a gap during satellite transition (2017-2018)
- Country-level aggregation loses sub-national variation
- GRACE uses centroid extraction (single grid cell per country)

### Future Improvements
- LSTM/Transformer models for long-range temporal dependencies
- Higher spatial resolution (sub-national, grid-level)
- Real-time data integration (near real-time GRACE-FO)
- Ensemble stacking (RF + XGBoost + Ridge)
- Climate projection integration (CMIP6 scenarios)
