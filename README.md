# 🌍 EDA-15: Water Stress & Drought Index Tracker

A comprehensive system for monitoring global water scarcity and drought levels, featuring ML-driven drought risk prediction and seasonal water stress zone mapping — powered by **real-world data** from three authoritative sources.

## 📋 Project Overview

| Source | Data Type | Coverage | Format |
|--------|-----------|----------|--------|
| **WRI Aqueduct 4.0** | Water risk indicators (13 metrics) | 228 countries, catchment-level | CSV (193 MB) |
| **FAO AQUASTAT** | Water resources & withdrawals (SDG 6.4.2) | 74 countries, 1960–2022 | CSV (936K rows) |
| **NASA GRACE/GRACE-FO** | Terrestrial Water Storage anomalies | 72 countries, monthly 2002–2026 | NetCDF (43 MB) |

**Consolidated Output:** 18,216 records × 39 features covering 72 countries with monthly resolution.

## 🚀 Quick Start

### Installation
```bash
cd /Users/sarvesh0955/College/bda
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Download Data (Manual)
Download the three datasets and place them in `data/raw/`:
- **Aqueduct:** [wri.org/data](https://www.wri.org/data/aqueduct-global-maps-40-data) → Extract ZIP to `data/raw/aqueduct/`
- **AQUASTAT:** [data.apps.fao.org/aquastat](https://data.apps.fao.org/aquastat/) → Bulk download CSV to `data/raw/aquastat/`
- **GRACE:** [podaac.jpl.nasa.gov](https://podaac.jpl.nasa.gov/) → Download NetCDF to `data/raw/grace/1.nc`

See [docs/extraction_workflow.md](docs/extraction_workflow.md) for detailed instructions.

### 2. Process Data
```bash
python process_real_data.py
```

### 3. Train ML Models
```bash
python train_models.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## 📁 Project Structure

```
bda/
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── process_real_data.py       # Data processing pipeline (Aqueduct + AQUASTAT + GRACE)
├── train_models.py            # ML model training script
├── data/
│   ├── raw/                   # Manually downloaded source data
│   │   ├── aqueduct/          # WRI Aqueduct 4.0 CSV + GDB
│   │   ├── aquastat/          # FAO AQUASTAT bulk CSV
│   │   └── grace/             # NASA GRACE NetCDF
│   └── processed/             # Cleaned & merged datasets
│       ├── aqueduct_cleaned.csv
│       ├── aquastat_cleaned.csv
│       ├── grace_cleaned.csv
│       └── drought_water_stress.csv   ← Main consolidated dataset
├── models/                    # Saved trained models (.joblib)
├── dashboard/
│   └── app.py                 # Streamlit web dashboard
├── src/
│   ├── models/                # ML model classes (classifier + predictor)
│   └── visualization/         # Maps & charts
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_extraction_and_handling.ipynb   # Geospatial data handling
│   ├── 02_eda_visualization.ipynb              # EDA & Visualization
│   ├── 03_drought_risk_modeling.ipynb          # ML modeling
│   └── 04_seasonal_water_stress_mapping.ipynb  # Mapping
└── docs/
    ├── extraction_workflow.md # How to download data
    ├── data_processing.md     # Processing pipeline documentation
    ├── ml_methods.md          # Model architecture & evaluation
    └── pipeline_guide.md      # Full pipeline guide
```

## 📓 Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `data_extraction_and_handling` | Download & explore WRI, FAO, and GRACE data |
| 02 | `eda_visualization` | Global maps, distributions, correlations |
| 03 | `drought_risk_modeling` | ML classification + regression + SHAP |
| 04 | `seasonal_water_stress_mapping` | K-Means clustering, animated maps |

## 🤖 ML Models

### Drought Risk Classifier
- **Models:** Random Forest + XGBoost (ensemble)
- **Target:** 4-class: Low / Moderate / High / Extreme
- **Features:** 20+ water stress indicators + temporal lag features
- **Explainability:** SHAP feature importance

### Water Stress Predictor
- **Models:** Ridge Regression (baseline) + XGBoost Regressor
- **Target:** Drought composite score (3-month forecast)
- **Features:** Lagged values + rolling statistics + cyclical temporal encoding

## 📊 Dashboard Features
- 🗺️ Interactive global choropleth maps
- 📈 Time series analysis by country
- 🎯 Risk distribution analytics
- 🔮 ML-powered drought risk predictions
- 📥 Data download & exploration

## 🔄 Pipeline Workflow

```
Manual Download → process_real_data.py → train_models.py → dashboard/app.py
```

For quarterly updates, re-download latest data files and re-run the pipeline.

## 📖 Documentation
- [Extraction Workflow](docs/extraction_workflow.md) — Data download instructions
- [Data Processing](docs/data_processing.md) — Processing pipeline details
- [ML Methods](docs/ml_methods.md) — Model architecture & evaluation
- [Pipeline Guide](docs/pipeline_guide.md) — Full pipeline & scheduling guide

## 📦 Key Dependencies
- `pandas`, `numpy`, `xarray`, `netCDF4` — Data handling
- `scikit-learn`, `xgboost`, `shap` — ML
- `plotly`, `matplotlib`, `folium` — Visualization
- `streamlit` — Dashboard

## 📝 Data Sources & Licenses
- **WRI Aqueduct 4.0:** CC BY 4.0 — [wri.org/aqueduct](https://www.wri.org/aqueduct)
- **FAO AQUASTAT:** Open access — [data.apps.fao.org/aquastat](https://data.apps.fao.org/aquastat/)
- **NASA GRACE:** Open access — [grace.jpl.nasa.gov](https://grace.jpl.nasa.gov/)
