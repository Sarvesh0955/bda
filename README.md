# 🌍 EDA-15: Water Stress & Drought Index Tracker

A comprehensive system for monitoring global water scarcity and drought levels, featuring ML-driven drought risk prediction and seasonal water stress zone mapping.

## 📋 Project Overview

This project integrates data from three authoritative sources to build a unified picture of global water stress:

| Source | Data Type | Coverage |
|--------|-----------|----------|
| **WRI Aqueduct 4.0** | Water risk indicators (13 metrics) | Global, catchment-level |
| **FAO AQUASTAT** | Water resources & withdrawals (SDG 6.4.2) | Country-level, 1960–2020 |
| **NASA GRACE/GRACE-FO** | Terrestrial Water Storage anomalies | Global grid, monthly 2002–present |

## 🚀 Quick Start

### Installation
```bash
cd /Users/sarvesh0955/College/bda
pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
python -m src.pipeline.quarterly_pipeline
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Individual Notebooks
```bash
jupyter notebook notebooks/
```

## 📁 Project Structure

```
bda/
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── data/
│   ├── raw/                   # Downloaded source data
│   │   ├── aqueduct/          # WRI Aqueduct CSVs
│   │   ├── aquastat/          # FAO AQUASTAT CSVs
│   │   └── grace/             # NASA GRACE netCDF/CSV
│   └── processed/             # Cleaned & merged datasets
│       └── drought_water_stress.csv
├── notebooks/
│   ├── 01_data_extraction_aqueduct.ipynb
│   ├── 02_data_extraction_aquastat.ipynb
│   ├── 03_data_extraction_grace.ipynb
│   ├── 04_data_consolidation.ipynb
│   ├── 05_eda_visualization.ipynb
│   ├── 06_drought_risk_modeling.ipynb
│   └── 07_seasonal_water_stress_mapping.ipynb
├── src/
│   ├── extractors/            # Data download & extraction
│   ├── processing/            # Cleaning & merging
│   ├── models/                # ML classifiers & predictors
│   ├── pipeline/              # Automated quarterly pipeline
│   └── visualization/         # Maps & charts
├── dashboard/
│   └── app.py                 # Streamlit web dashboard
├── models/                    # Saved trained models
└── docs/
    ├── extraction_workflow.md
    ├── ml_methods.md
    └── pipeline_guide.md
```

## 📓 Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `data_extraction_aqueduct` | Download & explore WRI water risk data |
| 02 | `data_extraction_aquastat` | Download & process FAO water resources data |
| 03 | `data_extraction_grace` | Download & process NASA TWS anomaly data |
| 04 | `data_consolidation` | Clean, merge all sources into unified CSV |
| 05 | `eda_visualization` | Global maps, distributions, correlations |
| 06 | `drought_risk_modeling` | ML classification + regression + SHAP |
| 07 | `seasonal_water_stress_mapping` | K-Means clustering, animated maps |

## 🤖 ML Models

### Drought Risk Classifier
- **Models:** Random Forest, XGBoost
- **Target:** 4-class: Low / Moderate / High / Extreme
- **Features:** ~20+ water stress indicators + lag features
- **Explainability:** SHAP feature importance

### Water Stress Predictor
- **Models:** Ridge Regression (baseline), XGBoost Regressor
- **Target:** Drought composite score (3-month forecast)
- **Features:** Lagged values + rolling statistics + temporal encoding

## 📊 Dashboard Features
- 🗺️ Interactive global choropleth maps
- 📈 Time series analysis by country
- 🎯 Risk distribution analytics
- 🔮 ML-powered drought risk predictions
- 📥 Data download & exploration

## 🔄 Automated Pipeline
The quarterly pipeline (`src/pipeline/quarterly_pipeline.py`) automates:
1. Data re-download from all sources
2. Cleaning and merging
3. Model re-training
4. Report generation

Schedule with cron:
```bash
0 6 1 1,4,7,10 * cd /path/to/bda && python -m src.pipeline.quarterly_pipeline
```

## 📖 Documentation
- [Extraction Workflow](docs/extraction_workflow.md) — Data source details & access methods
- [ML Methods](docs/ml_methods.md) — Model architecture & evaluation
- [Pipeline Guide](docs/pipeline_guide.md) — Automation & scheduling

## 📦 Key Dependencies
- `pandas`, `numpy`, `xarray` — Data handling
- `scikit-learn`, `xgboost`, `shap` — ML
- `plotly`, `matplotlib`, `folium` — Visualization
- `streamlit` — Dashboard
- `podaac-data-subscriber` — NASA data access

## 📝 Data Sources & Licenses
- **WRI Aqueduct 4.0:** CC BY 4.0 — [wri.org/aqueduct](https://www.wri.org/aqueduct)
- **FAO AQUASTAT:** Open access — [data.apps.fao.org/aquastat](https://data.apps.fao.org/aquastat/)
- **NASA GRACE:** Open access — [grace.jpl.nasa.gov](https://grace.jpl.nasa.gov/)
