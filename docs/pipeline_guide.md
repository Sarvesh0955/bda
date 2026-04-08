# Pipeline Guide

## Overview

The pipeline processes manually-downloaded raw datasets through cleaning, merging, and ML model training to produce a consolidated drought risk dataset and trained prediction models.

---

## Quick Start

### One-time Setup
```bash
cd /Users/sarvesh0955/College/bda
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Download Data (Manual)

Download the following datasets and place them in the specified locations:

| Source | Download From | Save To |
|--------|--------------|---------|
| WRI Aqueduct 4.0 | [wri.org/data](https://www.wri.org/data/aqueduct-global-maps-40-data) | `data/raw/aqueduct/` |
| FAO AQUASTAT | [data.apps.fao.org/aquastat](https://data.apps.fao.org/aquastat/) | `data/raw/aquastat/bulk_eng(in).csv` |
| NASA GRACE | [podaac.jpl.nasa.gov](https://podaac.jpl.nasa.gov/) | `data/raw/grace/1.nc` |

See [extraction_workflow.md](extraction_workflow.md) for detailed download instructions.

### Step 2: Process Data
```bash
source venv/bin/activate
python process_real_data.py
```

This reads all three raw datasets, cleans them, and produces:
- `data/processed/aqueduct_cleaned.csv` — 228 countries, 16 features
- `data/processed/aquastat_cleaned.csv` — 4,500 records (74 countries × years)
- `data/processed/grace_cleaned.csv` — 18,216 records (72 countries × months)
- `data/processed/drought_water_stress.csv` — Consolidated (18,216 × 39 features)

### Step 3: Train ML Models
```bash
source venv/bin/activate
python train_models.py
```

Trains drought risk classifiers and water stress predictors on the processed data. Models are saved to `models/`.

### Step 4: Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MANUAL DOWNLOAD                     │
│  Aqueduct ZIP + AQUASTAT CSV + GRACE NetCDF          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              process_real_data.py                     │
│  1. Clean Aqueduct (sentinel handling, aggregation)  │
│  2. Clean AQUASTAT (encoding, pivot, interpolation)  │
│  3. Clean GRACE (NetCDF extraction, derived features)│
│  4. Merge all three → consolidated CSV               │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                train_models.py                       │
│  1. Feature engineering (lags, cyclical encoding)    │
│  2. Temporal train/test split (pre/post 2018)        │
│  3. Train RF + XGBoost classifiers                   │
│  4. Train Ridge + XGBoost regressors                 │
│  5. Evaluate & save models                           │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               dashboard/app.py                       │
│  Streamlit dashboard with maps & predictions         │
└─────────────────────────────────────────────────────┘
```

---

## Scheduling Quarterly Updates

### Manual Update Process
1. Re-download latest data files from each source
2. Replace files in `data/raw/`
3. Run `python process_real_data.py`
4. Run `python train_models.py`

### Using Cron (macOS/Linux)
```bash
crontab -e

# Run quarterly on the 1st of Jan, Apr, Jul, Oct
0 6 1 1,4,7,10 * cd /Users/sarvesh0955/College/bda && source venv/bin/activate && python process_real_data.py && python train_models.py >> pipeline_runs.log 2>&1
```

---

## Outputs

| File | Location | Description |
|------|----------|-------------|
| `drought_water_stress.csv` | `data/processed/` | Consolidated dataset (18,216 rows × 39 cols) |
| `aqueduct_cleaned.csv` | `data/processed/` | Cleaned Aqueduct (228 countries) |
| `aquastat_cleaned.csv` | `data/processed/` | Cleaned AQUASTAT (74 countries, 1960–2022) |
| `grace_cleaned.csv` | `data/processed/` | Cleaned GRACE (72 countries, 2002–2026) |
| `drought_classifier_*.joblib` | `models/` | Trained classification models |
| `stress_predictor_*.joblib` | `models/` | Trained regression models |
| `pipeline_runs.log` | Project root | Pipeline run history |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `source venv/bin/activate && pip install -r requirements.txt` |
| `FileNotFoundError` on raw data | Ensure datasets are downloaded to `data/raw/` (see extraction_workflow.md) |
| AQUASTAT encoding error | File must use latin-1 encoding; re-download if corrupted |
| GRACE NetCDF read error | Install `netCDF4` and `xarray`: `pip install netCDF4 xarray` |
| Model training fails | Ensure `data/processed/drought_water_stress.csv` exists |
| Streamlit errors | Run from project root: `cd bda && streamlit run dashboard/app.py` |
