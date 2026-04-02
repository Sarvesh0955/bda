# Pipeline Guide

## Overview

The Quarterly Update Pipeline automates the full cycle of:
1. **Data download** from all 3 sources
2. **Cleaning and merging** into consolidated CSV
3. **ML model re-training** on expanded dataset
4. **Report generation**
5. **Run logging**

---

## Quick Start

### One-time Setup
```bash
# Clone and install dependencies
cd /Users/sarvesh0955/College/bda
pip install -r requirements.txt

# (For real GRACE data) Set Earthdata credentials
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

### Run the Pipeline
```bash
# Full pipeline (download + process + train + report)
python -m src.pipeline.quarterly_pipeline

# Skip download (use existing data)
python -c "
from src.pipeline.quarterly_pipeline import QuarterlyPipeline
pipeline = QuarterlyPipeline()
pipeline.run(skip_download=True)
"
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Pipeline Steps in Detail

### Step 1: Data Extraction
- Downloads WRI Aqueduct 4.0 CSV from WRI data portal
- Downloads FAO AQUASTAT bulk data via HTTP
- Downloads NASA GRACE netCDF files via `podaac-data-downloader`
- Falls back to synthetic data if downloads fail

### Step 2: Data Processing
- Cleans each dataset (outlier removal, missing value handling)
- Standardizes country codes across sources
- Merges on country_code + year/date
- Computes derived features (composite scores, z-scores, trends)
- Classifies drought risk (Low/Moderate/High/Extreme)

### Step 3: Model Training
- Re-trains Random Forest and XGBoost classifiers
- Re-trains Ridge and XGBoost regressors
- Saves updated models to `models/` directory

### Step 4: Report Generation
- Computes risk distribution statistics
- Identifies top-risk countries
- Logs all metrics

### Step 5: Logging
- Each run appended to `pipeline_runs.log` (JSON lines format)
- Includes: timestamp, elapsed time, record counts, model metrics

---

## Scheduling

### Using Cron (macOS/Linux)
Run quarterly on the 1st of January, April, July, October:

```bash
# Open crontab editor
crontab -e

# Add this line (adjusts path as needed):
0 6 1 1,4,7,10 * cd /Users/sarvesh0955/College/bda && /usr/bin/python3 -m src.pipeline.quarterly_pipeline >> pipeline.log 2>&1
```

### Using Python Schedule Library
```python
import schedule
import time
from src.pipeline.quarterly_pipeline import QuarterlyPipeline

def run_pipeline():
    pipeline = QuarterlyPipeline()
    pipeline.run()

# Run every quarter
schedule.every(90).days.do(run_pipeline)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EARTHDATA_USERNAME` | For real GRACE data | NASA Earthdata Login username |
| `EARTHDATA_PASSWORD` | For real GRACE data | NASA Earthdata Login password |

---

## Outputs

| File | Location | Description |
|------|----------|-------------|
| `drought_water_stress.csv` | `data/processed/` | Consolidated dataset |
| `aqueduct_cleaned.csv` | `data/processed/` | Cleaned Aqueduct data |
| `aquastat_cleaned.csv` | `data/processed/` | Cleaned AQUASTAT data |
| `grace_cleaned.csv` | `data/processed/` | Cleaned GRACE data |
| `drought_classifier_*.joblib` | `models/` | Trained classifier |
| `stress_predictor_*.joblib` | `models/` | Trained regressor |
| `pipeline_runs.log` | Project root | Pipeline run history |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Download timeouts | Increase timeout in `config.py` or use pre-downloaded data |
| GRACE auth fails | Verify `~/.netrc` has Earthdata credentials |
| Empty dataset | Check individual extractors with notebooks 01-03 |
| Model training fails | Ensure consolidated CSV exists (`data/processed/drought_water_stress.csv`) |
| Streamlit errors | Run from project root: `cd bda && streamlit run dashboard/app.py` |
