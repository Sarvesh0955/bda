"""
EDA-15: Water Stress & Drought Index Tracker
Central configuration for data sources, paths, and model parameters.
"""
import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─── Directory Paths ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_AQUEDUCT_DIR = RAW_DIR / "aqueduct"
RAW_AQUASTAT_DIR = RAW_DIR / "aquastat"
RAW_GRACE_DIR = RAW_DIR / "grace"

# Ensure directories exist
for d in [RAW_AQUEDUCT_DIR, RAW_AQUASTAT_DIR, RAW_GRACE_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Output Files ─────────────────────────────────────────────────────────────
CONSOLIDATED_CSV = PROCESSED_DIR / "drought_water_stress.csv"
AQUEDUCT_PROCESSED = PROCESSED_DIR / "aqueduct_cleaned.csv"
AQUASTAT_PROCESSED = PROCESSED_DIR / "aquastat_cleaned.csv"
GRACE_PROCESSED = PROCESSED_DIR / "grace_cleaned.csv"

# ─── WRI Aqueduct 4.0 ────────────────────────────────────────────────────────
# Baseline annual data (CSV) from WRI's data portal
AQUEDUCT_BASELINE_URL = (
    "https://raw.githubusercontent.com/wri/Aqueduct40/master/data/"
)
# Local raw file paths
AQUEDUCT_BASELINE_CSV = RAW_AQUEDUCT_DIR / "aqueduct40_baseline.csv"
AQUEDUCT_COUNTRY_CSV = RAW_AQUEDUCT_DIR / "aqueduct40_country_rankings.csv"

# Key indicators to extract (column names in Aqueduct 4.0)
AQUEDUCT_INDICATORS = [
    "bws_raw",     # Baseline Water Stress (raw ratio)
    "bws_score",   # Baseline Water Stress (score 0-5)
    "bws_cat",     # Baseline Water Stress (category)
    "bwd_raw",     # Baseline Water Depletion (raw ratio)
    "bwd_score",   # Baseline Water Depletion (score 0-5)
    "iav_raw",     # Interannual Variability (raw)
    "iav_score",   # Interannual Variability (score)
    "sev_raw",     # Seasonal Variability (raw)
    "sev_score",   # Seasonal Variability (score)
    "gtd_raw",     # Groundwater Table Decline (raw)
    "gtd_score",   # Groundwater Table Decline (score)
    "rfr_raw",     # Riverine Flood Risk (raw)
    "rfr_score",   # Riverine Flood Risk (score)
    "cfr_raw",     # Coastal Flood Risk (raw)
    "cfr_score",   # Coastal Flood Risk (score)
    "drr_raw",     # Drought Risk (raw)
    "drr_score",   # Drought Risk (score)
]

# ─── FAO AQUASTAT ─────────────────────────────────────────────────────────────
# AQUASTAT Dissemination Platform bulk CSV download
AQUASTAT_BASE_URL = "https://data.apps.fao.org/aquastat/"
AQUASTAT_CSV_URL = (
    "https://data.apps.fao.org/aquastat/data/csv"
)
AQUASTAT_RAW_CSV = RAW_AQUASTAT_DIR / "aquastat_bulk.csv"

# SDG 6.4.2 and key variables
AQUASTAT_VARIABLES = [
    "SDG 6.4.2. Water Stress",
    "Total renewable water resources",
    "Total water withdrawal",
    "Agricultural water withdrawal",
    "Industrial water withdrawal",
    "Municipal water withdrawal",
    "Total renewable water resources per capita",
    "Dependency ratio",
]

# ─── NASA GRACE / GRACE-FO ───────────────────────────────────────────────────
# PO.DAAC Data Subscriber configuration
GRACE_COLLECTION = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"
GRACE_START_DATE = "2002-04-01T00:00:00Z"
GRACE_END_DATE = "2025-12-31T00:00:00Z"

# NASA Earthdata credentials (set via environment variables)
EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME", "")
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD", "")

# NetCDF variable names in GRACE Mascon data
GRACE_VARIABLES = {
    "lwe_thickness": "Liquid Water Equivalent Thickness (cm)",
    "uncertainty": "Measurement Uncertainty (cm)",
    "lat": "Latitude",
    "lon": "Longitude",
    "time": "Time",
}

# ─── Data Processing ──────────────────────────────────────────────────────────
# Temporal alignment
TARGET_TEMPORAL_RESOLUTION = "monthly"  # Options: monthly, quarterly, annual
DATE_FORMAT = "%Y-%m"

# Drought risk classification thresholds (based on composite score)
DROUGHT_RISK_THRESHOLDS = {
    "Low": (0.0, 1.0),
    "Moderate": (1.0, 2.0),
    "High": (2.0, 3.5),
    "Extreme": (3.5, 5.0),
}

# ISO country code mapping file (bundled with geopandas)
COUNTRY_SHAPES_RESOLUTION = "110m"  # Options: 10m, 50m, 110m

# ─── ML Model Configuration ──────────────────────────────────────────────────
# Train/test split
TEMPORAL_SPLIT_YEAR = 2018  # Train on data before this year

# Random Forest defaults
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost defaults
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "mlogloss",
}

# XGBoost Regressor defaults (for water stress prediction)
XGB_REG_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Feature engineering
LAG_PERIODS = [1, 3, 6, 12]  # Months of lag features
ROLLING_WINDOWS = [3, 6, 12]  # Rolling mean/std windows

# Cross-validation
CV_FOLDS = 5
SCORING_CLASSIFICATION = "f1_weighted"
SCORING_REGRESSION = "neg_root_mean_squared_error"

# ─── Streamlit Dashboard ─────────────────────────────────────────────────────
DASHBOARD_TITLE = "🌍 Water Stress & Drought Index Tracker"
DASHBOARD_ICON = "💧"
MAP_CENTER = [20.0, 0.0]  # Default map center (lat, lon)
MAP_ZOOM = 2

# ─── Pipeline ─────────────────────────────────────────────────────────────────
PIPELINE_LOG_FILE = PROJECT_ROOT / "pipeline_runs.log"
QUARTERLY_SCHEDULE = "quarterly"  # Cron: 0 0 1 1,4,7,10 *
