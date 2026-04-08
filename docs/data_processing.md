# Data Processing Documentation

## EDA-15: Water Stress & Drought Index Tracker

---

## 1. Overview

This document describes the complete data processing pipeline that transforms raw, multi-source climate and hydrological data into a clean, consolidated dataset for drought risk analysis and ML modeling.

### Pipeline Summary

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  WRI Aqueduct 4.0 │     │   FAO AQUASTAT    │     │   NASA GRACE     │
│  (CSV, 193 MB)   │     │  (CSV, 936K rows) │     │  (NetCDF, 43 MB) │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
    ┌────▼─────┐            ┌────▼─────┐            ┌────▼─────┐
    │  Clean   │            │  Clean   │            │  Clean   │
    │  Rename  │            │  Filter  │            │  Extract │
    │  Agg     │            │  Pivot   │            │  Derive  │
    └────┬─────┘            └────┬─────┘            └────┬─────┘
         │                        │                        │
         ▼                        ▼                        ▼
  aqueduct_cleaned.csv    aquastat_cleaned.csv     grace_cleaned.csv
    (228 countries)        (4,500 records)         (18,216 records)
         │                        │                        │
         └────────────┬───────────┴────────────────────────┘
                      │
               ┌──────▼──────┐
               │    MERGE    │
               │  Composite  │
               │   Score     │
               └──────┬──────┘
                      │
                      ▼
         drought_water_stress.csv
         (18,216 rows × 39 cols)
```

### Script

The processing is performed by [`process_real_data.py`](../process_real_data.py), which can be run as:

```bash
source venv/bin/activate
python process_real_data.py
```

---

## 2. Data Sources

### 2.1 WRI Aqueduct 4.0 (Water Risk Atlas)

| Property | Value |
|----------|-------|
| **Source** | [World Resources Institute](https://www.wri.org/data/aqueduct-global-maps-40-data) |
| **Raw File** | `data/raw/aqueduct/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/Aqueduct40_baseline_annual_y2023m07d05.csv` |
| **Raw Size** | 193 MB |
| **Raw Dimensions** | 68,510 rows × 231 columns |
| **Granularity** | Sub-basin (HydroBASINS level) per country |
| **Temporal** | Baseline (static snapshot, no time series) |
| **License** | CC BY 4.0 |

**What it provides:**  
13 water risk indicators covering quantity, quality, and reputational concerns — including baseline water stress (BWS), water depletion (BWD), drought risk (DRR), groundwater table decline (GTD), and overall water risk scores.

### 2.2 FAO AQUASTAT

| Property | Value |
|----------|-------|
| **Source** | [FAO AQUASTAT Dissemination Platform](https://data.apps.fao.org/aquastat/) |
| **Raw File** | `data/raw/aquastat/bulk_eng(in).csv` |
| **Raw Dimensions** | 936,332 rows × 9 columns |
| **Format** | Long format (one row per country × variable × year) |
| **Encoding** | Latin-1 (contains special characters like "Türkiye") |
| **Temporal** | 1960–2022 (5-year survey intervals, interpolated) |

**What it provides:**  
National-level water resource statistics including SDG 6.4.2 (water stress %), total renewable water resources, water withdrawals by sector (agricultural, industrial, municipal), precipitation, and dam capacity.

### 2.3 NASA GRACE / GRACE-FO

| Property | Value |
|----------|-------|
| **Source** | [NASA PO.DAAC](https://podaac.jpl.nasa.gov/) — TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 |
| **Raw File** | `data/raw/grace/1.nc` |
| **Raw Size** | 43 MB |
| **Format** | NetCDF4 |
| **Variables** | `lwe_thickness` (TWS anomaly), `uncertainty`, `land_mask`, `scale_factor` |
| **Dimensions** | 253 time steps × 360 lat × 720 lon (0.5° grid) |
| **Temporal** | April 2002 – January 2026 (monthly) |

**What it provides:**  
Monthly Terrestrial Water Storage (TWS) anomalies measured by the GRACE and GRACE-FO satellite missions. TWS captures the integrated change in water stored on and below Earth's surface (groundwater, soil moisture, surface water, snow/ice).

---

## 3. Processing Steps

### 3.1 Aqueduct Processing (`process_aqueduct()`)

**Input:** 68,510 sub-basin rows × 231 columns  
**Output:** 228 country-level rows × 16 columns

#### Steps:

1. **Load raw CSV** — Read the full 193 MB Aqueduct baseline annual file.

2. **Column Renaming** — Map original Aqueduct column names to standardized names:

   | Original | Renamed |
   |----------|---------|
   | `bws_score` | `water_stress_score` |
   | `bwd_score` | `water_depletion_score` |
   | `drr_score` | `drought_risk_score` |
   | `gtd_score` | `groundwater_decline_score` |
   | `rfr_score` | `flood_risk_score` |
   | `w_awr_def_tot_score` | `overall_water_risk` |

3. **Drop NaN countries** — Remove rows where country name (`name_0`) is missing.

4. **Sentinel value handling** — Aqueduct uses `9999` as a sentinel value meaning "Arid and Low Water Use". These are replaced with `NaN` to prevent skewing aggregations.

5. **Score clipping** — All `_score` columns are clipped to the valid [0, 5] range.

6. **Missing value imputation** — Two-pass strategy:
   - First: fill with **country median** (captures regional patterns)
   - Then: fill remaining with **global median** (ensures no NaN)

7. **Country-level aggregation** — Sub-basin records are aggregated to country level using `mean` for all numeric indicators (unweighted). This produces one row per country with averaged water risk scores.

8. **Rounding** — All numeric columns rounded to 4 decimal places.

#### Output Schema:

| Column | Type | Description |
|--------|------|-------------|
| `country` | str | Country name |
| `country_code` | str | ISO 3166-1 alpha-3 code |
| `water_stress_score` | float | Baseline Water Stress (0–5) |
| `water_stress_raw` | float | Raw water stress ratio |
| `water_depletion_score` | float | Baseline Water Depletion (0–5) |
| `groundwater_decline_score` | float | Groundwater Table Decline (0–5) |
| `drought_risk_score` | float | Drought Risk (0–5) |
| `flood_risk_score` | float | Riverine Flood Risk (0–5) |
| `overall_water_risk` | float | Overall Water Risk (0–5) |
| `source` | str | Always "WRI_Aqueduct_4.0" |

---

### 3.2 AQUASTAT Processing (`process_aquastat()`)

**Input:** 936,332 rows × 9 columns (long format)  
**Output:** 4,500 rows × 16 columns (wide format, one row per country × year)

#### Steps:

1. **Load with encoding** — Read CSV using `latin-1` encoding to handle special characters (e.g., "Türkiye", "Côte d'Ivoire").

2. **Column standardization** — The raw AQUASTAT uses non-standard column names. These are mapped as:

   | Raw Position | Renamed |
   |-------------|---------|
   | Column 0 | `variable_id` |
   | Column 1 | `variable` |
   | Column 2 | `area_id` |
   | Column 3 | `country` |
   | Column 5 | `year` |
   | Column 6 | `value` |

3. **Type coercion** — `value` and `year` converted to numeric; rows with invalid values dropped.

4. **Variable filtering** — From 199 AQUASTAT variables, 12 key water-related variables are selected using fuzzy string matching:

   | AQUASTAT Variable | Clean Name | Records |
   |------------------|------------|---------|
   | Total renewable water resources | `total_renewable_water_km3` | 29,514 |
   | Total water withdrawal | `total_water_withdrawal_km3` | 25,984 |
   | Precipitation | `precipitation_mm` | 20,978 |
   | Municipal water withdrawal | `municipal_withdrawal_km3` | 19,665 |
   | Industrial water withdrawal | `industrial_withdrawal_km3` | 12,936 |
   | Agricultural water withdrawal | `agricultural_withdrawal_km3` | 12,784 |
   | Dependency ratio | `dependency_ratio_pct` | 11,128 |
   | Total dam capacity | `dam_capacity_km3` | 8,884 |
   | National Rainfall Index | `rainfall_index` | 8,595 |
   | SDG 6.4.2 Water Stress | `sdg642_water_stress_pct` | 6,685 |
   | Safe drinking water access | `safe_water_access_pct` | 6,222 |
   | Flood occurrence | `flood_occurrence` | 1,760 |

5. **ISO-3 country mapping** — Country names are mapped to ISO 3166-1 alpha-3 codes using a manually-curated dictionary (handles FAO-specific naming conventions like "Iran (Islamic Republic of)" → "IRN").

6. **Pivot long → wide** — Transforms from long format (one row per variable) to wide format (one row per country-year, each variable becomes a column).

7. **Temporal interpolation** — Missing years within each country are filled using linear interpolation (bidirectional). AQUASTAT data is collected in ~5-year intervals, so interpolation provides estimations for intervening years.

8. **Sorting** — Output sorted by country, then year.

#### Output Schema:

| Column | Type | Null % | Description |
|--------|------|--------|-------------|
| `country` | str | 0% | Country name |
| `country_code` | str | varies | ISO-3 code |
| `year` | int | 0% | Year (1960–2022) |
| `sdg642_water_stress_pct` | float | varies | SDG 6.4.2 water stress (%) |
| `total_renewable_water_km3` | float | varies | Total renewable water (km³/yr) |
| `total_water_withdrawal_km3` | float | varies | Total water withdrawal (km³/yr) |
| `agricultural_withdrawal_km3` | float | varies | Agricultural withdrawal (km³/yr) |
| `precipitation_mm` | float | varies | Annual precipitation (mm) |
| `source` | str | 0% | Always "FAO_AQUASTAT" |

---

### 3.3 GRACE Processing (`process_grace()`)

**Input:** NetCDF file (253 × 360 × 720)  
**Output:** 18,216 rows × 16 columns (country-level monthly time series)

#### Steps:

1. **Open NetCDF** — Read the GRACE MASCON dataset using `xarray`. Key variables:
   - `lwe_thickness`: TWS anomaly in cm (shape: 253 × 360 × 720)
   - `uncertainty`: Measurement uncertainty in cm
   - `time`: 253 monthly time steps (April 2002 – January 2026)

2. **Country centroid extraction** — 72 predefined country centroids are used to extract TWS values at the nearest grid cell. Each centroid is matched to the closest lat/lon index in the 0.5° GRACE grid.

   > **Note:** A full spatial aggregation using country boundaries (via `geopandas`) is preferred but requires downloading Natural Earth shapefiles. The centroid method provides representative values for each country.

3. **Quality filtering** — NaN values and fill values (> 10¹⁰) are discarded.

4. **Derived features** (per country time series):

   | Feature | Formula | Purpose |
   |---------|---------|---------|
   | `tws_3month_avg` | Rolling 3-month mean | Short-term smoothing |
   | `tws_6month_avg` | Rolling 6-month mean | Medium-term trend |
   | `tws_12month_avg` | Rolling 12-month mean | Annual trend |
   | `tws_rate_of_change` | Month-to-month difference | Velocity of change |
   | `tws_cumulative_change` | TWS − first observation | Long-term trajectory |
   | `tws_zscore` | (TWS − μ) / σ per country | Standardized anomaly |
   | `groundwater_anomaly_cm` | TWS × 0.4 | Estimated GW component |
   | `tws_trend_cm_yr` | Linear regression slope × 12 | Annual trend rate |

#### Output Schema:

| Column | Type | Description |
|--------|------|-------------|
| `country` | str | Country name |
| `country_code` | str | ISO-3 code |
| `lat`, `lon` | float | Country centroid coordinates |
| `date` | str | Year-month (YYYY-MM) |
| `tws_anomaly_cm` | float | TWS anomaly in cm (relative to 2004–2009 mean) |
| `uncertainty_cm` | float | GRACE measurement uncertainty in cm |
| `tws_*` | float | Derived temporal features (see above) |
| `source` | str | Always "NASA_GRACE" |

---

### 3.4 Merging & Consolidation (`merge_and_consolidate()`)

**Output:** 18,216 rows × 39 columns

#### Merge Strategy:

1. **GRACE as backbone** — The GRACE monthly time series provides the temporal backbone (monthly resolution, 253 time steps per country).

2. **Aqueduct join** (LEFT JOIN on `country_code`) — Static water risk scores are replicated across all months for each country. This adds baseline water stress context to each time step.

3. **AQUASTAT join** (LEFT JOIN on `country_code` + `year`) — Annual water resource statistics are matched to the corresponding year in the GRACE time series. Missing years result in NaN (17.6% null rate for AQUASTAT columns).

#### Composite Drought Score

A weighted composite score is computed from multiple indicators:

```
drought_composite_score = (0.25 × water_stress_score) +
                          (0.25 × drought_risk_score) +
                          (0.30 × tws_score_normalized) +
                          (0.20 × groundwater_decline_score)
```

Where `tws_score_normalized` maps TWS anomaly to a 0–5 scale:
- More negative TWS (water loss) → higher score (more drought)
- Normalized using 1st and 99th percentile as bounds, clipped to [0, 5]

#### Drought Risk Classification

| Class | Score Range | Records | Share |
|-------|-----------|---------|-------|
| **Low** | 0.0 – 1.0 | 563 | 3.1% |
| **Moderate** | 1.0 – 2.0 | 8,471 | 46.5% |
| **High** | 2.0 – 3.5 | 9,182 | 50.4% |
| **Extreme** | 3.5 – 5.0 | 0 | 0.0% |

#### Season Assignment

Each record is tagged with a meteorological season based on the month:

| Season | Months |
|--------|--------|
| DJF (Winter) | December, January, February |
| MAM (Spring) | March, April, May |
| JJA (Summer) | June, July, August |
| SON (Autumn) | September, October, November |

---

## 4. Output Datasets

### 4.1 Consolidated Dataset

**File:** `data/processed/drought_water_stress.csv`  
**Size:** 5.0 MB  
**Shape:** 18,216 rows × 39 columns

#### Complete Column Reference

| # | Column | Type | Null % | Source | Description |
|---|--------|------|--------|--------|-------------|
| 1 | `country` | str | 0% | All | Country name |
| 2 | `country_code` | str | 0% | All | ISO-3 code |
| 3 | `lat` | float | 0% | GRACE | Latitude of centroid |
| 4 | `lon` | float | 0% | GRACE | Longitude of centroid |
| 5 | `date` | str | 0% | GRACE | Year-month (YYYY-MM) |
| 6 | `year` | int | 0% | Derived | Year extracted from date |
| 7 | `season` | str | 0% | Derived | Meteorological season (DJF/MAM/JJA/SON) |
| 8 | `tws_anomaly_cm` | float | 0% | GRACE | Terrestrial Water Storage anomaly (cm) |
| 9 | `uncertainty_cm` | float | 0% | GRACE | GRACE measurement uncertainty (cm) |
| 10 | `groundwater_anomaly_cm` | float | 0% | Derived | Estimated GW anomaly (TWS × 0.4) |
| 11 | `tws_3month_avg` | float | 0% | Derived | 3-month rolling mean of TWS |
| 12 | `tws_6month_avg` | float | 0% | Derived | 6-month rolling mean of TWS |
| 13 | `tws_12month_avg` | float | 0% | Derived | 12-month rolling mean of TWS |
| 14 | `tws_rate_of_change` | float | 0.4% | Derived | Month-to-month TWS change |
| 15 | `tws_cumulative_change` | float | 0% | Derived | Cumulative change from first obs |
| 16 | `tws_zscore` | float | 0% | Derived | Z-score of TWS anomaly per country |
| 17 | `tws_trend_cm_yr` | float | 0% | Derived | Linear TWS trend (cm/year) |
| 18 | `water_stress_score` | float | 0% | Aqueduct | Baseline Water Stress score (0–5) |
| 19 | `water_stress_raw` | float | 0% | Aqueduct | Raw water stress ratio |
| 20 | `water_stress_category` | float | 0% | Aqueduct | Stress category (0–4) |
| 21 | `water_depletion_score` | float | 0% | Aqueduct | Baseline Water Depletion (0–5) |
| 22 | `water_depletion_raw` | float | 0% | Aqueduct | Raw depletion ratio |
| 23 | `interannual_variability` | float | 0% | Aqueduct | Interannual variability score |
| 24 | `seasonal_variability` | float | 0% | Aqueduct | Seasonal variability score |
| 25 | `groundwater_decline_score` | float | 0% | Aqueduct | GW table decline score (0–5) |
| 26 | `drought_risk_score` | float | 0% | Aqueduct | Drought Risk score (0–5) |
| 27 | `drought_risk_raw` | float | 0% | Aqueduct | Raw drought risk |
| 28 | `flood_risk_score` | float | 0% | Aqueduct | Riverine Flood Risk (0–5) |
| 29 | `overall_water_risk` | float | 0% | Aqueduct | Overall Water Risk (0–5) |
| 30 | `sdg642_water_stress_pct` | float | 17.6% | AQUASTAT | SDG 6.4.2 Water Stress (%) |
| 31 | `total_renewable_water_km3` | float | 17.6% | AQUASTAT | Total renewable water (km³/yr) |
| 32 | `total_water_withdrawal_km3` | float | 17.6% | AQUASTAT | Total water withdrawal (km³/yr) |
| 33 | `agricultural_withdrawal_km3` | float | 17.6% | AQUASTAT | Agricultural use (km³/yr) |
| 34 | `industrial_withdrawal_km3` | float | 17.6% | AQUASTAT | Industrial use (km³/yr) |
| 35 | `municipal_withdrawal_km3` | float | 17.6% | AQUASTAT | Municipal use (km³/yr) |
| 36 | `precipitation_mm` | float | 17.6% | AQUASTAT | Annual precipitation (mm) |
| 37 | `drought_composite_score` | float | 0% | Derived | Weighted composite (0–5) |
| 38 | `drought_risk_class` | str | 0% | Derived | Low / Moderate / High / Extreme |
| 39 | `source` | str | 0% | — | Source dataset identifier |

### 4.2 Individual Cleaned Datasets

| File | Size | Rows | Countries | Features |
|------|------|------|-----------|----------|
| `aqueduct_cleaned.csv` | 31 KB | 228 | 228 | 16 |
| `aquastat_cleaned.csv` | 538 KB | 4,500 | 74 | 16 |
| `grace_cleaned.csv` | 2,063 KB | 18,216 | 72 | 16 |
| **drought_water_stress.csv** | **5,109 KB** | **18,216** | **72** | **39** |

---

## 5. Data Quality

### 5.1 Null Values

| Category | Columns | Null % | Reason |
|----------|---------|--------|--------|
| Core (GRACE + Aqueduct) | TWS, water stress, drought risk | **0%** | Fully populated |
| Temporal | `tws_rate_of_change` | **0.4%** | First record per country has no prior month |
| AQUASTAT | All AQUASTAT columns | **17.6%** | Years without survey data |

### 5.2 Sentinel Value Handling

Aqueduct uses `9999` to indicate "Arid and Low Water Use" zones. These are:
- Replaced with `NaN` before aggregation
- Filled with country median → global median
- Not treated as valid high-stress indicators

### 5.3 Statistical Summary (Key Variables)

| Variable | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| `tws_anomaly_cm` | 0.78 | 12.28 | −45.05 | 116.95 |
| `uncertainty_cm` | 2.42 | 1.65 | 0.55 | 36.02 |
| `water_stress_score` | 2.12 | 1.47 | 0.00 | 4.85 |
| `drought_risk_score` | 2.12 | 0.91 | 0.00 | 3.85 |
| `drought_composite_score` | 2.00 | 0.53 | 0.45 | 3.40 |

### 5.4 Top 10 Most Drought-Stressed Countries

| Rank | Country | Water Stress | Drought Risk | Composite Score |
|------|---------|-------------|--------------|-----------------|
| 1 | Syria | 3.98 | 3.51 | 2.98 |
| 2 | Israel | 4.48 | 2.42 | 2.94 |
| 3 | Iran | 3.92 | 2.59 | 2.86 |
| 4 | Tunisia | 3.90 | 2.74 | 2.79 |
| 5 | Yemen | 4.69 | 2.03 | 2.78 |
| 6 | Lebanon | 4.36 | 3.15 | 2.77 |
| 7 | Saudi Arabia | 4.85 | 0.82 | 2.75 |
| 8 | Jordan | 4.22 | 1.56 | 2.67 |
| 9 | South Africa | 3.67 | 2.53 | 2.52 |
| 10 | Morocco | 2.94 | 3.19 | 2.51 |

---

## 6. Known Limitations

1. **GRACE spatial resolution** — TWS values are extracted at country centroid points (single grid cell per country). Large countries (e.g., Russia, USA, China) have significant sub-national variability not captured.

2. **Aqueduct is static** — Aqueduct 4.0 provides baseline scores without temporal dimension. The same score is replicated across all months in the time series.

3. **AQUASTAT survey intervals** — Data is collected roughly every 5 years. Linear interpolation fills gaps but may not capture rapid changes (e.g., dam construction, policy shifts).

4. **GRACE gap (2017–2018)** — The transition between GRACE and GRACE-FO introduces a ~11 month data gap. The MASCON product handles this internally, but fewer observations exist during this period.

5. **Country naming** — AQUASTAT uses FAO naming conventions (e.g., "Iran (Islamic Republic of)") which require manual mapping to ISO-3 codes. Some edge-case countries may be lost in the mapping.

---

## 7. Reproducibility

### Requirements

```
pandas >= 2.0
numpy >= 1.24
xarray >= 2023.0
netCDF4
geopandas (optional, for spatial aggregation)
```

### Running the pipeline

```bash
cd /path/to/bda
source venv/bin/activate
python process_real_data.py
```

Execution time: ~15 seconds on a modern machine.

### Input files required

```
data/raw/aqueduct/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/
    Aqueduct40_baseline_annual_y2023m07d05.csv

data/raw/aquastat/
    bulk_eng(in).csv

data/raw/grace/
    1.nc
```

---

## 8. Changelog

| Date | Description |
|------|-------------|
| 2026-04-03 | Initial processing with real downloaded datasets |
| 2026-04-03 | Trimmed consolidated output from 201 to 39 essential columns |
| 2026-04-03 | Extended GRACE coverage from 40 to 72 country centroids |
