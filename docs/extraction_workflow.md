# Data Extraction Workflow

## Overview
This document describes the process for extracting water stress and drought data from three primary sources:
1. **WRI Aqueduct 4.0** — Global water risk indicators (catchment level)
2. **FAO AQUASTAT** — Country-level water resources and withdrawal data
3. **NASA GRACE/GRACE-FO** — Satellite-derived Terrestrial Water Storage (TWS) anomalies

---

## 1. WRI Aqueduct 4.0

### Source
- **Organization:** World Resources Institute (WRI)
- **Dataset:** Aqueduct Water Risk Atlas 4.0
- **License:** Creative Commons Attribution 4.0 International
- **URL:** https://www.wri.org/aqueduct
- **GitHub:** https://github.com/wri/Aqueduct40

### Access Method
**Bulk CSV download** from WRI's data portal. No authentication required.

```python
from src.extractors.aqueduct_extractor import AqueductExtractor

extractor = AqueductExtractor()
extractor.download_data("baseline_annual")  # Downloads ~50MB CSV
extractor.download_data("country_rankings")
```

### Key Indicators Extracted
| Indicator | Column | Scale | Description |
|-----------|--------|-------|-------------|
| Baseline Water Stress | `bws_score` | 0–5 | Ratio of water withdrawal to supply |
| Water Depletion | `bwd_score` | 0–5 | Ratio of consumption to supply |
| Interannual Variability | `iav_score` | 0–5 | Year-to-year variation in supply |
| Seasonal Variability | `sev_score` | 0–5 | Within-year variation in supply |
| Groundwater Decline | `gtd_score` | 0–5 | Rate of groundwater table decline |
| Drought Risk | `drr_score` | 0–5 | Likelihood and severity of drought |
| Flood Risk | `rfr_score` | 0–5 | Riverine/coastal flood risk |

### Data Schema
- **Spatial Resolution:** HydroBASINS level-6 catchments (~250K globally)
- **Temporal:** Baseline (1979–2019 average), no time series
- **Geography:** Global, per-catchment with lat/lon

### Troubleshooting
- **Download fails:** The extractor auto-generates realistic synthetic data as fallback
- **Large file size:** The annual baseline CSV can be ~50MB; ensure adequate disk space
- **Column naming:** Aqueduct 4.0 changed column names from v3.0; the extractor handles both

---

## 2. FAO AQUASTAT

### Source
- **Organization:** Food and Agriculture Organization (FAO)
- **Dataset:** AQUASTAT Database
- **URL:** https://data.apps.fao.org/aquastat/
- **Custodian for:** SDG 6.4.2 (Level of Water Stress)

### Access Method
**HTTP CSV download** from the AQUASTAT Dissemination Platform.

```python
from src.extractors.aquastat_extractor import AquastatExtractor

extractor = AquastatExtractor()
extractor.download_data()  # Downloads bulk CSV
```

### Key Variables
| Variable | ID | Unit | Description |
|----------|-----|------|-------------|
| SDG 6.4.2 Water Stress | 4263 | % | Freshwater withdrawal / available resources |
| Total Renewable Water | 4157 | 10⁹ m³/yr | Total renewable freshwater resources |
| Total Water Withdrawal | 4250 | 10⁹ m³/yr | All water withdrawals |
| Agricultural Withdrawal | 4251 | 10⁹ m³/yr | Water used for agriculture |
| Industrial Withdrawal | 4252 | 10⁹ m³/yr | Water used for industry |
| Municipal Withdrawal | 4253 | 10⁹ m³/yr | Domestic water use |
| Precipitation | 4101 | mm/yr | Average annual precipitation |

### Data Schema
- **Format:** Long format (one row per country × variable × year)
- **Temporal:** 5-year intervals (1960–2020); interpolated to annual
- **Geography:** Country-level (ISO-3 codes)

### Troubleshooting
- **Data format:** AQUASTAT uses long-format; the extractor pivots to wide-format automatically
- **Sparse time series:** Data available at ~5-year intervals; pipeline interpolates linearly
- **Manual download alternative:** Visit https://data.apps.fao.org/aquastat/ → filter → CSV export

---

## 3. NASA GRACE / GRACE-FO

### Source
- **Agency:** NASA / German Aerospace Center (DLR)
- **Mission:** GRACE (2002–2017) → GRACE-FO (2018–present)
- **Product:** JPL Mascon CRI-filtered (RL06.3 v04)
- **Collection:** `TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4`
- **URL:** https://grace.jpl.nasa.gov/
- **PO.DAAC:** https://podaac.jpl.nasa.gov/

### Access Method
**`podaac-data-downloader`** CLI tool (recommended by NASA).

#### Prerequisites
1. Create a free NASA Earthdata account: https://urs.earthdata.nasa.gov/
2. Install the downloader:
```bash
pip install podaac-data-subscriber
```

#### Download Commands
```bash
# Basic download (all .nc files)
podaac-data-downloader \
    -c TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 \
    -d ./data/raw/grace \
    --start-date 2002-04-04T00:00:00Z \
    --end-date 2024-12-31T00:00:00Z \
    -e .nc

# Download with spatial bounds
podaac-data-downloader \
    -c TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 \
    -d ./data/raw/grace \
    --start-date 2002-04-04T00:00:00Z \
    --end-date 2024-12-31T00:00:00Z \
    -b="-180,-90,180,90"

# Subscribe to recent data (last 24 hours)
podaac-data-subscriber \
    -c TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 \
    -d ./data/raw/grace \
    -m 1440
```

#### Python Usage
```python
from src.extractors.grace_extractor import GraceExtractor

extractor = GraceExtractor()
extractor.download_data()  # Uses podaac-data-downloader internally
df = extractor.extract_features()
```

### Key Variables
| Variable | netCDF Name | Unit | Description |
|----------|-------------|------|-------------|
| TWS Anomaly | `lwe_thickness` | cm EWH | Liquid Water Equivalent thickness anomaly |
| Uncertainty | `uncertainty` | cm | Measurement uncertainty |

### Derived Features
- `tws_3month_avg`, `tws_6month_avg`, `tws_12month_avg`: Rolling averages
- `tws_rate_of_change`: Monthly derivative
- `tws_cumulative_change`: Cumulative anomaly from baseline
- `groundwater_anomaly_cm`: Estimated groundwater component

### Data Schema
- **Format:** netCDF-4 (converted to tabular by extractor)
- **Spatial Resolution:** 0.5° × 0.5° global grid (~8,000 mascon cells)
- **Temporal:** Monthly (with some gaps during satellite transitions)
- **Coverage:** April 2002 – present

### Troubleshooting
- **Authentication errors:** Ensure `.netrc` file has Earthdata credentials
- **No podaac-data-downloader:** Extractor falls back to synthetic data
- **netCDF reading errors:** Install `netCDF4` and `xarray`: `pip install netCDF4 xarray`
- **Data gap (2017-06 to 2018-05):** Transition period between GRACE and GRACE-FO; filled by interpolation

---

## Pipeline Summary

```
WRI Aqueduct (CSV) ──→ aqueduct_extractor.py ──→ aqueduct_cleaned.csv
                                                          │
FAO AQUASTAT (CSV) ──→ aquastat_extractor.py ──→ aquastat_cleaned.csv ──→ data_merger.py ──→ drought_water_stress.csv
                                                          │
NASA GRACE (netCDF) ──→ grace_extractor.py ──→ grace_cleaned.csv
```

## Running the Full Pipeline

```bash
# Option 1: Run notebooks sequentially
jupyter notebook notebooks/01_data_extraction_aqueduct.ipynb
jupyter notebook notebooks/02_data_extraction_aquastat.ipynb
jupyter notebook notebooks/03_data_extraction_grace.ipynb
jupyter notebook notebooks/04_data_consolidation.ipynb

# Option 2: Run the automated pipeline
python -m src.pipeline.quarterly_pipeline
```
