# Data Extraction Workflow

## Overview

This document describes the data sources and manual download process for the Water Stress & Drought Index Tracker. All three datasets are **manually downloaded** from their respective portals and placed in the `data/raw/` directory.

---

## 1. WRI Aqueduct 4.0

### Source
- **Organization:** World Resources Institute (WRI)
- **Dataset:** Aqueduct Water Risk Atlas 4.0
- **License:** Creative Commons Attribution 4.0 International
- **URL:** https://www.wri.org/data/aqueduct-global-maps-40-data
- **GitHub:** https://github.com/wri/Aqueduct40

### Download Instructions

1. Visit: https://www.wri.org/data/aqueduct-global-maps-40-data
2. Click **"Download the dataset directly here"** (ZIP file, ~260 MB)
3. Extract the ZIP to `data/raw/aqueduct/`

Expected directory structure after extraction:
```
data/raw/aqueduct/
└── Aqueduct40_waterrisk_download_Y2023M07D05/
    ├── CVS/
    │   ├── Aqueduct40_baseline_annual_y2023m07d05.csv   ← Primary file (193 MB)
    │   ├── Aqueduct40_baseline_monthly_y2023m07d05.csv
    │   └── Aqueduct40_future_annual_y2023m07d05.csv
    ├── GDB/
    └── Aqueduct40_README.xlsx
```

### Key Indicators
| Indicator | Column | Scale | Description |
|-----------|--------|-------|-------------|
| Baseline Water Stress | `bws_score` | 0–5 | Ratio of water withdrawal to supply |
| Water Depletion | `bwd_score` | 0–5 | Ratio of consumption to supply |
| Interannual Variability | `iav_score` | 0–5 | Year-to-year variation in supply |
| Seasonal Variability | `sev_score` | 0–5 | Within-year variation in supply |
| Groundwater Decline | `gtd_score` | 0–5 | Rate of groundwater table decline |
| Drought Risk | `drr_score` | 0–5 | Likelihood and severity of drought |
| Flood Risk | `rfr_score` | 0–5 | Riverine/coastal flood risk |
| Overall Water Risk | `w_awr_def_tot_score` | 0–5 | Composite risk score |

### Data Characteristics
- **Spatial Resolution:** HydroBASINS level-6 catchments (~68,510 rows globally)
- **Temporal:** Baseline (1979–2019 average), no time series
- **Total Columns:** 231 (13 indicators × multiple representations + metadata)
- **Sentinel Value:** `9999` indicates "Arid and Low Water Use" zones

---

## 2. FAO AQUASTAT

### Source
- **Organization:** Food and Agriculture Organization (FAO)
- **Dataset:** AQUASTAT Database
- **URL:** https://data.apps.fao.org/aquastat/
- **Custodian for:** SDG 6.4.2 (Level of Water Stress)

### Download Instructions

1. Visit: https://data.apps.fao.org/aquastat/
2. Close the welcome splash screen
3. Click **"Share / Download"** in the top toolbar
4. Select **"Bulk Download"** → choose **CSV** format
5. Save as `data/raw/aquastat/bulk_eng(in).csv`

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

### Data Characteristics
- **Format:** Long format (one row per country × variable × year)
- **Encoding:** Latin-1 (special characters like Türkiye)
- **Rows:** ~936,000 (199 unique variables)
- **Temporal:** 5-year intervals (1960–2022)
- **Geography:** Country-level

---

## 3. NASA GRACE / GRACE-FO

### Source
- **Agency:** NASA / German Aerospace Center (DLR)
- **Mission:** GRACE (2002–2017) → GRACE-FO (2018–present)
- **Product:** JPL Mascon CRI-filtered (RL06.3 v04)
- **Collection:** `TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4`
- **URL:** https://podaac.jpl.nasa.gov/

### Download Instructions

#### Option A: PO.DAAC Web Portal (no account needed for browse)
1. Visit: https://podaac.jpl.nasa.gov/
2. Search for `TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4`
3. Download the latest NetCDF file
4. Save as `data/raw/grace/1.nc`

#### Option B: podaac-data-downloader CLI
Requires a free NASA Earthdata account (https://urs.earthdata.nasa.gov/).

```bash
pip install podaac-data-subscriber

podaac-data-downloader \
    -c TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 \
    -d ./data/raw/grace \
    --start-date 2002-04-04T00:00:00Z \
    --end-date 2026-01-31T00:00:00Z \
    -e .nc
```

### Key Variables
| Variable | netCDF Name | Unit | Description |
|----------|-------------|------|-------------|
| TWS Anomaly | `lwe_thickness` | cm EWH | Liquid Water Equivalent thickness anomaly |
| Uncertainty | `uncertainty` | cm | Measurement uncertainty |
| Land Mask | `land_mask` | — | Land/ocean indicator |

### Data Characteristics
- **Format:** NetCDF-4
- **Spatial Resolution:** 0.5° × 0.5° global grid (360 × 720)
- **Temporal:** Monthly (253 time steps: April 2002 – January 2026)
- **File Size:** ~43 MB
- **Gap:** ~11 months during GRACE → GRACE-FO transition (mid-2017 to mid-2018)

---

## Processing Pipeline

After downloading all files manually, run:

```bash
source venv/bin/activate
python process_real_data.py
```

This produces clean datasets in `data/processed/`:

```
data/raw/                                    data/processed/
├── aqueduct/...csv (193 MB)  ──────→  aqueduct_cleaned.csv  (31 KB, 228 countries)
├── aquastat/bulk_eng(in).csv ──────→  aquastat_cleaned.csv   (538 KB, 74 countries)
└── grace/1.nc (43 MB)        ──────→  grace_cleaned.csv      (2 MB, 72 countries)
                                              │
                                              ▼
                                    drought_water_stress.csv  (5 MB, 18,216 rows × 39 cols)
```

See [data_processing.md](data_processing.md) for detailed processing documentation.
