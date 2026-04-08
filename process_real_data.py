"""
EDA-15: Process REAL downloaded datasets
Reads actual WRI Aqueduct, FAO AQUASTAT, and NASA GRACE data,
cleans, merges, and outputs consolidated CSV.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import config


# ──────────────────────────────────────────────────────────────
# 1. AQUEDUCT (real CSV — 68,510 rows × 231 columns)
# ──────────────────────────────────────────────────────────────
def process_aqueduct():
    """Process real WRI Aqueduct 4.0 baseline annual CSV."""
    logger.info("=" * 60)
    logger.info("Processing WRI Aqueduct 4.0 data...")

    csv_path = (
        config.RAW_AQUEDUCT_DIR
        / "Aqueduct40_waterrisk_download_Y2023M07D05"
        / "CVS"
        / "Aqueduct40_baseline_annual_y2023m07d05.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Aqueduct CSV not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  Raw: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # Rename key columns
    df = df_raw.rename(columns={
        "name_0": "country", "gid_0": "country_code",
        "bws_raw": "water_stress_raw", "bws_score": "water_stress_score",
        "bws_cat": "water_stress_category", "bws_label": "water_stress_label",
        "bwd_raw": "water_depletion_raw", "bwd_score": "water_depletion_score",
        "iav_score": "interannual_variability", "sev_score": "seasonal_variability",
        "gtd_score": "groundwater_decline_score",
        "drr_raw": "drought_risk_raw", "drr_score": "drought_risk_score",
        "rfr_score": "flood_risk_score",
        "w_awr_def_tot_raw": "overall_water_risk_raw",
        "w_awr_def_tot_score": "overall_water_risk",
    })

    # Drop rows where country is NaN
    df = df.dropna(subset=["country"])

    # Replace 9999 sentinel with NaN
    score_cols = [c for c in df.columns if "_score" in c or "_raw" in c]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] >= 9999, col] = np.nan

    for col in [c for c in df.columns if "_score" in c]:
        if col in df.columns:
            df[col] = df[col].clip(0, 5)

    # Fill NaN with country median, then global median
    for col in score_cols:
        if col in df.columns:
            df[col] = df.groupby("country")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # Aggregate to country level
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ["pfaf_id", "aq30_id", "aqid", "area_km2"]]

    country_agg = df.groupby(["country", "country_code"]).agg(
        **{col: (col, "mean") for col in numeric_cols if col in df.columns}
    ).reset_index()

    for col in country_agg.select_dtypes(include=[np.number]).columns:
        country_agg[col] = country_agg[col].round(4)

    country_agg["source"] = "WRI_Aqueduct_4.0"
    logger.info(f"  Cleaned: {len(country_agg)} countries, {len(country_agg.columns)} features")

    out = config.AQUEDUCT_PROCESSED
    country_agg.to_csv(out, index=False)
    logger.info(f"  ✅ Saved to: {out}")
    return country_agg


# ──────────────────────────────────────────────────────────────
# 2. AQUASTAT (real CSV — 936K rows, latin-1 encoded)
# ──────────────────────────────────────────────────────────────
def process_aquastat():
    """Process real FAO AQUASTAT bulk CSV."""
    logger.info("=" * 60)
    logger.info("Processing FAO AQUASTAT data...")

    csv_path = config.RAW_AQUASTAT_DIR / "bulk_eng(in).csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"AQUASTAT CSV not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    logger.info(f"  Raw: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    # Standardize column names
    df = df_raw.rename(columns={
        df_raw.columns[0]: "variable_id",
        df_raw.columns[1]: "variable",
        df_raw.columns[2]: "area_id",
        df_raw.columns[3]: "country",
        df_raw.columns[4]: "year_id",
        df_raw.columns[5]: "year",
        df_raw.columns[6]: "value",
    })

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["value", "year"])

    logger.info(f"  Unique variables: {df['variable'].nunique()}")

    # Filter key water variables
    key_variables = {
        "SDG 6.4.2. Water Stress": "sdg642_water_stress_pct",
        "Total renewable water resources": "total_renewable_water_km3",
        "Total water withdrawal": "total_water_withdrawal_km3",
        "Agricultural water withdrawal": "agricultural_withdrawal_km3",
        "Industrial water withdrawal": "industrial_withdrawal_km3",
        "Municipal water withdrawal": "municipal_withdrawal_km3",
        "Total renewable water resources per capita": "renewable_water_per_capita",
        "Precipitation": "precipitation_mm",
        "Dependency ratio": "dependency_ratio_pct",
        "Total dam capacity": "dam_capacity_km3",
        "Flood occurrence": "flood_occurrence",
        "Drought occurrence": "drought_occurrence",
        "National Rainfall Index": "rainfall_index",
        "Total population with access to safe drinking-water": "safe_water_access_pct",
    }

    def match_variable(var_name):
        var_lower = str(var_name).lower()
        for key, clean_name in key_variables.items():
            if key.lower() in var_lower:
                return clean_name
        return None

    df["clean_variable"] = df["variable"].apply(match_variable)
    df_filtered = df[df["clean_variable"].notna()].copy()
    logger.info(f"  Filtered: {len(df_filtered):,} rows ({df_filtered['clean_variable'].nunique()} variables)")

    # Build ISO-3 mapping
    country_to_iso = _build_country_iso_map()
    df_filtered["country_code"] = df_filtered["country"].map(country_to_iso)

    # Pivot to wide format
    pivot = df_filtered.pivot_table(
        index=["country", "country_code", "year"],
        columns="clean_variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivot = pivot.sort_values(["country", "year"]).reset_index(drop=True)

    # Interpolate within each country
    numeric_cols = pivot.select_dtypes(include=[np.number]).columns.difference(["year"])
    pivot[numeric_cols] = pivot.groupby("country")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    pivot["source"] = "FAO_AQUASTAT"
    logger.info(f"  Cleaned: {len(pivot):,} rows, {pivot['country'].nunique()} countries")

    out = config.AQUASTAT_PROCESSED
    pivot.to_csv(out, index=False)
    logger.info(f"  ✅ Saved to: {out}")
    return pivot


# ──────────────────────────────────────────────────────────────
# 3. GRACE (real NetCDF — 253 timesteps × 360 × 720)
# ──────────────────────────────────────────────────────────────
def process_grace():
    """Process real NASA GRACE NetCDF to country-level monthly CSV."""
    import xarray as xr

    logger.info("=" * 60)
    logger.info("Processing NASA GRACE data...")

    nc_path = config.RAW_GRACE_DIR / "1.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"GRACE NetCDF not found at: {nc_path}")

    ds = xr.open_dataset(nc_path)
    logger.info(f"  Variables: {list(ds.data_vars)}")
    logger.info(f"  Time: {len(ds.time)} steps ({str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]})")

    tws_data = ds["lwe_thickness"].values  # (253, 360, 720)
    unc_data = ds["uncertainty"].values if "uncertainty" in ds.data_vars else None
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    time_vals = pd.to_datetime(ds.time.values)

    # Country centroids — cover major countries globally
    centroids = {
        "AFG": ("Afghanistan", 33.9, 67.7), "DZA": ("Algeria", 28.0, 1.7),
        "AGO": ("Angola", -11.2, 17.9), "ARG": ("Argentina", -38.4, -63.6),
        "AUS": ("Australia", -25.3, 133.8), "BGD": ("Bangladesh", 23.7, 90.3),
        "BFA": ("Burkina Faso", 12.3, -1.6), "BRA": ("Brazil", -14.0, -51.0),
        "CAN": ("Canada", 56.1, -106.3), "TCD": ("Chad", 15.5, 18.7),
        "CHL": ("Chile", -35.7, -71.5), "CHN": ("China", 35.0, 105.0),
        "COL": ("Colombia", 4.6, -74.3), "COD": ("DR Congo", -4.0, 21.8),
        "CIV": ("Ivory Coast", 7.5, -5.5), "EGY": ("Egypt", 26.8, 30.8),
        "ETH": ("Ethiopia", 9.1, 40.5), "FRA": ("France", 46.2, 2.2),
        "DEU": ("Germany", 51.2, 10.4), "GHA": ("Ghana", 7.9, -1.0),
        "IND": ("India", 20.6, 79.0), "IDN": ("Indonesia", -5.0, 120.0),
        "IRN": ("Iran", 32.4, 53.7), "IRQ": ("Iraq", 33.2, 43.7),
        "ISR": ("Israel", 31.0, 34.9), "ITA": ("Italy", 41.9, 12.6),
        "JPN": ("Japan", 36.2, 138.2), "JOR": ("Jordan", 30.6, 36.2),
        "KEN": ("Kenya", -0.02, 37.9), "KWT": ("Kuwait", 29.3, 47.5),
        "LBY": ("Libya", 26.3, 17.2), "MDG": ("Madagascar", -18.8, 47.0),
        "MYS": ("Malaysia", 4.2, 101.9), "MLI": ("Mali", 17.6, -4.0),
        "MAR": ("Morocco", 31.8, -7.1), "MOZ": ("Mozambique", -18.7, 35.5),
        "MMR": ("Myanmar", 19.8, 96.2), "NPL": ("Nepal", 28.4, 84.1),
        "NER": ("Niger", 17.6, 8.1), "NGA": ("Nigeria", 9.0, 8.0),
        "OMN": ("Oman", 21.5, 55.9), "PAK": ("Pakistan", 30.0, 69.0),
        "PER": ("Peru", -9.2, -75.0), "PHL": ("Philippines", 12.9, 121.8),
        "POL": ("Poland", 51.9, 19.1), "ROU": ("Romania", 45.9, 25.0),
        "RUS": ("Russia", 61.5, 105.0), "SAU": ("Saudi Arabia", 23.9, 45.1),
        "SEN": ("Senegal", 14.5, -14.5), "SOM": ("Somalia", 5.2, 46.2),
        "ZAF": ("South Africa", -30.6, 22.9), "KOR": ("South Korea", 35.9, 128.0),
        "ESP": ("Spain", 40.5, -3.7), "SDN": ("Sudan", 12.9, 30.2),
        "SYR": ("Syria", 34.8, 38.0), "TZA": ("Tanzania", -6.4, 34.9),
        "THA": ("Thailand", 15.9, 100.9), "TUN": ("Tunisia", 33.9, 9.5),
        "TUR": ("Turkey", 38.9, 35.2), "UGA": ("Uganda", 1.4, 32.3),
        "UKR": ("Ukraine", 48.4, 31.2), "ARE": ("UAE", 23.4, 53.8),
        "GBR": ("United Kingdom", 55.4, -3.4),
        "USA": ("United States", 38.0, -97.0),
        "VEN": ("Venezuela", 6.4, -66.6), "VNM": ("Vietnam", 14.1, 108.3),
        "YEM": ("Yemen", 15.6, 48.5), "ZMB": ("Zambia", -13.1, 27.8),
        "ZWE": ("Zimbabwe", -20.0, 30.0), "MEX": ("Mexico", 23.6, -102.5),
        "CMR": ("Cameroon", 7.4, 12.4), "LBN": ("Lebanon", 33.9, 35.8),
    }

    logger.info(f"  Extracting TWS for {len(centroids)} countries × {len(time_vals)} months...")

    records = []
    for code, (name, clat, clon) in centroids.items():
        lat_idx = np.argmin(np.abs(lat_vals - clat))
        lon_idx = np.argmin(np.abs(lon_vals - clon))

        for t_idx, t in enumerate(time_vals):
            try:
                val = float(tws_data[t_idx, lat_idx, lon_idx])
                if np.isnan(val) or abs(val) > 1e10:
                    continue
                rec = {
                    "country": name, "country_code": code,
                    "lat": clat, "lon": clon,
                    "date": t.strftime("%Y-%m"),
                    "tws_anomaly_cm": round(val, 4),
                }
                if unc_data is not None:
                    try:
                        unc_val = float(unc_data[t_idx, lat_idx, lon_idx])
                        if not np.isnan(unc_val) and abs(unc_val) < 1e10:
                            rec["uncertainty_cm"] = round(unc_val, 4)
                    except:
                        pass
                records.append(rec)
            except (IndexError, ValueError):
                continue

    ds.close()
    grace_df = pd.DataFrame(records)
    grace_df = grace_df.sort_values(["country_code", "date"]).reset_index(drop=True)

    # Derived features
    for window in [3, 6, 12]:
        grace_df[f"tws_{window}month_avg"] = (
            grace_df.groupby("country_code")["tws_anomaly_cm"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean()).round(4)
        )

    grace_df["tws_rate_of_change"] = (
        grace_df.groupby("country_code")["tws_anomaly_cm"].transform(lambda x: x.diff()).round(4)
    )
    grace_df["tws_cumulative_change"] = (
        grace_df.groupby("country_code")["tws_anomaly_cm"]
        .transform(lambda x: x - x.iloc[0]).round(4)
    )
    grace_df["tws_zscore"] = (
        grace_df.groupby("country_code")["tws_anomaly_cm"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)).round(4)
    )
    grace_df["groundwater_anomaly_cm"] = (grace_df["tws_anomaly_cm"] * 0.4).round(4)

    def compute_trend(group):
        if len(group) < 6:
            return pd.Series(0.0, index=group.index)
        x = np.arange(len(group))
        y = group.values
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return pd.Series(0.0, index=group.index)
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return pd.Series(round(slope * 12, 4), index=group.index)

    grace_df["tws_trend_cm_yr"] = (
        grace_df.groupby("country_code")["tws_anomaly_cm"].transform(compute_trend)
    )

    grace_df["source"] = "NASA_GRACE"
    logger.info(f"  Cleaned: {len(grace_df):,} rows, {grace_df['country_code'].nunique()} countries")
    logger.info(f"  Date range: {grace_df['date'].min()} to {grace_df['date'].max()}")

    out = config.GRACE_PROCESSED
    grace_df.to_csv(out, index=False)
    logger.info(f"  ✅ Saved to: {out}")
    return grace_df


# ──────────────────────────────────────────────────────────────
# 4. MERGE ALL SOURCES
# ──────────────────────────────────────────────────────────────
def merge_and_consolidate(aqueduct_df, aquastat_df, grace_df):
    """Merge all three cleaned datasets into one consolidated CSV."""
    logger.info("=" * 60)
    logger.info("Merging all datasets...")

    merged = grace_df.copy()

    # Join Aqueduct (static, by country_code)
    aq_cols = [c for c in aqueduct_df.columns
               if c not in ["country", "source", "lat", "lon"]]
    merged = merged.merge(aqueduct_df[aq_cols], on="country_code", how="left", suffixes=("", "_aq"))
    logger.info(f"  After Aqueduct join: {len(merged):,} rows")

    # Join AQUASTAT (by country_code + year)
    merged["year"] = merged["date"].str[:4].astype(int)

    if "year" in aquastat_df.columns:
        aqua_cols = [c for c in aquastat_df.columns if c not in ["country", "source"]]
        merged = merged.merge(aquastat_df[aqua_cols], on=["country_code", "year"],
                              how="left", suffixes=("", "_aqua"))
    logger.info(f"  After AQUASTAT join: {len(merged):,} rows")

    # Composite drought score
    components, weights = [], []
    if "water_stress_score" in merged.columns:
        components.append("water_stress_score"); weights.append(0.25)
    if "drought_risk_score" in merged.columns:
        components.append("drought_risk_score"); weights.append(0.25)
    if "tws_anomaly_cm" in merged.columns:
        tws_min = merged["tws_anomaly_cm"].quantile(0.01)
        tws_max = merged["tws_anomaly_cm"].quantile(0.99)
        merged["tws_score_norm"] = (5.0 * (1.0 - (merged["tws_anomaly_cm"] - tws_min) / (tws_max - tws_min + 1e-8))).clip(0, 5)
        components.append("tws_score_norm"); weights.append(0.3)
    if "groundwater_decline_score" in merged.columns:
        components.append("groundwater_decline_score"); weights.append(0.2)

    w = np.array(weights); w = w / w.sum()
    merged["drought_composite_score"] = sum(
        merged[c].fillna(0) * wt for c, wt in zip(components, w)
    ).round(4)

    merged["drought_risk_class"] = merged["drought_composite_score"].apply(
        lambda s: "Low" if s < 1.0 else "Moderate" if s < 2.0 else "High" if s < 3.5 else "Extreme"
    )

    # Season
    merged["month"] = merged["date"].str[5:7].astype(int)
    merged["season"] = merged["month"].map({
        12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"
    })
    merged.drop(columns=["month", "tws_score_norm"], inplace=True, errors="ignore")
    merged = merged.sort_values(["country_code", "date"]).reset_index(drop=True)

    logger.info(f"  Final: {len(merged):,} rows × {len(merged.columns)} columns")
    logger.info(f"  Countries: {merged['country_code'].nunique()}")
    logger.info(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    logger.info(f"  Drought risk:\n{merged['drought_risk_class'].value_counts().to_string()}")

    out = config.CONSOLIDATED_CSV
    merged.to_csv(out, index=False)
    logger.info(f"  ✅ Saved to: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return merged


# Helper
def _build_country_iso_map():
    mapping = {
        "Afghanistan": "AFG", "Algeria": "DZA", "Angola": "AGO",
        "Argentina": "ARG", "Australia": "AUS", "Bangladesh": "BGD",
        "Bolivia (Plurinational State of)": "BOL", "Brazil": "BRA",
        "Burkina Faso": "BFA", "Cameroon": "CMR", "Canada": "CAN",
        "Chad": "TCD", "Chile": "CHL", "China": "CHN", "Colombia": "COL",
        "Congo": "COG", "Côte d'Ivoire": "CIV",
        "Democratic Republic of the Congo": "COD", "Egypt": "EGY",
        "Ethiopia": "ETH", "France": "FRA", "Germany": "DEU",
        "Ghana": "GHA", "India": "IND", "Indonesia": "IDN",
        "Iran (Islamic Republic of)": "IRN", "Iraq": "IRQ",
        "Israel": "ISR", "Italy": "ITA", "Japan": "JPN",
        "Jordan": "JOR", "Kenya": "KEN", "Kuwait": "KWT",
        "Lao People's Democratic Republic": "LAO",
        "Lebanon": "LBN", "Libya": "LBY", "Madagascar": "MDG",
        "Malaysia": "MYS", "Mali": "MLI", "Mexico": "MEX",
        "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR",
        "Nepal": "NPL", "Niger": "NER", "Nigeria": "NGA",
        "Oman": "OMN", "Pakistan": "PAK", "Peru": "PER",
        "Philippines": "PHL", "Poland": "POL", "Romania": "ROU",
        "Russian Federation": "RUS", "Saudi Arabia": "SAU",
        "Senegal": "SEN", "Somalia": "SOM", "South Africa": "ZAF",
        "Korea, Republic of": "KOR", "Spain": "ESP",
        "Sudan": "SDN", "Syrian Arab Republic": "SYR",
        "Tanzania, United Republic of": "TZA",
        "Thailand": "THA", "Tunisia": "TUN",
        "Türkiye": "TUR", "Turkey": "TUR",
        "Uganda": "UGA", "Ukraine": "UKR",
        "United Arab Emirates": "ARE",
        "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "United States of America": "USA",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Viet Nam": "VNM", "Yemen": "YEM",
        "Zambia": "ZMB", "Zimbabwe": "ZWE",
        "Iran": "IRN", "Russia": "RUS",
        "United Kingdom": "GBR", "United States": "USA",
        "Venezuela": "VEN", "Vietnam": "VNM",
        "South Korea": "KOR", "Syria": "SYR",
        "Tanzania": "TZA", "Bolivia": "BOL", "Laos": "LAO",
        "Burma": "MMR", "Ivory Coast": "CIV", "Cuba": "CUB",
    }
    return mapping


if __name__ == "__main__":
    logger.info("🌍 EDA-15: Processing REAL downloaded datasets")
    logger.info("=" * 60)

    aqueduct_df = process_aqueduct()
    aquastat_df = process_aquastat()
    grace_df = process_grace()
    consolidated = merge_and_consolidate(aqueduct_df, aquastat_df, grace_df)

    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL DONE!")
    logger.info(f"  Aqueduct:     {len(aqueduct_df)} countries")
    logger.info(f"  AQUASTAT:     {len(aquastat_df)} records")
    logger.info(f"  GRACE:        {len(grace_df)} records")
    logger.info(f"  Consolidated: {len(consolidated)} rows × {len(consolidated.columns)} cols")
    logger.info(f"  Output:       {config.CONSOLIDATED_CSV}")
    logger.info("=" * 60)
