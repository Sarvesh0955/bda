"""
Data Cleaner Module
Handles missing values, standardizes formats, and normalizes identifiers
across the three data sources (Aqueduct, AQUASTAT, GRACE).
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and standardize water stress data from multiple sources."""

    # Standard ISO-3 country codes we use
    ISO3_MAPPING = {
        "united states of america": "USA",
        "united states": "USA",
        "russian federation": "RUS",
        "russia": "RUS",
        "iran (islamic republic of)": "IRN",
        "iran": "IRN",
        "türkiye": "TUR",
        "turkey": "TUR",
        "venezuela (bolivarian republic of)": "VEN",
        "venezuela": "VEN",
        "united kingdom of great britain and northern ireland": "GBR",
        "united kingdom": "GBR",
        "korea, republic of": "KOR",
        "south korea": "KOR",
        "tanzania, united republic of": "TZA",
        "tanzania": "TZA",
        "congo, democratic republic of the": "COD",
        "democratic republic of the congo": "COD",
        "côte d'ivoire": "CIV",
        "ivory coast": "CIV",
    }

    def __init__(self):
        pass

    def clean_aqueduct(self, df):
        """
        Clean Aqueduct data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or semi-processed Aqueduct data

        Returns
        -------
        pd.DataFrame : cleaned data
        """
        logger.info(f"Cleaning Aqueduct data ({len(df)} rows)...")
        df = df.copy()

        # Standardize country codes
        if "country" in df.columns:
            df["country_code"] = df.apply(
                lambda row: self._resolve_country_code(
                    row.get("country_code", ""),
                    row.get("country", "")
                ), axis=1
            )

        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0:
                if missing_pct < 0.3:
                    # Forward-fill then backward-fill for small gaps
                    df[col] = df[col].fillna(df[col].median())
                else:
                    logger.warning(f"  {col}: {missing_pct:.1%} missing — using 0")
                    df[col] = df[col].fillna(0)

        # Clip score columns to valid range
        score_cols = [c for c in df.columns if "score" in c.lower()]
        for col in score_cols:
            df[col] = df[col].clip(0, 5)

        # Remove duplicates
        before = len(df)
        id_cols = [c for c in ["country_code", "lat", "lon"] if c in df.columns]
        if id_cols:
            df = df.drop_duplicates(subset=id_cols, keep="first")
        logger.info(f"  Removed {before - len(df)} duplicates → {len(df)} rows")

        return df

    def clean_aquastat(self, df):
        """
        Clean AQUASTAT data.

        Parameters
        ----------
        df : pd.DataFrame
            Pivoted AQUASTAT data (one row per country-year)

        Returns
        -------
        pd.DataFrame : cleaned data
        """
        logger.info(f"Cleaning AQUASTAT data ({len(df)} rows)...")
        df = df.copy()

        # Standardize country codes
        if "country" in df.columns:
            df["country_code"] = df.apply(
                lambda row: self._resolve_country_code(
                    row.get("country_code", ""),
                    row.get("country", "")
                ), axis=1
            )

        # Handle time series gaps with interpolation
        if "year" in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c != "year"]

            # Interpolate within each country
            cleaned = []
            for code in df["country_code"].unique():
                country_data = df[df["country_code"] == code].copy()
                country_data = country_data.sort_values("year")

                for col in numeric_cols:
                    if col in country_data.columns:
                        # Linear interpolation for time series
                        country_data[col] = country_data[col].interpolate(
                            method="linear", limit_direction="both"
                        )

                cleaned.append(country_data)

            df = pd.concat(cleaned, ignore_index=True)

        # Remove extreme outliers (> 5 * IQR)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["year"]:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 5 * iqr
                upper = q3 + 5 * iqr
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers > 0:
                    df[col] = df[col].clip(lower, upper)
                    logger.info(f"  Clipped {outliers} outliers in {col}")

        return df

    def clean_grace(self, df):
        """
        Clean GRACE TWS anomaly data.

        Parameters
        ----------
        df : pd.DataFrame
            GRACE time series data

        Returns
        -------
        pd.DataFrame : cleaned data
        """
        logger.info(f"Cleaning GRACE data ({len(df)} rows)...")
        df = df.copy()

        # Standardize dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # Remove rows with invalid dates
            invalid_dates = df["date"].isna().sum()
            if invalid_dates > 0:
                logger.info(f"  Removed {invalid_dates} rows with invalid dates")
                df = df.dropna(subset=["date"])

        # Fill small gaps in time series (within each region)
        if "country_code" in df.columns and "date" in df.columns:
            cleaned = []
            for code in df["country_code"].unique():
                region_data = df[df["country_code"] == code].copy()
                region_data = region_data.sort_values("date")

                # Interpolate TWS gaps (max 3 months)
                tws_cols = [c for c in region_data.columns if "tws" in c.lower() or "anomaly" in c.lower()]
                for col in tws_cols:
                    if col in region_data.columns:
                        region_data[col] = region_data[col].interpolate(
                            method="linear", limit=3, limit_direction="both"
                        )

                cleaned.append(region_data)

            df = pd.concat(cleaned, ignore_index=True)

        # Convert date back to string
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.strftime("%Y-%m")

        return df

    def _resolve_country_code(self, code, name):
        """Resolve country code from code or name."""
        if pd.notna(code) and str(code).strip():
            return str(code).strip().upper()[:3]

        if pd.notna(name):
            name_lower = str(name).strip().lower()
            if name_lower in self.ISO3_MAPPING:
                return self.ISO3_MAPPING[name_lower]

        return str(code) if pd.notna(code) else "UNK"

    def add_season(self, df):
        """Add season column based on date."""
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            df["season"] = dates.dt.month.map({
                12: "DJF", 1: "DJF", 2: "DJF",
                3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA",
                9: "SON", 10: "SON", 11: "SON",
            })
        return df

    def validate_data(self, df, name="dataset"):
        """Print data quality report."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Data Quality Report: {name}")
        logger.info(f"{'='*60}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        has_missing = missing[missing > 0]
        if len(has_missing) > 0:
            logger.info(f"  Missing values:")
            for col in has_missing.index:
                logger.info(f"    {col}: {has_missing[col]} ({missing_pct[col]}%)")
        else:
            logger.info(f"  No missing values ✓")

        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info(f"  Numeric columns: {list(numeric_cols)}")

        logger.info(f"{'='*60}\n")
        return {
            "shape": df.shape,
            "missing": has_missing.to_dict() if len(has_missing) > 0 else {},
            "dtypes": df.dtypes.value_counts().to_dict(),
        }
