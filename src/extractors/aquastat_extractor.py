"""
FAO AQUASTAT Data Extractor
Downloads and processes water resources data from the FAO AQUASTAT database.
"""
import logging
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from io import StringIO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class AquastatExtractor:
    """Extract and process FAO AQUASTAT water resources data."""

    # Known AQUASTAT variable IDs for key indicators
    VARIABLE_IDS = {
        4263: "SDG 6.4.2. Water Stress (%)",
        4157: "Total renewable water resources (10^9 m3/year)",
        4250: "Total water withdrawal (10^9 m3/year)",
        4251: "Agricultural water withdrawal (10^9 m3/year)",
        4252: "Industrial water withdrawal (10^9 m3/year)",
        4253: "Municipal water withdrawal (10^9 m3/year)",
        4154: "Total renewable water resources per capita (m3/inhab/year)",
        4101: "Precipitation (mm/year)",
        4155: "Dependency ratio (%)",
        4549: "Flood occurrence (number)",
        4550: "Drought occurrence (number)",
    }

    def __init__(self, raw_dir=None, processed_dir=None):
        self.raw_dir = Path(raw_dir) if raw_dir else config.RAW_AQUASTAT_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else config.PROCESSED_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_data(self, force=False):
        """
        Download AQUASTAT bulk data.

        Attempts to download from FAO's data portal. Falls back to
        synthetic data if the download fails.

        Returns
        -------
        Path : path to the downloaded/generated file
        """
        output_file = config.AQUASTAT_RAW_CSV

        if output_file.exists() and not force:
            logger.info(f"File already exists: {output_file}")
            return output_file

        # Try multiple known download endpoints
        urls_to_try = [
            "https://data.apps.fao.org/aquastat/data/csv",
            "https://storage.googleapis.com/fao-aquastat/aquastat_data.csv",
        ]

        for url in urls_to_try:
            try:
                logger.info(f"Attempting download from: {url}")
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()

                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Downloaded AQUASTAT data to: {output_file}")
                return output_file

            except requests.RequestException as e:
                logger.warning(f"Download from {url} failed: {e}")
                continue

        # Fallback: generate synthetic data
        logger.warning("All download attempts failed. Generating synthetic data.")
        return self._generate_synthetic_data(output_file)

    def _generate_synthetic_data(self, output_file):
        """Generate realistic synthetic AQUASTAT data for development."""
        np.random.seed(42)

        countries = {
            "IND": ("India", 4),       "CHN": ("China", 4),
            "USA": ("United States of America", 840),
            "BRA": ("Brazil", 76),     "NGA": ("Nigeria", 566),
            "PAK": ("Pakistan", 586),  "IDN": ("Indonesia", 360),
            "BGD": ("Bangladesh", 50), "RUS": ("Russian Federation", 643),
            "MEX": ("Mexico", 484),    "ETH": ("Ethiopia", 231),
            "EGY": ("Egypt", 818),     "DEU": ("Germany", 276),
            "TUR": ("Türkiye", 792),   "IRN": ("Iran (Islamic Republic of)", 364),
            "THA": ("Thailand", 764),  "FRA": ("France", 250),
            "GBR": ("United Kingdom of Great Britain and Northern Ireland", 826),
            "ITA": ("Italy", 380),     "ZAF": ("South Africa", 710),
            "KEN": ("Kenya", 404),     "SAU": ("Saudi Arabia", 682),
            "IRQ": ("Iraq", 368),      "AFG": ("Afghanistan", 4),
            "YEM": ("Yemen", 887),     "AUS": ("Australia", 36),
            "ESP": ("Spain", 724),     "MAR": ("Morocco", 504),
            "COL": ("Colombia", 170),  "ARG": ("Argentina", 32),
            "DZA": ("Algeria", 12),    "SDN": ("Sudan", 729),
            "PER": ("Peru", 604),      "GHA": ("Ghana", 288),
            "MOZ": ("Mozambique", 508), "NPL": ("Nepal", 524),
            "CHL": ("Chile", 152),     "MYS": ("Malaysia", 458),
            "VEN": ("Venezuela (Bolivarian Republic of)", 862),
            "CMR": ("Cameroon", 120),  "JPN": ("Japan", 392),
        }

        # Water stress profiles per country (base %)
        stress_base = {
            "IND": 65, "CHN": 42, "USA": 22, "BRA": 5, "NGA": 25,
            "PAK": 85, "IDN": 12, "BGD": 15, "RUS": 4, "MEX": 45,
            "ETH": 8, "EGY": 117, "DEU": 18, "TUR": 40, "IRN": 85,
            "THA": 20, "FRA": 16, "GBR": 10, "ITA": 35, "ZAF": 50,
            "KEN": 12, "SAU": 943, "IRQ": 78, "AFG": 45, "YEM": 170,
            "AUS": 18, "ESP": 33, "MAR": 45, "COL": 4, "ARG": 10,
            "DZA": 65, "SDN": 95, "PER": 6, "GHA": 8, "MOZ": 2,
            "NPL": 5, "CHL": 35, "MYS": 8, "VEN": 3, "CMR": 1,
            "JPN": 21,
        }

        # Renewable water resources (km³/year) approximate
        renewable_water = {
            "IND": 1911, "CHN": 2840, "USA": 3069, "BRA": 8233, "NGA": 286,
            "PAK": 247, "IDN": 2019, "BGD": 1227, "RUS": 4508, "MEX": 462,
            "ETH": 122, "EGY": 57, "DEU": 154, "TUR": 211, "IRN": 137,
            "THA": 438, "FRA": 211, "GBR": 147, "ITA": 191, "ZAF": 51,
            "KEN": 31, "SAU": 2.4, "IRQ": 90, "AFG": 65, "YEM": 2.1,
            "AUS": 492, "ESP": 112, "MAR": 29, "COL": 2360, "ARG": 876,
            "DZA": 11.7, "SDN": 30, "PER": 1913, "GHA": 53, "MOZ": 218,
            "NPL": 210, "CHL": 922, "MYS": 580, "VEN": 1233, "CMR": 283,
            "JPN": 430,
        }

        years = list(range(1960, 2025, 5))
        records = []

        for iso, (name, code) in countries.items():
            base_stress = stress_base.get(iso, 20)
            base_renewable = renewable_water.get(iso, 100)

            for year in years:
                # Simulate trends: stress increases over time, water decreases
                year_factor = (year - 1960) / 65
                stress_trend = base_stress * (1 + 0.3 * year_factor)
                water_trend = base_renewable * (1 - 0.05 * year_factor)

                total_withdrawal = water_trend * (stress_trend / 100)
                agri_withdrawal = total_withdrawal * np.random.uniform(0.55, 0.85)
                industrial_withdrawal = total_withdrawal * np.random.uniform(0.05, 0.25)
                municipal_withdrawal = total_withdrawal - agri_withdrawal - industrial_withdrawal

                precipitation = np.random.uniform(200, 2500) * (1 - 0.02 * year_factor)

                records.extend([
                    {
                        "Area": name, "Area Id": code, "Variable": "SDG 6.4.2. Water Stress",
                        "Variable Id": 4263, "Year": year,
                        "Value": round(stress_trend + np.random.normal(0, base_stress * 0.05), 2),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Total renewable water resources",
                        "Variable Id": 4157, "Year": year,
                        "Value": round(water_trend + np.random.normal(0, base_renewable * 0.02), 2),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Total water withdrawal",
                        "Variable Id": 4250, "Year": year,
                        "Value": round(total_withdrawal + np.random.normal(0, 2), 2),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Agricultural water withdrawal",
                        "Variable Id": 4251, "Year": year,
                        "Value": round(max(0, agri_withdrawal + np.random.normal(0, 1)), 2),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Industrial water withdrawal",
                        "Variable Id": 4252, "Year": year,
                        "Value": round(max(0, industrial_withdrawal + np.random.normal(0, 0.5)), 2),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Municipal water withdrawal",
                        "Variable Id": 4253, "Year": year,
                        "Value": round(max(0, municipal_withdrawal + np.random.normal(0, 0.5)), 2),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Precipitation",
                        "Variable Id": 4101, "Year": year,
                        "Value": round(max(50, precipitation), 1),
                        "Symbol": "", "Iso3": iso,
                    },
                    {
                        "Area": name, "Area Id": code,
                        "Variable": "Total renewable water resources per capita",
                        "Variable Id": 4154, "Year": year,
                        "Value": round(max(10, water_trend * 1e9 / (50e6 * (1 + year_factor))), 1),
                        "Symbol": "", "Iso3": iso,
                    },
                ])

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        logger.info(f"Synthetic AQUASTAT data ({len(df)} records) saved to: {output_file}")
        return output_file

    def load_raw_data(self):
        """Load raw AQUASTAT data from CSV."""
        filepath = config.AQUASTAT_RAW_CSV
        if not filepath.exists():
            logger.info("Raw data not found, downloading...")
            filepath = self.download_data()
        return pd.read_csv(filepath)

    def extract_features(self, df=None):
        """
        Extract and pivot key water variables from AQUASTAT long-format data.

        Returns
        -------
        pd.DataFrame : wide-format DataFrame with one row per country-year
        """
        if df is None:
            df = self.load_raw_data()

        logger.info(f"Processing {len(df)} AQUASTAT records...")

        # Filter for key variables
        key_vars = list(self.VARIABLE_IDS.values())
        # Try matching on Variable column
        if "Variable" in df.columns:
            # Fuzzy match: check if any key variable substring is in the data
            mask = df["Variable"].apply(
                lambda v: any(kv.split("(")[0].strip().lower() in str(v).lower() for kv in key_vars)
                if pd.notna(v) else False
            )
            df_filtered = df[mask].copy()
        else:
            df_filtered = df.copy()

        if len(df_filtered) == 0:
            logger.warning("No matching variables found. Using all data.")
            df_filtered = df.copy()

        # Standardize column names
        col_map = {}
        for col in df_filtered.columns:
            col_lower = col.lower().strip()
            if col_lower in ("area", "country", "name"):
                col_map[col] = "country"
            elif col_lower in ("iso3", "iso_a3", "iso", "country_code"):
                col_map[col] = "country_code"
            elif col_lower in ("year", "time_period"):
                col_map[col] = "year"
            elif col_lower in ("value",):
                col_map[col] = "value"
            elif col_lower in ("variable", "indicator"):
                col_map[col] = "variable"

        df_filtered = df_filtered.rename(columns=col_map)

        # Pivot from long to wide format
        required = ["country", "country_code", "year", "variable", "value"]
        available = [c for c in required if c in df_filtered.columns]

        if set(available) >= {"country", "year", "variable", "value"}:
            # Clean variable names for column headers
            df_filtered["variable_clean"] = df_filtered["variable"].apply(self._clean_variable_name)

            pivot = df_filtered.pivot_table(
                index=["country", "country_code", "year"] if "country_code" in df_filtered.columns
                else ["country", "year"],
                columns="variable_clean",
                values="value",
                aggfunc="first",
            ).reset_index()

            # Flatten multi-level columns
            if isinstance(pivot.columns, pd.MultiIndex):
                pivot.columns = ["_".join(str(c) for c in col).strip("_") for col in pivot.columns]
        else:
            pivot = df_filtered

        # Add source
        pivot["source"] = "FAO_AQUASTAT"

        # Sort
        sort_cols = [c for c in ["country", "year"] if c in pivot.columns]
        if sort_cols:
            pivot = pivot.sort_values(sort_cols).reset_index(drop=True)

        logger.info(f"Extracted {len(pivot)} country-year records with {len(pivot.columns)} features")
        return pivot

    @staticmethod
    def _clean_variable_name(name):
        """Convert variable name to a clean column name."""
        clean = str(name).lower()
        # Extract core meaning
        replacements = {
            "sdg 6.4.2. water stress": "sdg642_water_stress_pct",
            "total renewable water resources per capita": "renewable_water_per_capita",
            "total renewable water resources": "total_renewable_water_km3",
            "total water withdrawal": "total_water_withdrawal_km3",
            "agricultural water withdrawal": "agricultural_withdrawal_km3",
            "industrial water withdrawal": "industrial_withdrawal_km3",
            "municipal water withdrawal": "municipal_withdrawal_km3",
            "precipitation": "precipitation_mm",
            "dependency ratio": "dependency_ratio_pct",
            "flood occurrence": "flood_occurrence",
            "drought occurrence": "drought_occurrence",
        }
        for key, value in replacements.items():
            if key in clean:
                return value
        # Fallback: sanitize
        return clean.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")[:50]

    def save_processed(self, df=None):
        """Extract features and save to processed directory."""
        if df is None:
            df = self.extract_features()
        output_path = config.AQUASTAT_PROCESSED
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed AQUASTAT data to: {output_path}")
        return output_path

    def get_water_stress_rankings(self, df=None, year=None):
        """Get countries ranked by water stress level."""
        if df is None:
            df = self.extract_features()

        if year and "year" in df.columns:
            df = df[df["year"] == year]

        stress_col = [c for c in df.columns if "water_stress" in c.lower()]
        if stress_col:
            return df.sort_values(stress_col[0], ascending=False)
        return df


def main():
    """Run AQUASTAT data extraction pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    extractor = AquastatExtractor()
    extractor.download_data()
    df = extractor.extract_features()

    print(f"\nExtracted {len(df)} records:")
    print(df.head(10))
    print(f"\nColumns: {list(df.columns)}")

    extractor.save_processed(df)


if __name__ == "__main__":
    main()
