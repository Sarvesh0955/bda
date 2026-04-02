"""
Data Merger Module
Merges cleaned data from WRI Aqueduct, FAO AQUASTAT, and NASA GRACE
into a consolidated drought/water stress dataset.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge multiple water stress data sources into a unified dataset."""

    def __init__(self):
        self.drought_thresholds = config.DROUGHT_RISK_THRESHOLDS

    def merge_all(self, aqueduct_df, aquastat_df, grace_df):
        """
        Merge all three data sources into a consolidated dataset.

        Strategy:
        - GRACE provides the temporal backbone (monthly data 2002–2024)
        - Aqueduct provides spatial water stress indicators (static baseline)
        - AQUASTAT provides country-level socioeconomic water indicators (5-year)

        Parameters
        ----------
        aqueduct_df : pd.DataFrame
            Cleaned Aqueduct data (spatial, mostly static)
        aquastat_df : pd.DataFrame
            Cleaned AQUASTAT data (country × year)
        grace_df : pd.DataFrame
            Cleaned GRACE data (country × month)

        Returns
        -------
        pd.DataFrame : consolidated dataset
        """
        logger.info("Merging all data sources...")
        logger.info(f"  Aqueduct: {aqueduct_df.shape}")
        logger.info(f"  AQUASTAT:  {aquastat_df.shape}")
        logger.info(f"  GRACE:     {grace_df.shape}")

        # Step 1: Aggregate Aqueduct to country level (mean of catchments)
        aqueduct_country = self._aggregate_aqueduct_to_country(aqueduct_df)
        logger.info(f"  Aqueduct (country-level): {aqueduct_country.shape}")

        # Step 2: Start with GRACE as the temporal backbone
        merged = grace_df.copy()
        if "date" in merged.columns:
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged["year"] = merged["date"].dt.year

        # Step 3: Merge Aqueduct (country-level static indicators)
        if "country_code" in aqueduct_country.columns and "country_code" in merged.columns:
            # Drop overlapping columns before merge
            aq_cols = [c for c in aqueduct_country.columns
                       if c not in merged.columns or c == "country_code"]
            merged = merged.merge(
                aqueduct_country[aq_cols],
                on="country_code",
                how="left",
                suffixes=("", "_aqueduct"),
            )
            logger.info(f"  After Aqueduct merge: {merged.shape}")

        # Step 4: Merge AQUASTAT (country × year, forward-filled)
        if "year" in merged.columns and "year" in aquastat_df.columns:
            # Prepare AQUASTAT: ensure standardized column names
            aquastat_cols = [c for c in aquastat_df.columns
                            if c not in merged.columns or c in ["country_code", "year"]]
            aquastat_subset = aquastat_df[aquastat_cols].copy()

            # AQUASTAT data is sparse (5-year intervals), so forward-fill to annual
            aquastat_filled = self._fill_aquastat_annual(aquastat_subset)

            merged = merged.merge(
                aquastat_filled,
                on=["country_code", "year"],
                how="left",
                suffixes=("", "_aquastat"),
            )
            logger.info(f"  After AQUASTAT merge: {merged.shape}")

        # Step 5: Compute derived features
        merged = self._compute_derived_features(merged)

        # Step 6: Classify drought risk
        merged = self._classify_drought_risk(merged)

        # Step 7: Add season
        if "date" in merged.columns:
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged["season"] = merged["date"].dt.month.map({
                12: "DJF", 1: "DJF", 2: "DJF",
                3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA",
                9: "SON", 10: "SON", 11: "SON",
            })
            merged["date"] = merged["date"].dt.strftime("%Y-%m")

        # Step 8: Clean up
        # Drop helper columns
        drop_cols = [c for c in merged.columns if c.endswith("_aqueduct") or c.endswith("_aquastat")]
        merged = merged.drop(columns=drop_cols, errors="ignore")

        # Reorder columns
        priority_cols = [
            "country", "country_code", "lat", "lon", "date", "year", "season",
            "water_stress_score", "water_depletion_score", "drought_risk_score",
            "tws_anomaly_cm", "groundwater_anomaly_cm",
            "sdg642_water_stress_pct", "total_renewable_water_km3",
            "total_water_withdrawal_km3", "precipitation_mm",
            "drought_risk_class", "drought_composite_score",
        ]
        available_priority = [c for c in priority_cols if c in merged.columns]
        other_cols = [c for c in merged.columns if c not in available_priority]
        merged = merged[available_priority + other_cols]

        logger.info(f"\nFinal merged dataset: {merged.shape}")
        logger.info(f"Columns: {list(merged.columns)}")

        return merged

    def _aggregate_aqueduct_to_country(self, df):
        """Aggregate catchment-level Aqueduct data to country means."""
        if "country_code" not in df.columns:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove lat/lon from aggregation
        numeric_cols = [c for c in numeric_cols if c not in ["lat", "lon"]]

        agg_dict = {col: "mean" for col in numeric_cols}
        # Also get lat/lon centroids
        if "lat" in df.columns:
            agg_dict["lat"] = "mean"
        if "lon" in df.columns:
            agg_dict["lon"] = "mean"

        country_agg = df.groupby("country_code").agg(agg_dict).reset_index()

        # Round
        for col in country_agg.select_dtypes(include=[np.number]).columns:
            country_agg[col] = country_agg[col].round(4)

        return country_agg

    def _fill_aquastat_annual(self, df):
        """Forward-fill AQUASTAT 5-year data to annual resolution."""
        if "year" not in df.columns or "country_code" not in df.columns:
            return df

        filled = []
        for code in df["country_code"].unique():
            country = df[df["country_code"] == code].copy()
            country = country.sort_values("year")

            if len(country) < 2:
                filled.append(country)
                continue

            # Create full year range
            min_year = int(country["year"].min())
            max_year = int(country["year"].max())
            full_years = pd.DataFrame({"year": range(min_year, max_year + 1)})
            full_years["country_code"] = code

            # Merge and forward-fill
            expanded = full_years.merge(country, on=["country_code", "year"], how="left")
            numeric_cols = expanded.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c != "year"]
            expanded[numeric_cols] = expanded[numeric_cols].interpolate(method="linear")
            expanded = expanded.ffill().bfill()

            filled.append(expanded)

        return pd.concat(filled, ignore_index=True)

    def _compute_derived_features(self, df):
        """Compute derived drought and water stress features."""
        logger.info("Computing derived features...")

        # Drought Composite Score (0-5 scale)
        # Weighted combination of available indicators
        components = []
        weights = []

        if "water_stress_score" in df.columns:
            components.append(df["water_stress_score"].fillna(0))
            weights.append(0.3)

        if "drought_risk_score" in df.columns:
            components.append(df["drought_risk_score"].fillna(0))
            weights.append(0.25)

        if "tws_anomaly_cm" in df.columns:
            # Normalize TWS anomaly to 0-5 scale (more negative = higher score)
            tws_norm = (-df["tws_anomaly_cm"].fillna(0)).clip(-20, 20)
            tws_score = ((tws_norm + 20) / 40 * 5).clip(0, 5)
            components.append(tws_score)
            weights.append(0.25)

        if "groundwater_anomaly_cm" in df.columns:
            gw_norm = (-df["groundwater_anomaly_cm"].fillna(0)).clip(-15, 15)
            gw_score = ((gw_norm + 15) / 30 * 5).clip(0, 5)
            components.append(gw_score)
            weights.append(0.2)

        if components:
            # Normalize weights
            total_weight = sum(weights)
            norm_weights = [w / total_weight for w in weights]

            df["drought_composite_score"] = sum(
                c * w for c, w in zip(components, norm_weights)
            ).round(3)
        else:
            df["drought_composite_score"] = 0

        # Water balance indicator
        if "total_renewable_water_km3" in df.columns and "total_water_withdrawal_km3" in df.columns:
            df["water_balance_km3"] = (
                df["total_renewable_water_km3"] - df["total_water_withdrawal_km3"]
            )

        # TWS anomaly severity (z-score within each country)
        if "tws_anomaly_cm" in df.columns and "country_code" in df.columns:
            df["tws_zscore"] = df.groupby("country_code")["tws_anomaly_cm"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            ).round(3)

        return df

    def _classify_drought_risk(self, df):
        """Classify drought risk based on composite score."""
        if "drought_composite_score" not in df.columns:
            df["drought_risk_class"] = "Unknown"
            return df

        conditions = [
            df["drought_composite_score"] < self.drought_thresholds["Low"][1],
            df["drought_composite_score"] < self.drought_thresholds["Moderate"][1],
            df["drought_composite_score"] < self.drought_thresholds["High"][1],
            df["drought_composite_score"] >= self.drought_thresholds["High"][1],
        ]
        choices = ["Low", "Moderate", "High", "Extreme"]
        df["drought_risk_class"] = np.select(conditions, choices, default="Unknown")

        # Log distribution
        dist = df["drought_risk_class"].value_counts()
        logger.info(f"  Drought risk distribution:\n{dist.to_string()}")

        return df

    def save_consolidated(self, df, output_path=None):
        """Save consolidated dataset to CSV."""
        output = output_path or config.CONSOLIDATED_CSV
        df.to_csv(output, index=False)
        logger.info(f"Saved consolidated dataset ({len(df)} rows) to: {output}")
        return output

    def get_summary(self, df):
        """Get summary statistics of the merged dataset."""
        summary = {
            "total_records": len(df),
            "countries": df["country_code"].nunique() if "country_code" in df.columns else 0,
            "date_range": f"{df['date'].min()} to {df['date'].max()}" if "date" in df.columns else "N/A",
            "features": len(df.columns),
            "drought_distribution": df["drought_risk_class"].value_counts().to_dict()
            if "drought_risk_class" in df.columns else {},
        }
        return summary


def main():
    """Run the data merging pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Load processed data from each source
    aqueduct_path = config.AQUEDUCT_PROCESSED
    aquastat_path = config.AQUASTAT_PROCESSED
    grace_path = config.GRACE_PROCESSED

    for path, name in [(aqueduct_path, "Aqueduct"), (aquastat_path, "AQUASTAT"), (grace_path, "GRACE")]:
        if not path.exists():
            logger.error(f"{name} processed data not found at {path}. Run extractors first.")
            return

    aqueduct_df = pd.read_csv(aqueduct_path)
    aquastat_df = pd.read_csv(aquastat_path)
    grace_df = pd.read_csv(grace_path)

    merger = DataMerger()
    consolidated = merger.merge_all(aqueduct_df, aquastat_df, grace_df)

    # Save
    merger.save_consolidated(consolidated)

    # Print summary
    summary = merger.get_summary(consolidated)
    print(f"\n{'='*60}")
    print("CONSOLIDATED DATASET SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
