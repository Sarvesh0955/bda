"""
Quarterly Update Pipeline
Automated pipeline for downloading, processing, modeling,
and reporting on water stress & drought indicators.
"""
import logging
import time
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class QuarterlyPipeline:
    """Automated quarterly update pipeline for drought tracking."""

    def __init__(self):
        self.log_file = config.PIPELINE_LOG_FILE
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}

    def run(self, skip_download=False):
        """
        Execute the full quarterly pipeline.

        Steps:
        1. Download/refresh data from all sources
        2. Clean and merge datasets
        3. Re-train ML models
        4. Generate reports
        5. Log run metadata
        """
        start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f" QUARTERLY PIPELINE RUN: {self.run_id}")
        logger.info(f"{'='*60}")

        try:
            # Step 1: Data Extraction
            if not skip_download:
                self._step_extract_data()

            # Step 2: Data Cleaning & Merging
            df = self._step_process_data()

            # Step 3: ML Model Training
            self._step_train_models(df)

            # Step 4: Generate Reports
            self._step_generate_reports(df)

            # Step 5: Log results
            elapsed = time.time() - start_time
            self._log_run(elapsed, success=True)

            logger.info(f"\n{'='*60}")
            logger.info(f" PIPELINE COMPLETED in {elapsed:.1f}s")
            logger.info(f"{'='*60}")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Pipeline failed after {elapsed:.1f}s: {e}")
            self._log_run(elapsed, success=False, error=str(e))
            raise

    def _step_extract_data(self):
        """Step 1: Download fresh data from all sources."""
        logger.info("\n📥 Step 1: Extracting data...")

        from src.extractors.aqueduct_extractor import AqueductExtractor
        from src.extractors.aquastat_extractor import AquastatExtractor
        from src.extractors.grace_extractor import GraceExtractor

        # Aqueduct
        aq = AqueductExtractor()
        aq.download_data("baseline_annual", force=True)
        aq_df = aq.extract_features()
        aq.save_processed(aq_df)
        self.metrics["aqueduct_records"] = len(aq_df)

        # AQUASTAT
        aqua = AquastatExtractor()
        aqua.download_data(force=True)
        aqua_df = aqua.extract_features()
        aqua.save_processed(aqua_df)
        self.metrics["aquastat_records"] = len(aqua_df)

        # GRACE
        grace = GraceExtractor()
        grace.download_data(force=True)
        grace_df = grace.extract_features()
        grace.save_processed(grace_df)
        self.metrics["grace_records"] = len(grace_df)

        logger.info(f"  ✅ Data extraction complete")

    def _step_process_data(self):
        """Step 2: Clean and merge all data sources."""
        import pandas as pd
        from src.processing.data_cleaner import DataCleaner
        from src.processing.data_merger import DataMerger

        logger.info("\n🔧 Step 2: Processing data...")

        # Load processed data
        aqueduct_df = pd.read_csv(config.AQUEDUCT_PROCESSED)
        aquastat_df = pd.read_csv(config.AQUASTAT_PROCESSED)
        grace_df = pd.read_csv(config.GRACE_PROCESSED)

        # Clean
        cleaner = DataCleaner()
        aqueduct_df = cleaner.clean_aqueduct(aqueduct_df)
        aquastat_df = cleaner.clean_aquastat(aquastat_df)
        grace_df = cleaner.clean_grace(grace_df)

        # Merge
        merger = DataMerger()
        consolidated = merger.merge_all(aqueduct_df, aquastat_df, grace_df)
        merger.save_consolidated(consolidated)

        self.metrics["consolidated_records"] = len(consolidated)
        self.metrics["countries"] = consolidated["country_code"].nunique() if "country_code" in consolidated.columns else 0

        logger.info(f"  ✅ Data processing complete ({len(consolidated)} records)")
        return consolidated

    def _step_train_models(self, df):
        """Step 3: Re-train ML models on updated data."""
        logger.info("\n🤖 Step 3: Training models...")

        from src.models.drought_classifier import DroughtClassifier
        from src.models.stress_predictor import StressPredictor

        # Drought Classifier
        classifier = DroughtClassifier()
        try:
            X_train, X_test, y_train, y_test, features = classifier.prepare_data(df)
            classifier.train_all(X_train, y_train, tune=False)  # Faster for pipeline
            results = classifier.evaluate(X_test, y_test)
            classifier.save_model()
            self.metrics["classifier_results"] = {
                k: v["f1_weighted"] for k, v in results.items()
            }
            logger.info(f"  ✅ Classifier trained")
        except Exception as e:
            logger.warning(f"  ⚠️ Classifier training failed: {e}")

        # Stress Predictor
        predictor = StressPredictor()
        try:
            df_with_lags = predictor.create_lag_features(df)
            X_train, X_test, y_train, y_test, features = predictor.prepare_data(df_with_lags)
            predictor.train_all(X_train, y_train, tune=False)
            results = predictor.evaluate(X_test, y_test)
            predictor.save_model()
            self.metrics["predictor_results"] = results
            logger.info(f"  ✅ Predictor trained")
        except Exception as e:
            logger.warning(f"  ⚠️ Predictor training failed: {e}")

    def _step_generate_reports(self, df):
        """Step 4: Generate summary reports."""
        logger.info("\n📊 Step 4: Generating reports...")

        # Summary stats
        if "drought_risk_class" in df.columns:
            dist = df["drought_risk_class"].value_counts().to_dict()
            self.metrics["risk_distribution"] = dist

        if "country_code" in df.columns and "drought_composite_score" in df.columns:
            top_risk = (
                df.groupby("country")["drought_composite_score"]
                .mean().sort_values(ascending=False).head(10)
                .to_dict()
            )
            self.metrics["top_risk_countries"] = top_risk

        logger.info(f"  ✅ Reports generated")

    def _log_run(self, elapsed, success=True, error=None):
        """Log pipeline run metadata."""
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "success": success,
            "error": error,
            "metrics": self.metrics,
        }

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"Run logged to: {self.log_file}")


def main():
    """Run the quarterly pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    pipeline = QuarterlyPipeline()
    pipeline.run(skip_download=False)


if __name__ == "__main__":
    main()
