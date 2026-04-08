"""
EDA-15: Train ML Models on Real Processed Data
Trains drought risk classifiers and water stress predictors.
"""
import logging
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import config
from src.models.drought_classifier import DroughtClassifier
from src.models.stress_predictor import StressPredictor


def train_classifier(df):
    """Train drought risk classification models."""
    logger.info("=" * 60)
    logger.info("TASK A: Drought Risk Classification")
    logger.info("=" * 60)

    classifier = DroughtClassifier()

    # Update feature list to match our real data columns
    classifier.FEATURE_COLS = [
        "water_stress_score", "water_depletion_score", "drought_risk_score",
        "interannual_variability", "seasonal_variability", "groundwater_decline_score",
        "flood_risk_score", "overall_water_risk",
        "tws_anomaly_cm", "groundwater_anomaly_cm", "uncertainty_cm",
        "tws_3month_avg", "tws_6month_avg", "tws_12month_avg",
        "tws_rate_of_change", "tws_cumulative_change", "tws_zscore",
        "tws_trend_cm_yr",
        "sdg642_water_stress_pct", "total_renewable_water_km3",
        "total_water_withdrawal_km3", "precipitation_mm",
        "drought_composite_score",
    ]

    # Prepare data
    X_train, X_test, y_train, y_test, features = classifier.prepare_data(df)

    # Train
    classifier.train_all(X_train, y_train, tune=True)

    # Evaluate
    results = classifier.evaluate(X_test, y_test)

    # Feature importance
    importance = classifier.get_feature_importance()
    if importance is not None:
        logger.info(f"\nTop 10 Features ({classifier.best_model_name}):")
        logger.info(f"\n{importance.head(10).to_string(index=False)}")

    # Save all models
    for name in classifier.models:
        classifier.save_model(name)

    return classifier, results


def train_predictor(df):
    """Train water stress prediction models."""
    logger.info("\n" + "=" * 60)
    logger.info("TASK B: Water Stress Prediction (3-month forecast)")
    logger.info("=" * 60)

    predictor = StressPredictor()

    # Create lag features
    df_lagged = predictor.create_lag_features(df)

    if len(df_lagged) == 0:
        logger.error("No data after creating lag features!")
        return None, {}

    # Prepare data
    X_train, X_test, y_train, y_test, features = predictor.prepare_data(df_lagged)

    # Train
    predictor.train_all(X_train, y_train, tune=True)

    # Evaluate
    results = predictor.evaluate(X_test, y_test)

    # Save all models
    for name in predictor.models:
        predictor.save_model(name)

    return predictor, results


def log_run(clf_results, reg_results, df):
    """Log training run to pipeline_runs.log."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_rows": len(df),
        "data_columns": len(df.columns),
        "countries": df["country_code"].nunique() if "country_code" in df.columns else 0,
        "date_range": f"{df['date'].min()} to {df['date'].max()}" if "date" in df.columns else "N/A",
        "classification": {},
        "regression": {},
    }

    for name, metrics in clf_results.items():
        log_entry["classification"][name] = {
            "f1_weighted": metrics["f1_weighted"],
            "accuracy": metrics["accuracy"],
        }

    for name, metrics in reg_results.items():
        log_entry["regression"][name] = metrics

    with open(config.PIPELINE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Run logged to: {config.PIPELINE_LOG_FILE}")


if __name__ == "__main__":
    logger.info("🤖 EDA-15: Training ML Models on Real Data")
    logger.info("=" * 60)

    # Load consolidated data
    data_path = config.CONSOLIDATED_CSV
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.error("Run 'python process_real_data.py' first.")
        exit(1)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Countries: {df['country_code'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Risk distribution:\n{df['drought_risk_class'].value_counts().to_string()}")

    # Train classifier
    classifier, clf_results = train_classifier(df)

    # Train predictor
    predictor, reg_results = train_predictor(df)

    # Log run
    log_run(clf_results, reg_results, df)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("\n📊 Classification Results:")
    for name, metrics in clf_results.items():
        logger.info(f"  {name}: F1={metrics['f1_weighted']:.4f}, Acc={metrics['accuracy']:.4f}")

    logger.info("\n📈 Regression Results:")
    for name, metrics in reg_results.items():
        logger.info(f"  {name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")

    logger.info(f"\n📁 Models saved to: {config.MODELS_DIR}")
    import os
    for f in sorted(config.MODELS_DIR.glob("*.joblib")):
        logger.info(f"  {f.name} ({os.path.getsize(f)/1024:.0f} KB)")
