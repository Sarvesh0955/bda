"""
Drought Risk Classifier
ML models for classifying drought risk into categories
(Low / Moderate / High / Extreme) using ensemble methods.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Using Random Forest only.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class DroughtClassifier:
    """
    Classify drought risk level based on water stress indicators.

    Supports Random Forest and XGBoost classifiers with
    hyperparameter tuning and SHAP explainability.
    """

    TARGET_COL = "drought_risk_class"
    RISK_CLASSES = ["Low", "Moderate", "High", "Extreme"]

    # Features to use for classification
    FEATURE_COLS = [
        "water_stress_score", "water_depletion_score", "drought_risk_score",
        "interannual_variability", "seasonal_variability", "groundwater_decline_score",
        "flood_risk_score", "overall_water_risk",
        "tws_anomaly_cm", "groundwater_anomaly_cm",
        "tws_3month_avg", "tws_6month_avg", "tws_12month_avg",
        "tws_rate_of_change", "tws_cumulative_change",
        "gw_6month_avg", "gw_12month_avg",
        "sdg642_water_stress_pct", "total_renewable_water_km3",
        "total_water_withdrawal_km3", "precipitation_mm",
        "drought_composite_score", "tws_zscore",
    ]

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else config.MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model_name = None
        self.feature_names = []

    def prepare_data(self, df, test_size_year=None):
        """
        Prepare features and target from the consolidated dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Consolidated dataset with drought_risk_class column
        test_size_year : int
            Year to split on (temporal split). Data before this year = train.

        Returns
        -------
        tuple : (X_train, X_test, y_train, y_test, feature_names)
        """
        if self.TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COL}' not found in data")

        # Select available features
        available_features = [c for c in self.FEATURE_COLS if c in df.columns]
        if not available_features:
            # Fallback: use all numeric columns except target
            available_features = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in [self.TARGET_COL, "year"]
            ]

        logger.info(f"Using {len(available_features)} features: {available_features}")
        self.feature_names = available_features

        # Extract features and target
        X = df[available_features].copy()
        y = df[self.TARGET_COL].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Encode target
        self.label_encoder.fit(self.RISK_CLASSES)
        y_encoded = self.label_encoder.transform(y)

        # Temporal split
        split_year = test_size_year or config.TEMPORAL_SPLIT_YEAR
        if "year" in df.columns:
            train_mask = df["year"] < split_year
            test_mask = df["year"] >= split_year

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                # Fallback to random split
                logger.warning(f"Temporal split at {split_year} yields empty set. Using 80/20 random split.")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                X_train = X[train_mask]
                X_test = X[test_mask]
                y_train = y_encoded[train_mask]
                y_test = y_encoded[test_mask]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Class distribution (train): {np.bincount(y_train)}")
        logger.info(f"Class distribution (test):  {np.bincount(y_test)}")

        return X_train, X_test, y_train, y_test, available_features

    def train_random_forest(self, X_train, y_train, tune=True):
        """Train a Random Forest classifier."""
        logger.info("Training Random Forest...")

        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, 20],
                "min_samples_split": [3, 5],
                "min_samples_leaf": [1, 2],
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=min(config.CV_FOLDS, 3), shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring=config.SCORING_CLASSIFICATION,
                n_jobs=-1, verbose=0, refit=True,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"  Best RF params: {grid_search.best_params_}")
            logger.info(f"  Best CV score: {grid_search.best_score_:.4f}")
        else:
            model = RandomForestClassifier(**config.RF_PARAMS)
            model.fit(X_train, y_train)

        self.models["RandomForest"] = model
        return model

    def train_xgboost(self, X_train, y_train, tune=True):
        """Train an XGBoost classifier."""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not available. Skipping.")
            return None

        logger.info("Training XGBoost...")

        if tune:
            param_grid = {
                "n_estimators": [200, 300],
                "max_depth": [6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            }
            base_model = XGBClassifier(
                random_state=42, eval_metric="mlogloss",
                use_label_encoder=False,
            )
            cv = StratifiedKFold(n_splits=min(config.CV_FOLDS, 3), shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring=config.SCORING_CLASSIFICATION,
                n_jobs=-1, verbose=0, refit=True,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"  Best XGB params: {grid_search.best_params_}")
            logger.info(f"  Best CV score: {grid_search.best_score_:.4f}")
        else:
            model = XGBClassifier(**config.XGB_PARAMS, use_label_encoder=False)
            model.fit(X_train, y_train)

        self.models["XGBoost"] = model
        return model

    def train_all(self, X_train, y_train, tune=True):
        """Train all available models."""
        self.train_random_forest(X_train, y_train, tune=tune)
        self.train_xgboost(X_train, y_train, tune=tune)
        return self.models

    def evaluate(self, X_test, y_test):
        """
        Evaluate all trained models.

        Returns
        -------
        dict : model name → evaluation metrics
        """
        results = {}
        best_f1 = -1

        for name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {name}")
            logger.info(f"{'='*50}")

            y_pred = model.predict(X_test)

            # Metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)

            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
            )
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "f1_weighted": round(f1, 4),
                "accuracy": round(acc, 4),
                "classification_report": report,
                "confusion_matrix": cm,
            }

            logger.info(f"  F1 (weighted): {f1:.4f}")
            logger.info(f"  Accuracy: {acc:.4f}")
            logger.info(f"\n{classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)}")

            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name

        logger.info(f"\nBest model: {self.best_model_name} (F1={best_f1:.4f})")
        return results

    def get_feature_importance(self, model_name=None):
        """Get feature importance from the specified model."""
        name = model_name or self.best_model_name
        if name not in self.models:
            return None

        model = self.models[name]
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": self.feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            return importance
        return None

    def get_shap_values(self, X, model_name=None, max_samples=500):
        """Compute SHAP values for model explainability."""
        if not HAS_SHAP:
            logger.warning("SHAP not installed.")
            return None

        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            return None

        # Sample data if too large
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
        else:
            X_sample = X

        logger.info(f"Computing SHAP values for {name} on {len(X_sample)} samples...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        return shap_values, X_sample

    def predict(self, X, model_name=None):
        """Make predictions using the best model."""
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found. Train first.")

        y_pred = model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X, model_name=None):
        """Get prediction probabilities."""
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found.")

        proba = model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.label_encoder.classes_)

    def save_model(self, model_name=None):
        """Save model to disk."""
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            return None

        filepath = self.model_dir / f"drought_classifier_{name.lower()}.joblib"
        joblib.dump({
            "model": model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "model_name": name,
        }, filepath)
        logger.info(f"Model saved to: {filepath}")
        return filepath

    def load_model(self, filepath):
        """Load model from disk."""
        data = joblib.load(filepath)
        name = data["model_name"]
        self.models[name] = data["model"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        self.best_model_name = name
        logger.info(f"Loaded model: {name}")
        return data["model"]


def main():
    """Run drought classification pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Load consolidated data
    data_path = config.CONSOLIDATED_CSV
    if not data_path.exists():
        logger.error(f"Consolidated data not found at {data_path}. Run data merger first.")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")

    # Initialize classifier
    classifier = DroughtClassifier()

    # Prepare data
    X_train, X_test, y_train, y_test, features = classifier.prepare_data(df)

    # Train models
    classifier.train_all(X_train, y_train, tune=True)

    # Evaluate
    results = classifier.evaluate(X_test, y_test)

    # Feature importance
    importance = classifier.get_feature_importance()
    if importance is not None:
        print(f"\nTop 10 Important Features ({classifier.best_model_name}):")
        print(importance.head(10).to_string(index=False))

    # Save best model
    classifier.save_model()


if __name__ == "__main__":
    main()
