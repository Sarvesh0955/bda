"""
Water Stress Predictor
Regression models for predicting future water stress scores
based on temporal trends and multi-source indicators.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class StressPredictor:
    """
    Predict future water stress scores (regression).

    Uses lagged features and temporal indicators to forecast
    water stress 1 quarter (3 months) ahead.
    """

    TARGET_COL = "drought_composite_score"
    FORECAST_HORIZON = 3  # months ahead

    FEATURE_COLS = [
        "water_stress_score", "water_depletion_score", "drought_risk_score",
        "tws_anomaly_cm", "groundwater_anomaly_cm",
        "tws_3month_avg", "tws_6month_avg", "tws_12month_avg",
        "tws_rate_of_change",
        "sdg642_water_stress_pct", "total_renewable_water_km3",
        "precipitation_mm",
    ]

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else config.MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model_name = None
        self.feature_names = []

    def create_lag_features(self, df, target_col=None):
        """
        Create lagged features for time series prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with country_code and date columns
        target_col : str
            Column to create lags for

        Returns
        -------
        pd.DataFrame : data with lag features added
        """
        target = target_col or self.TARGET_COL
        if target not in df.columns:
            logger.warning(f"Target '{target}' not found. Available: {list(df.columns)}")
            return df

        result = df.copy()

        # Sort by country and date
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"], errors="coerce")
        sort_cols = [c for c in ["country_code", "date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(sort_cols)

        # Create lag features per country
        if "country_code" in result.columns:
            for lag in config.LAG_PERIODS:
                result[f"{target}_lag{lag}"] = result.groupby("country_code")[target].shift(lag)

            # Rolling statistics
            for window in config.ROLLING_WINDOWS:
                result[f"{target}_rolling_mean_{window}"] = (
                    result.groupby("country_code")[target]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                result[f"{target}_rolling_std_{window}"] = (
                    result.groupby("country_code")[target]
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )

            # Create target: value N months in the future
            result[f"{target}_future"] = (
                result.groupby("country_code")[target].shift(-self.FORECAST_HORIZON)
            )
        else:
            for lag in config.LAG_PERIODS:
                result[f"{target}_lag{lag}"] = result[target].shift(lag)
            result[f"{target}_future"] = result[target].shift(-self.FORECAST_HORIZON)

        # Add temporal features
        if "date" in result.columns:
            result["month"] = result["date"].dt.month
            result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
            result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

        # Drop rows with NaN target
        target_future = f"{target}_future"
        if target_future in result.columns:
            before = len(result)
            result = result.dropna(subset=[target_future])
            logger.info(f"  Dropped {before - len(result)} rows without future target")

        return result

    def prepare_data(self, df, test_size_year=None):
        """
        Prepare features for regression.

        Returns
        -------
        tuple : (X_train, X_test, y_train, y_test, feature_names)
        """
        target_future = f"{self.TARGET_COL}_future"
        if target_future not in df.columns:
            df = self.create_lag_features(df)

        # Build feature list
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        lag_cols = [c for c in df.columns if "_lag" in c or "_rolling_" in c]
        temporal_cols = [c for c in ["month_sin", "month_cos"] if c in df.columns]
        all_features = list(set(available + lag_cols + temporal_cols))

        self.feature_names = all_features
        logger.info(f"Using {len(all_features)} features for prediction")

        X = df[all_features].fillna(0)
        y = df[target_future]

        # Temporal split
        split_year = test_size_year or config.TEMPORAL_SPLIT_YEAR
        if "year" in df.columns:
            train_mask = df["year"] < split_year
            test_mask = df["year"] >= split_year

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=all_features, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=all_features, index=X_test.index
        )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test, all_features

    def train_ridge(self, X_train, y_train, tune=True):
        """Train Ridge Regression baseline."""
        logger.info("Training Ridge Regression...")

        if tune:
            param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            model = GridSearchCV(
                Ridge(), param_grid,
                cv=min(5, len(X_train) // 10),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            logger.info(f"  Best alpha: {model.best_params_['alpha']}")
            self.models["Ridge"] = model.best_estimator_
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            self.models["Ridge"] = model

        return self.models["Ridge"]

    def train_xgboost(self, X_train, y_train, tune=True):
        """Train XGBoost Regressor."""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not available.")
            return None

        logger.info("Training XGBoost Regressor...")

        if tune:
            param_grid = {
                "n_estimators": [200, 300],
                "max_depth": [6, 8],
                "learning_rate": [0.05, 0.1],
            }
            model = GridSearchCV(
                XGBRegressor(random_state=42, subsample=0.8, colsample_bytree=0.8),
                param_grid,
                cv=min(5, len(X_train) // 10),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            logger.info(f"  Best params: {model.best_params_}")
            self.models["XGBoost_Reg"] = model.best_estimator_
        else:
            model = XGBRegressor(**config.XGB_REG_PARAMS)
            model.fit(X_train, y_train)
            self.models["XGBoost_Reg"] = model

        return self.models["XGBoost_Reg"]

    def train_all(self, X_train, y_train, tune=True):
        """Train all regression models."""
        self.train_ridge(X_train, y_train, tune=tune)
        self.train_xgboost(X_train, y_train, tune=tune)
        return self.models

    def evaluate(self, X_test, y_test):
        """Evaluate all models."""
        results = {}
        best_rmse = float("inf")

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
            }

            logger.info(f"\n{name}:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE:  {mae:.4f}")
            logger.info(f"  R²:   {r2:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                self.best_model_name = name

        logger.info(f"\nBest model: {self.best_model_name} (RMSE={best_rmse:.4f})")
        return results

    def predict(self, X, model_name=None):
        """Make predictions."""
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found.")
        return model.predict(X)

    def save_model(self, model_name=None):
        """Save model to disk."""
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            return None

        filepath = self.model_dir / f"stress_predictor_{name.lower()}.joblib"
        joblib.dump({
            "model": model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_name": name,
            "forecast_horizon": self.FORECAST_HORIZON,
        }, filepath)
        logger.info(f"Model saved to: {filepath}")
        return filepath

    def load_model(self, filepath):
        """Load model from disk."""
        data = joblib.load(filepath)
        name = data["model_name"]
        self.models[name] = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.best_model_name = name
        return data["model"]


def main():
    """Run stress prediction pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    data_path = config.CONSOLIDATED_CSV
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    predictor = StressPredictor()

    # Create lag features
    df = predictor.create_lag_features(df)

    # Prepare data
    X_train, X_test, y_train, y_test, features = predictor.prepare_data(df)

    # Train
    predictor.train_all(X_train, y_train, tune=True)

    # Evaluate
    results = predictor.evaluate(X_test, y_test)

    # Save
    predictor.save_model()

    print("\nPrediction Results:")
    for name, metrics in results.items():
        print(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
