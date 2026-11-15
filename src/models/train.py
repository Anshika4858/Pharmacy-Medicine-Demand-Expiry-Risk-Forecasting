"""
Model Training, Cross-Model Comparison, and Evaluation Pipeline.
ALGIONS AI LABS - Assessment Project

Implements:
1. Baselines:
   - Moving Average Baseline (7-day / 14-day rolling mean)
   - Seasonal Naive Baseline (Same day of previous week lag_7)
2. Machine Learning Candidate Models:
   - Ridge Regression (Regularized Linear Baseline)
   - Random Forest Regressor (Ensemble Bagging)
   - XGBoost Regressor (State-of-the-Art Gradient Boosted Decision Trees)
3. Rigorous Evaluation Metrics:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - WAPE (Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y))
   - R^2 (Coefficient of Determination)
4. Model Selection & Artifact Serialization
"""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression metrics for demand forecasting:
    - MAE: Average absolute unit error
    - RMSE: Penalizes larger deviations more heavily
    - WAPE: sum(|y - y_hat|) / sum(y) - industry standard for inventory forecasting
    - R2: Proportion of variance explained
    """
    y_true = np.asarray(y_true)
    # Clip negative predictions to zero (sales cannot be negative)
    y_pred = np.clip(np.asarray(y_pred), 0.0, None)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    sum_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (sum_actual + 1e-6))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE": round(wape, 4),
        "WAPE_pct": round(wape * 100, 2),
        "R2": round(r2, 4),
    }


class MovingAverageBaseline:
    """Baseline predictor using the 7-day lagged moving average."""
    def __init__(self, rolling_col: str = "rolling_mean_7"):
        self.rolling_col = rolling_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.rolling_col].values


class SeasonalNaiveBaseline:
    """Baseline predictor using the 7-day lag (same day previous week)."""
    def __init__(self, lag_col: str = "lag_7"):
        self.lag_col = lag_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag_col].values


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Trains all candidate models, evaluates on validation set,
    selects champion model, and validates on held-out test set.
    """
    logger.info("Loading chronologically split feature datasets...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val_features.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_features.csv")
    
    with open(PROCESSED_DATA_DIR / "feature_names.json", "r") as f:
        feature_cols = json.load(f)
        
    X_train = train_df[feature_cols]
    y_train = train_df["units_sold"].values
    
    X_val = val_df[feature_cols]
    y_val = val_df["units_sold"].values
    
    X_test = test_df[feature_cols]
    y_test = test_df["units_sold"].values
    
    logger.info("Training features shape: %s | Val: %s | Test: %s", X_train.shape, X_val.shape, X_test.shape)
    
    # Define models dictionary
    models = {
        "Baseline_7D_Moving_Average": MovingAverageBaseline("rolling_mean_7"),
        "Baseline_Seasonal_Naive_Lag7": SeasonalNaiveBaseline("lag_7"),
        "Ridge_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=RANDOM_SEED))
        ]),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "XGBoost_Regressor": xgb.XGBRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
    }
    
    val_results = {}
    fitted_models = {}
    
    for name, model in models.items():
        logger.info("Fitting and evaluating model: %s", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        preds_val = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, preds_val)
        val_results[name] = val_metrics
        logger.info(" -> Val Metrics: MAE=%.2f, RMSE=%.2f, WAPE=%.2f%%, R2=%.3f",
                    val_metrics["MAE"], val_metrics["RMSE"], val_metrics["WAPE_pct"], val_metrics["R2"])
        
    # Select champion model based on lowest Validation WAPE
    champion_name = min(
        [k for k in models.keys() if not k.startswith("Baseline")],
        key=lambda k: val_results[k]["WAPE"]
    )
    logger.info("Selected Champion ML Model: %s (Val WAPE: %.2f%%)", champion_name, val_results[champion_name]["WAPE_pct"])
    
    # Evaluate all models on held-out test set
    test_results = {}
    for name, model in fitted_models.items():
        preds_test = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, preds_test)
        test_results[name] = test_metrics
        logger.info(" -> Test Metrics [%s]: MAE=%.2f, RMSE=%.2f, WAPE=%.2f%%, R2=%.3f",
                    name, test_metrics["MAE"], test_metrics["RMSE"], test_metrics["WAPE_pct"], test_metrics["R2"])
        
    champion_model = fitted_models[champion_name]
    
    # Extract Feature Importance for Champion Model
    feature_importance_dict = {}
    if champion_name == "XGBoost_Regressor":
        importances = champion_model.feature_importances_
        for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            feature_importance_dict[feat] = float(round(imp, 5))
    elif champion_name == "Random_Forest":
        importances = champion_model.feature_importances_
        for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            feature_importance_dict[feat] = float(round(imp, 5))
            
    # Serialize champion model and metadata
    model_save_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(champion_model, model_save_path)
    logger.info("Champion model saved to %s", model_save_path)
    
    metadata = {
        "champion_model_name": champion_name,
        "training_timestamp": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "dataset_shapes": {
            "train": list(X_train.shape),
            "val": list(X_val.shape),
            "test": list(X_test.shape),
        },
        "validation_metrics": val_results,
        "held_out_test_metrics": test_results,
        "top_features": dict(list(feature_importance_dict.items())[:15]),
    }
    
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    with open(MODEL_DIR / "evaluation_metrics.json", "w") as f:
        json.dump({
            "validation_comparison": val_results,
            "test_comparison": test_results,
            "champion_model": champion_name
        }, f, indent=2)
        
    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance_dict, f, indent=2)
        
    logger.info("Successfully trained, evaluated, and saved all models.")
    return metadata


if __name__ == "__main__":
    meta = train_and_evaluate_all_models()
    print("\n--- TEST SET LEADERBOARD ---")
    for m, res in meta["held_out_test_metrics"].items():
        print(f"{m:30s} | MAE: {res['MAE']:5.2f} | RMSE: {res['RMSE']:5.2f} | WAPE: {res['WAPE_pct']:5.2f}% | R2: {res['R2']:5.3f}")

