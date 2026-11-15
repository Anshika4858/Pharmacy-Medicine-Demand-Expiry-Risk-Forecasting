"""
Feature Engineering and Chronological Splitting Pipeline.
ALGIONS AI LABS - Assessment Project

Constructs leakage-safe predictive features for pharmacy demand forecasting:
- Strictly shifted lag features (t-1, t-2, t-3, t-7, t-14, t-21, t-28, t-30)
- Strictly shifted rolling summary statistics (means, standard deviations, min, max, ratios)
- Cyclical calendar and seasonal representations (sine/cosine day of year, day of week)
- Commercial & clinical product attributes (profit margins, Rx status, cold chain requirement)
- Chronological train/val/test partitions avoiding any future lookahead leakage
"""
import logging
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
    LAG_DAYS,
    ROLLING_WINDOWS,
)
from src.data.quality_checks import verify_temporal_leakage_safety

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar and cyclical seasonal features."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    
    # Cyclical day of year representation
    day_of_year = dt.dt.dayofyear
    df["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return df


def create_lag_and_rolling_features(
    sales_df: pd.DataFrame,
    lag_days: List[int] = LAG_DAYS,
    rolling_windows: List[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Constructs lag and rolling statistics on units_sold strictly grouped by product_id.
    CRITICAL FOR LEAKAGE PREVENTION:
    All rolling windows operate on shift(1) so that current day target units_sold
    is NEVER included in feature calculations.
    """
    sales_df = sales_df.sort_values(["product_id", "date"]).reset_index(drop=True)
    grouped = sales_df.groupby("product_id")["units_sold"]
    
    # 1. Discrete Lags
    for lag in lag_days:
        sales_df[f"lag_{lag}"] = grouped.shift(lag)
        
    # 2. Rolling features based on shifted target (shift(1))
    shifted_target = grouped.shift(1)
    
    for window in rolling_windows:
        rolling_obj = shifted_target.groupby(sales_df["product_id"]).rolling(window=window, min_periods=window)
        
        # Mean & Std
        sales_df[f"rolling_mean_{window}"] = rolling_obj.mean().reset_index(level=0, drop=True)
        sales_df[f"rolling_std_{window}"] = rolling_obj.std().reset_index(level=0, drop=True).fillna(0.0)
        sales_df[f"rolling_max_{window}"] = rolling_obj.max().reset_index(level=0, drop=True)
        sales_df[f"rolling_min_{window}"] = rolling_obj.min().reset_index(level=0, drop=True)
        
    # Velocity / Momentum Ratio: Short term velocity (7d) vs Medium term velocity (28d)
    sales_df["velocity_ratio_7_28"] = (
        (sales_df["rolling_mean_7"] + 1e-4) / (sales_df["rolling_mean_28"] + 1e-4)
    )
    
    return sales_df


def merge_product_metadata(sales_df: pd.DataFrame, meds_df: pd.DataFrame) -> pd.DataFrame:
    """Enriches daily sales records with static product properties."""
    # Drop columns from meds_subset that already exist in sales_df (except product_id)
    overlapping_cols = [c for c in meds_df.columns if c in sales_df.columns and c != "product_id"]
    meds_clean = meds_df.drop(columns=overlapping_cols)
    
    merged = sales_df.merge(meds_clean, on="product_id", how="left")
    
    # Compute commercial features
    merged["profit_margin"] = (
        (merged["effective_price"] - merged["unit_cost"]) / (merged["effective_price"] + 1e-4)
    ).clip(lower=0.0)
    
    merged["is_prescription_required"] = merged["is_prescription_required"].astype(int)
    merged["is_cold_chain"] = (merged["storage_condition"] == "Cold Chain (2-8°C)").astype(int)
    
    # One-hot encode therapeutic category
    category_dummies = pd.get_dummies(merged["category"], prefix="cat", drop_first=False).astype(int)
    merged = pd.concat([merged, category_dummies], axis=1)
    
    return merged


def build_feature_pipeline() -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes full feature generation pipeline and returns curated feature matrix
    along with canonical feature names.
    """
    sales_path = RAW_DATA_DIR / "daily_sales.csv"
    meds_path = RAW_DATA_DIR / "medicines.csv"
    
    sales_df = pd.read_csv(sales_path)
    meds_df = pd.read_csv(meds_path)
    
    logger.info("Building features from %d raw sales records...", len(sales_df))
    
    # 1. Temporal calendar features
    df = create_time_features(sales_df)
    
    # 2. Shifted lag and rolling statistics
    df = create_lag_and_rolling_features(df)
    
    # 3. Product catalog metadata
    df = merge_product_metadata(df, meds_df)
    
    # 4. Drop warm-up period (first 30 days where lag_30 is NaN)
    # This ensures zero NaN values in training and eliminates future imputation artifacts
    initial_len = len(df)
    df = df.dropna(subset=[f"lag_{max(LAG_DAYS)}", f"rolling_mean_{max(ROLLING_WINDOWS)}"]).reset_index(drop=True)
    logger.info("Dropped %d warm-up records with unpopulated lags (retained %d clean records).", initial_len - len(df), len(df))
    
    # Define canonical feature columns used for ML modeling
    feature_cols = [
        # Lags
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28", "lag_30",
        # Rolling statistics
        "rolling_mean_7", "rolling_std_7", "rolling_max_7", "rolling_min_7",
        "rolling_mean_14", "rolling_std_14", "rolling_max_14", "rolling_min_14",
        "rolling_mean_28", "rolling_std_28", "rolling_max_28", "rolling_min_28",
        "velocity_ratio_7_28",
        # Calendar & Seasonality
        "day_of_week", "day_of_month", "month", "quarter", "is_weekend",
        "sin_day_of_year", "cos_day_of_year",
        # Pricing, Promotions & Catalog Attributes
        "effective_price", "unit_cost", "profit_margin", "discount_pct",
        "is_promotion", "is_prescription_required", "is_cold_chain", "shelf_life_months",
    ]
    
    # Add one-hot category columns
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    feature_cols.extend(sorted(cat_cols))
    
    logger.info("Constructed %d total model features.", len(feature_cols))
    return df, feature_cols


def partition_and_save_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Partitions the feature matrix strictly chronologically into Train, Validation, and Test sets.
    Saves processed CSV files and metadata.
    """
    df, feature_cols = build_feature_pipeline()
    
    # Chronological Split
    train_df = df[df["date"] <= TRAIN_END_DATE].copy()
    val_df = df[(df["date"] > TRAIN_END_DATE) & (df["date"] <= VAL_END_DATE)].copy()
    test_df = df[df["date"] >= TEST_START_DATE].copy()
    
    # Leakage check
    leakage_audit = verify_temporal_leakage_safety(
        train_df["date"], val_df["date"], test_df["date"]
    )
    if not leakage_audit["overall_leakage_safe"]:
        raise ValueError(f"Temporal leakage detected! Audit: {leakage_audit}")
        
    logger.info("Chronological Split Verification: %s", leakage_audit["status"])
    logger.info(" - Train set: %d records (%s to %s)", len(train_df), train_df["date"].min(), train_df["date"].max())
    logger.info(" - Val set:   %d records (%s to %s)", len(val_df), val_df["date"].min(), val_df["date"].max())
    logger.info(" - Test set:  %d records (%s to %s)", len(test_df), test_df["date"].min(), test_df["date"].max())
    
    # Save processed files
    train_path = PROCESSED_DATA_DIR / "train_features.csv"
    val_path = PROCESSED_DATA_DIR / "val_features.csv"
    test_path = PROCESSED_DATA_DIR / "test_features.csv"
    feat_path = PROCESSED_DATA_DIR / "feature_names.json"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
        
    logger.info("Saved processed datasets and feature list successfully.")
    return train_df, val_df, test_df, feature_cols


if __name__ == "__main__":
    partition_and_save_datasets()
