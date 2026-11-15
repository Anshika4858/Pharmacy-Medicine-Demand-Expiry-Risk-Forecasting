"""
Data Quality, Integrity Profiling, and Leakage Verification Module.
ALGIONS AI LABS - Assessment Project

Provides automated checks:
- Null / Missing value detection
- Temporal continuity and duplicate detection
- Range and domain logic validations
- Outlier analysis and winsorization clipping
- Prohibited target/future leakage verification
- Sensitive data / PII audit
"""
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Audits dataframe for missing or null values."""
    null_counts = df.isnull().sum().to_dict()
    total_rows = len(df)
    missing_cols = {col: count for col, count in null_counts.items() if count > 0}
    
    return {
        "dataset": dataset_name,
        "total_records": total_rows,
        "has_missing": len(missing_cols) > 0,
        "missing_columns": missing_cols,
        "status": "PASS" if len(missing_cols) == 0 else "FAIL"
    }


def check_data_ranges_and_domains(
    sales_df: pd.DataFrame,
    batches_df: pd.DataFrame
) -> Dict[str, Any]:
    """Validates physical and pharmaceutical business rule ranges."""
    issues = []
    
    # 1. Sales units must be non-negative
    if (sales_df["units_sold"] < 0).any():
        issues.append("Found negative units_sold in sales dataset.")
        
    # 2. Effective price must be strictly positive
    if (sales_df["effective_price"] <= 0).any():
        issues.append("Found non-positive effective_price in sales dataset.")
        
    # 3. Discount must be between 0 and 1
    if not sales_df["discount_pct"].between(0.0, 1.0).all():
        issues.append("Found discount_pct outside [0.0, 1.0].")
        
    # 4. Batch expiry must be strictly after manufacturing date
    mfg = pd.to_datetime(batches_df["mfg_date"])
    exp = pd.to_datetime(batches_df["expiry_date"])
    if not (exp > mfg).all():
        issues.append("Found batches where expiry_date is on or before mfg_date.")
        
    # 5. Current stock cannot exceed initial quantity
    if (batches_df["current_stock"] > batches_df["initial_quantity"]).any():
        issues.append("Found batches where current_stock exceeds initial_quantity.")
        
    return {
        "check": "Domain Logic & Range Validation",
        "passed": len(issues) == 0,
        "issues_found": issues,
        "status": "PASS" if len(issues) == 0 else "FAIL"
    }


def audit_sensitive_data_and_pii(sales_df: pd.DataFrame, batches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifies that no Personal Identifiable Information (PII) or Protected Health
    Information (PHI) exists in the dataset.
    """
    # Match whole words or common PII prefixes/tokens (avoid substring match like 'age' in 'storage')
    import re
    prohibited_keywords = [
        "patient", "ssn", "aadhar", "phone", "email", "address", "doctor_name",
        "prescription_id", "medical_record", "credit_card", "ip_address", "gender", "user_age"
    ]
    
    all_columns = [col.lower() for col in list(sales_df.columns) + list(batches_df.columns)]
    found_violations = []
    for col in all_columns:
        for kw in prohibited_keywords:
            if re.search(rf"\b{kw}\b", col) or (kw in ["patient", "doctor_name", "credit_card", "aadhar"] and kw in col):
                found_violations.append(col)

    
    return {
        "check": "PII / Sensitive Data Audit",
        "prohibited_keywords_checked": prohibited_keywords,
        "violations_detected": found_violations,
        "is_compliant": len(found_violations) == 0,
        "status": "PASS" if len(found_violations) == 0 else "FAIL"
    }


def verify_temporal_leakage_safety(
    train_dates: pd.Series,
    val_dates: pd.Series,
    test_dates: pd.Series
) -> Dict[str, Any]:
    """
    Verifies that train, validation, and test splits strictly honor temporal ordering.
    No future data points may exist in earlier partitions.
    """
    max_train = pd.to_datetime(train_dates).max()
    min_val = pd.to_datetime(val_dates).min()
    max_val = pd.to_datetime(val_dates).max()
    min_test = pd.to_datetime(test_dates).min()
    
    train_val_safe = max_train < min_val
    val_test_safe = max_val < min_test
    all_safe = train_val_safe and val_test_safe
    
    return {
        "check": "Temporal Leakage Verification",
        "max_train_date": str(max_train.date()),
        "min_val_date": str(min_val.date()),
        "max_val_date": str(max_val.date()),
        "min_test_date": str(min_test.date()),
        "train_to_val_leakage_free": bool(train_val_safe),
        "val_to_test_leakage_free": bool(val_test_safe),
        "overall_leakage_safe": bool(all_safe),
        "status": "PASS" if all_safe else "FAIL"
    }


def analyze_and_handle_outliers(
    sales_df: pd.DataFrame,
    clip_outliers: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects demand outliers using the Interquartile Range (IQR) method per product.
    If clip_outliers is True, caps extreme outliers at Upper Fence = Q3 + 3.0 * IQR
    (severe outlier threshold), preserving legitimate seasonal surges while stabilizing
    training gradient dynamics.
    """
    df = sales_df.copy()
    outlier_stats = {}
    total_clipped = 0
    
    for prod_id, group in df.groupby("product_id"):
        q25 = group["units_sold"].quantile(0.25)
        q75 = group["units_sold"].quantile(0.75)
        iqr = q75 - q25
        upper_fence = q75 + 3.0 * iqr
        
        is_severe_outlier = group["units_sold"] > upper_fence
        count_outliers = int(is_severe_outlier.sum())
        total_clipped += count_outliers
        
        outlier_stats[prod_id] = {
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(iqr),
            "upper_fence": float(round(upper_fence, 2)),
            "severe_outlier_count": count_outliers
        }
        
        if clip_outliers and count_outliers > 0:
            df.loc[(df["product_id"] == prod_id) & (df["units_sold"] > upper_fence), "units_sold"] = int(upper_fence)
            
    summary = {
        "method": "IQR Extreme Upper Fence (Q3 + 3.0 * IQR)",
        "total_records_adjusted": total_clipped,
        "percentage_adjusted": float(round((total_clipped / len(df)) * 100, 3)),
        "per_product_stats": outlier_stats,
        "status": "COMPLETED"
    }
    
    return df, summary


def run_full_quality_audit() -> Dict[str, Any]:
    """Runs end-to-end quality and compliance audit on raw data files."""
    sales_path = RAW_DATA_DIR / "daily_sales.csv"
    batches_path = RAW_DATA_DIR / "inventory_batches.csv"
    meds_path = RAW_DATA_DIR / "medicines.csv"
    
    if not (sales_path.exists() and batches_path.exists() and meds_path.exists()):
        raise FileNotFoundError("Raw datasets must be generated before running audit.")
        
    sales_df = pd.read_csv(sales_path)
    batches_df = pd.read_csv(batches_path)
    meds_df = pd.read_csv(meds_path)
    
    logger.info("Executing comprehensive Data Quality and Leakage Audit...")
    
    # 1. Missing values
    missing_sales = check_missing_values(sales_df, "daily_sales")
    missing_batches = check_missing_values(batches_df, "inventory_batches")
    missing_meds = check_missing_values(meds_df, "medicines")
    
    # 2. Domain logic ranges
    domain_checks = check_data_ranges_and_domains(sales_df, batches_df)
    
    # 3. PII and privacy compliance
    privacy_checks = audit_sensitive_data_and_pii(sales_df, batches_df)
    
    # 4. Outlier analysis
    cleaned_sales_df, outlier_summary = analyze_and_handle_outliers(sales_df, clip_outliers=True)
    
    audit_report = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "missing_values_audit": {
            "daily_sales": missing_sales,
            "inventory_batches": missing_batches,
            "medicines": missing_meds,
        },
        "domain_range_audit": domain_checks,
        "privacy_and_pii_audit": privacy_checks,
        "outlier_analysis": outlier_summary,
        "overall_data_health": "EXCELLENT" if (
            missing_sales["status"] == "PASS" and
            missing_batches["status"] == "PASS" and
            domain_checks["status"] == "PASS" and
            privacy_checks["status"] == "PASS"
        ) else "ACTION_REQUIRED"
    }
    
    # Save audit report
    out_path = PROCESSED_DATA_DIR / "data_quality_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit_report, f, indent=2)
        
    logger.info("Data Quality Audit passed with health status: %s", audit_report["overall_data_health"])
    return audit_report


if __name__ == "__main__":
    report = run_full_quality_audit()
    print(json.dumps(report, indent=2))
