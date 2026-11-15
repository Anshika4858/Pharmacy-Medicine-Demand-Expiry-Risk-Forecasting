"""
End-to-End Master Pipeline Runner for Pharmacy Medicine Demand & Expiry-Risk Forecasting.
ALGIONS AI LABS - Assessment Project

Single-command reproducible pipeline:
1. Generates synthetic master catalog, sales transactions, and batch inventory.
2. Performs automated data quality checks, outlier handling, and leakage verification.
3. Builds leakage-safe time-lagged and rolling feature matrices with chronological splitting.
4. Trains, tunes, and compares Baselines vs Ridge vs Random Forest vs XGBoost models.
5. Saves the champion model artifact and metadata.
6. Executes FEFO expiry-risk simulation and historical backtesting.
"""
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineRunner")


def main():
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("PHARMACY MEDICINE DEMAND & EXPIRY-RISK FORECASTING PIPELINE")
    logger.info("ALGIONS AI LABS - Individual Project Assessment")
    logger.info("================================================================================")
    
    # Step 1: Synthetic Data Generation
    logger.info("\n--- STEP 1: SYNTHETIC DATA GENERATION ---")
    from src.data.generate_synthetic_data import run_data_generation
    meds_df, sales_df, batches_df = run_data_generation()
    logger.info("Generated %d medicines, %d sales transactions, %d inventory batches.",
                len(meds_df), len(sales_df), len(batches_df))
    
    # Step 2: Data Quality & Leakage Audit
    logger.info("\n--- STEP 2: DATA INTEGRITY & LEAKAGE AUDIT ---")
    from src.data.quality_checks import run_full_quality_audit
    quality_report = run_full_quality_audit()
    logger.info("Data Health Status: %s", quality_report["overall_data_health"])
    
    # Step 3: Feature Engineering & Chronological Split
    logger.info("\n--- STEP 3: LEAKAGE-SAFE FEATURE ENGINEERING & SPLIT ---")
    from src.features.feature_engineering import partition_and_save_datasets
    train_df, val_df, test_df, feature_cols = partition_and_save_datasets()
    logger.info("Constructed %d model features across Train (%d), Val (%d), Test (%d) splits.",
                len(feature_cols), len(train_df), len(val_df), len(test_df))
    
    # Step 4: Model Training & Evaluation
    logger.info("\n--- STEP 4: MODEL TRAINING & CROSS-MODEL BENCHMARKING ---")
    from src.models.train import train_and_evaluate_all_models
    meta = train_and_evaluate_all_models()
    logger.info("Champion Model Selected: %s", meta["champion_model_name"])
    
    # Print Test Leaderboard
    print("\n" + "="*80)
    print("DEMAND FORECASTING EVALUATION LEADERBOARD (HELD-OUT TEST SET)")
    print("="*80)
    print(f"{'Model Name':<32} | {'MAE':<7} | {'RMSE':<7} | {'WAPE':<8} | {'R2 Score':<8}")
    print("-"*80)
    for model_name, res in meta["held_out_test_metrics"].items():
        print(f"{model_name:<32} | {res['MAE']:<7.2f} | {res['RMSE']:<7.2f} | {res['WAPE_pct']:<7.2f}% | {res['R2']:<8.3f}")
    print("="*80)
    
    # Step 5: Expiry Risk Engine & Historical Backtest
    logger.info("\n--- STEP 5: FEFO EXPIRY-RISK SIMULATION & BACKTESTING ---")
    from src.models.expiry_risk_engine import run_full_expiry_risk_pipeline
    df_assessed, classification_metrics = run_full_expiry_risk_pipeline()
    
    high_risk = (df_assessed["risk_level"] == "HIGH_RISK").sum()
    med_risk = (df_assessed["risk_level"] == "MEDIUM_RISK").sum()
    low_risk = (df_assessed["risk_level"] == "LOW_RISK").sum()
    cap_at_risk = df_assessed["capital_at_risk_cost"].sum()
    
    cm = classification_metrics["confusion_matrix"]
    m = classification_metrics["metrics"]
    
    print("\n" + "="*80)
    print("EXPIRY-RISK CLASSIFICATION BACKTESTING EVALUATION")
    print("="*80)
    print(f"Precision:   {m['Precision']*100:.2f}% | Recall (Sensitivity): {m['Recall_Sensitivity']*100:.2f}%")
    print(f"F1-Score:    {m['F1_Score']*100:.2f}% | Specificity:          {m['Specificity']*100:.2f}%")
    print(f"Confusion Matrix: [TP: {cm['TP']}, FP: {cm['FP']}, FN: {cm['FN']}, TN: {cm['TN']}]")
    print("-"*80)
    print(f"Active Batches Monitored: {len(df_assessed)} | High Risk: {high_risk} | Medium Risk: {med_risk} | Safe: {low_risk}")
    print(f"Total Capital at Risk: ${cap_at_risk:,.2f}")
    print("="*80)
    
    elapsed = time.time() - start_time
    logger.info("\nPipeline completed successfully in %.2f seconds.", elapsed)


if __name__ == "__main__":
    main()

