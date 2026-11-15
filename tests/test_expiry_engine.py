"""
Unit Tests for Expiry-Risk Decision Engine and FEFO Logic.
ALGIONS AI LABS - Assessment Project
"""
import unittest
import json
import pandas as pd
from pathlib import Path

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RISK_LEVEL_HIGH, RISK_LEVEL_LOW
from src.models.expiry_risk_engine import simulate_fefo_expiry_risk, evaluate_historical_expiry_risk_classification


class TestExpiryEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.active_report_path = PROCESSED_DATA_DIR / "active_batches_risk_report.csv"
        cls.batches_df = pd.read_csv(RAW_DATA_DIR / "inventory_batches.csv")
        with open(MODEL_DIR / "expiry_risk_classification_metrics.json", "r") as f:
            cls.class_metrics = json.load(f)

    def test_active_batches_risk_report_exists(self):
        """Verifies active batches report exists and has classified batches."""
        self.assertTrue(self.active_report_path.exists())
        df = pd.read_csv(self.active_report_path)
        self.assertGreater(len(df), 0)
        self.assertIn("risk_level", df.columns)
        self.assertIn("capital_at_risk_cost", df.columns)
        self.assertIn("disclaimer", df.columns)

    def test_fefo_simulation_critical_expiry_triggers_high_risk(self):
        """
        Synthesizes a near-expiry batch with excessive stock
        and verifies the FEFO simulation tags it as HIGH_RISK.
        """
        batches = pd.DataFrame([{
            "batch_id": "TEST_BAT_CRITICAL",
            "product_id": "MED001",
            "product_name": "Amoxicillin 500mg",
            "category": "Antibiotics",
            "current_stock": 1000,
            "unit_cost": 4.5,
            "unit_price": 9.0,
            "expiry_date": "2026-01-15",
            "mfg_date": "2024-01-15",
            "supplier_id": "SUPP_SUN_PHARMA",
            "supplier_name": "Sun Pharma Distributors",
            "supplier_return_window_days": 90,
            "status": "ACTIVE"
        }])
        
        # Low daily demand forecast (e.g. 10 units/day for 15 days = 150 units total)
        dates = pd.date_range("2026-01-01", "2026-01-15", freq="D")
        forecast_df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "forecasted_demand": [10.0] * len(dates)
        })
        
        results = simulate_fefo_expiry_risk(batches, forecast_df, as_of_date="2025-12-31")
        self.assertEqual(len(results), 1)
        res = results[0]
        
        self.assertEqual(res["risk_level"], RISK_LEVEL_HIGH)
        self.assertGreater(res["expected_unsold_units"], 500)
        self.assertEqual(res["action_code"], "FEFO_CLEARANCE_DISCOUNT")

    def test_classification_metrics_completeness(self):
        """Verifies confusion matrix and precision/recall backtesting metrics."""
        cm = self.class_metrics["confusion_matrix"]
        m = self.class_metrics["metrics"]
        
        self.assertIn("TP", cm)
        self.assertIn("FP", cm)
        self.assertIn("FN", cm)
        self.assertIn("TN", cm)
        
        self.assertGreaterEqual(m["Recall_Sensitivity"], 0.70, "Recall must be prioritized in pharmacy safety.")
        self.assertGreaterEqual(m["Specificity"], 0.70)


if __name__ == "__main__":
    unittest.main()

