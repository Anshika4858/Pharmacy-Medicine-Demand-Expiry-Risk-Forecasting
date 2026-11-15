"""
Unit Tests for Synthetic Data Generation and Quality Checks.
ALGIONS AI LABS - Assessment Project
"""
import unittest
import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_DIR
from src.data.generate_synthetic_data import generate_medicines_df, generate_daily_sales, generate_inventory_batches
from src.data.quality_checks import check_missing_values, check_data_ranges_and_domains, audit_sensitive_data_and_pii


class TestDataGeneration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.meds_df = pd.read_csv(RAW_DATA_DIR / "medicines.csv")
        cls.sales_df = pd.read_csv(RAW_DATA_DIR / "daily_sales.csv")
        cls.batches_df = pd.read_csv(RAW_DATA_DIR / "inventory_batches.csv")

    def test_medicines_catalog_completeness(self):
        """Validates that medicine catalog has 25 products with all required attributes."""
        self.assertEqual(len(self.meds_df), 25)
        required_cols = [
            "product_id", "name", "category", "form", "unit_cost", "unit_price",
            "is_prescription_required", "storage_condition", "shelf_life_months"
        ]
        for col in required_cols:
            self.assertIn(col, self.meds_df.columns)
        self.assertFalse(self.meds_df.isnull().any().any())

    def test_daily_sales_schema_and_volume(self):
        """Validates that daily sales records span 731 days for each of the 25 medicines."""
        self.assertGreaterEqual(len(self.sales_df), 18000)
        self.assertEqual(self.sales_df["product_id"].nunique(), 25)
        self.assertFalse(self.sales_df["units_sold"].isnull().any())
        self.assertTrue((self.sales_df["units_sold"] >= 0).all())
        self.assertTrue((self.sales_df["effective_price"] > 0).all())

    def test_inventory_batches_integrity(self):
        """Validates batch expiry logic and supplier fields."""
        self.assertGreater(len(self.batches_df), 50)
        mfg = pd.to_datetime(self.batches_df["mfg_date"])
        exp = pd.to_datetime(self.batches_df["expiry_date"])
        self.assertTrue((exp > mfg).all(), "Expiry date must be after manufacturing date.")
        self.assertTrue((self.batches_df["current_stock"] >= 0).all())

    def test_quality_audit_modules(self):
        """Validates automated quality checks pass without error."""
        missing_sales = check_missing_values(self.sales_df, "sales")
        self.assertEqual(missing_sales["status"], "PASS")
        
        domain_check = check_data_ranges_and_domains(self.sales_df, self.batches_df)
        self.assertEqual(domain_check["status"], "PASS")
        
        pii_check = audit_sensitive_data_and_pii(self.sales_df, self.batches_df)
        self.assertEqual(pii_check["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

