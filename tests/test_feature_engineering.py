"""
Unit Tests for Feature Engineering and Leakage Prevention.
ALGIONS AI LABS - Assessment Project
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, TRAIN_END_DATE, VAL_END_DATE, TEST_START_DATE
from src.data.quality_checks import verify_temporal_leakage_safety


class TestFeatureEngineering(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_features.csv")
        cls.val_df = pd.read_csv(PROCESSED_DATA_DIR / "val_features.csv")
        cls.test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_features.csv")

    def test_chronological_split_integrity(self):
        """Verifies strictly that no future records leak into earlier partitions."""
        audit = verify_temporal_leakage_safety(
            self.train_df["date"], self.val_df["date"], self.test_df["date"]
        )
        self.assertTrue(audit["overall_leakage_safe"])
        self.assertEqual(audit["status"], "PASS")
        
        # Max train date <= TRAIN_END_DATE
        self.assertLessEqual(self.train_df["date"].max(), TRAIN_END_DATE)
        self.assertGreater(self.val_df["date"].min(), TRAIN_END_DATE)
        self.assertLessEqual(self.val_df["date"].max(), VAL_END_DATE)
        self.assertGreaterEqual(self.test_df["date"].min(), TEST_START_DATE)

    def test_feature_matrix_cleanliness(self):
        """Verifies that processed training matrices contain zero NaN values."""
        self.assertFalse(self.train_df.isnull().any().any())
        self.assertFalse(self.val_df.isnull().any().any())
        self.assertFalse(self.test_df.isnull().any().any())

    def test_lag_leakage_prevention(self):
        """
        Validates that lag_1 at time t equals units_sold at time t-1.
        Confirms target is strictly shifted and not synchronously accessible.
        """
        med1_train = self.train_df[self.train_df["product_id"] == "MED001"].sort_values("date").reset_index(drop=True)
        # Check first 10 consecutive days
        for i in range(1, 10):
            self.assertEqual(
                med1_train.loc[i, "lag_1"],
                med1_train.loc[i - 1, "units_sold"],
                "lag_1 at row i must match units_sold at row i-1"
            )


if __name__ == "__main__":
    unittest.main()

