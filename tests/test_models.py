"""
Unit Tests for Model Artifacts and Forecasting Accuracy.
ALGIONS AI LABS - Assessment Project
"""
import unittest
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import MODEL_DIR, PROCESSED_DATA_DIR
from src.models.train import calculate_metrics, MovingAverageBaseline


class TestModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = MODEL_DIR / "best_model.joblib"
        cls.metadata_path = MODEL_DIR / "model_metadata.json"
        cls.eval_path = MODEL_DIR / "evaluation_metrics.json"
        
        cls.model = joblib.load(cls.model_path)
        with open(cls.metadata_path, "r") as f:
            cls.metadata = json.load(f)
        with open(cls.eval_path, "r") as f:
            cls.eval_metrics = json.load(f)
            
        cls.test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_features.csv")
        with open(PROCESSED_DATA_DIR / "feature_names.json", "r") as f:
            cls.feature_cols = json.load(f)

    def test_champion_model_artifact_exists_and_loads(self):
        """Verifies champion model artifact is serialized and loadable."""
        self.assertTrue(self.model_path.exists())
        self.assertIsNotNone(self.model)
        self.assertIn("champion_model_name", self.metadata)

    def test_champion_model_inference_shape_and_range(self):
        """Verifies champion model returns valid non-negative predictions."""
        X_sample = self.test_df[self.feature_cols].head(20)
        preds = self.model.predict(X_sample)
        
        self.assertEqual(len(preds), 20)
        self.assertFalse(np.isnan(preds).any())
        self.assertFalse(np.isinf(preds).any())

    def test_metrics_calculation_logic(self):
        """Validates MAE, RMSE, and WAPE calculation functions."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 38.0])
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics["MAE"], 2.25, places=2)
        self.assertAlmostEqual(metrics["WAPE"], 0.09, places=2)
        self.assertEqual(metrics["WAPE_pct"], 9.0)

    def test_held_out_test_outperforms_moving_average_baseline(self):
        """Confirms that ML model is evaluated against baselines on test set."""
        test_comp = self.eval_metrics["test_comparison"]
        self.assertIn("Baseline_7D_Moving_Average", test_comp)
        self.assertIn(self.metadata["champion_model_name"], test_comp)


if __name__ == "__main__":
    unittest.main()

