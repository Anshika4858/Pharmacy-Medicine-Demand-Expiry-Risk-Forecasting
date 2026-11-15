"""
Unit Tests for FastAPI REST Endpoints and Real Model Inferences.
ALGIONS AI LABS - Assessment Project
"""
import unittest
from src.api.main import (
    health_check,
    list_products,
    predict_product_demand,
    predict_batch_expiry_risk,
    DemandPredictionRequest,
    BatchRiskRequest,
)


class TestAPIEndpoints(unittest.TestCase):
    
    def test_health_endpoint(self):
        """Validates /health endpoint outputs status, champion model, and metrics."""
        res = health_check()
        self.assertEqual(res["status"], "HEALTHY")
        self.assertIn("champion_model", res)
        self.assertIn("regulatory_disclaimer", res)

    def test_products_endpoint(self):
        """Validates /products returns product catalog."""
        res = list_products()
        self.assertEqual(res["total_products"], 25)
        self.assertIn("disclaimer", res)

    def test_predict_demand_real_model_inference(self):
        """Submits real sample input to demand forecast and verifies real model outputs."""
        payload = DemandPredictionRequest(
            product_id="MED001",
            horizon_days=30,
            as_of_date="2025-12-31"
        )
        res = predict_product_demand(payload)
        
        self.assertEqual(res.product_id, "MED001")
        self.assertEqual(len(res.daily_forecasts), 30)
        self.assertGreater(res.total_projected_demand, 0.0)
        self.assertIn("DECISION SUPPORT", res.disclaimer)

    def test_predict_batch_expiry_risk_real_inference(self):
        """Submits sample batch to expiry risk engine and verifies classification."""
        payload = BatchRiskRequest(
            batch_id="TEST_B001",
            product_id="MED001",
            current_stock=2000,
            expiry_date="2026-01-20",
            unit_cost=4.5,
            unit_price=9.0,
            supplier_name="Sun Pharma Distributors",
            supplier_return_window_days=90,
            as_of_date="2025-12-31"
        )
        res = predict_batch_expiry_risk(payload)
        
        self.assertEqual(res.batch_id, "TEST_B001")
        self.assertIn(res.risk_level, ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"])
        self.assertGreater(res.capital_at_risk_cost, 0.0)
        self.assertIsNotNone(res.recommended_operational_action)


if __name__ == "__main__":
    unittest.main()

