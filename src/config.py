"""
Configuration and constants for Pharmacy Medicine Demand & Expiry-Risk Forecasting.
ALGIONS AI LABS - Assessment Project
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Create directories if they don't exist
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Time Horizon & Temporal Split Configuration
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"  # 731 days (2 full years, 2024 is leap year)
TRAIN_END_DATE = "2025-06-30"  # Train: 2024-01-01 to 2025-06-30 (547 days, ~75%)
VAL_END_DATE = "2025-09-30"    # Val:   2025-07-01 to 2025-09-30 (92 days, ~12.5%)
TEST_START_DATE = "2025-10-01" # Test:  2025-10-01 to 2025-12-31 (92 days, ~12.5%)

# Forecasting Parameters
FORECAST_HORIZON_DAYS = 30
SHORT_HORIZON_DAYS = 14

# Lag & Rolling Window Features
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 30]
ROLLING_WINDOWS = [7, 14, 28]

# Expiry Risk Thresholds
EXPIRY_CRITICAL_DAYS = 30  # Less than 30 days to expiry
EXPIRY_WARNING_DAYS = 60   # 30-60 days to expiry
SUPPLIER_RETURN_WINDOW_DAYS = 75  # Return window for manufacturer credit

# Risk Classification Categories
RISK_LEVEL_HIGH = "HIGH_RISK"
RISK_LEVEL_MEDIUM = "MEDIUM_RISK"
RISK_LEVEL_LOW = "LOW_RISK"

# Advisory Disclaimer
ADVISORY_DISCLAIMER = (
    "OPERATIONAL DECISION SUPPORT ONLY: This solution provides inventory demand "
    "projections and stock rotation guidance for pharmacy management. It does NOT "
    "provide clinical, medical, dosage, or therapeutic treatment advice. All pharmaceutical "
    "dispensing and stock disposal decisions must adhere to licensed pharmacist oversight "
    "and regulatory compliance standards (CDSCO / FDA / Good Pharmacy Practice)."
)

# Product Catalog Definition (25 representative medicines across 5 therapeutic classes)
PRODUCT_CATALOG = [
    # Category 1: Antibiotics & Anti-Infectives
    {
        "product_id": "MED001",
        "name": "Amoxicillin 500mg",
        "category": "Antibiotics",
        "form": "Capsule",
        "base_daily_demand": 45,
        "demand_std": 12,
        "unit_cost": 4.50,
        "unit_price": 9.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.40,
        "seasonality_monsoon_boost": 1.25,
    },
    {
        "product_id": "MED002",
        "name": "Azithromycin 500mg",
        "category": "Antibiotics",
        "form": "Tablet",
        "base_daily_demand": 30,
        "demand_std": 9,
        "unit_cost": 12.00,
        "unit_price": 22.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.60,
        "seasonality_monsoon_boost": 1.30,
    },
    {
        "product_id": "MED003",
        "name": "Ciprofloxacin 500mg",
        "category": "Antibiotics",
        "form": "Tablet",
        "base_daily_demand": 22,
        "demand_std": 6,
        "unit_cost": 6.00,
        "unit_price": 12.50,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.10,
        "seasonality_monsoon_boost": 1.45,
    },
    {
        "product_id": "MED004",
        "name": "Doxycycline 100mg",
        "category": "Antibiotics",
        "form": "Capsule",
        "base_daily_demand": 18,
        "demand_std": 5,
        "unit_cost": 5.00,
        "unit_price": 11.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.15,
        "seasonality_monsoon_boost": 1.50,
    },
    {
        "product_id": "MED005",
        "name": "Cefixime 200mg",
        "category": "Antibiotics",
        "form": "Tablet",
        "base_daily_demand": 26,
        "demand_std": 7,
        "unit_cost": 15.00,
        "unit_price": 28.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.35,
        "seasonality_monsoon_boost": 1.40,
    },

    # Category 2: Analgesics & Antipyretics
    {
        "product_id": "MED006",
        "name": "Paracetamol 650mg",
        "category": "Analgesics",
        "form": "Tablet",
        "base_daily_demand": 110,
        "demand_std": 25,
        "unit_cost": 1.20,
        "unit_price": 3.00,
        "is_prescription_required": False,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.50,
        "seasonality_monsoon_boost": 1.40,
    },
    {
        "product_id": "MED007",
        "name": "Ibuprofen 400mg",
        "category": "Analgesics",
        "form": "Tablet",
        "base_daily_demand": 42,
        "demand_std": 10,
        "unit_cost": 2.50,
        "unit_price": 6.00,
        "is_prescription_required": False,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.20,
        "seasonality_monsoon_boost": 1.10,
    },
    {
        "product_id": "MED008",
        "name": "Aceclofenac 100mg",
        "category": "Analgesics",
        "form": "Tablet",
        "base_daily_demand": 35,
        "demand_std": 8,
        "unit_cost": 4.00,
        "unit_price": 8.50,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.15,
        "seasonality_monsoon_boost": 1.05,
    },
    {
        "product_id": "MED009",
        "name": "Tramadol 50mg",
        "category": "Analgesics",
        "form": "Capsule",
        "base_daily_demand": 12,
        "demand_std": 3,
        "unit_cost": 8.00,
        "unit_price": 18.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.05,
        "seasonality_monsoon_boost": 1.00,
    },
    {
        "product_id": "MED010",
        "name": "Diclofenac Sodium 50mg",
        "category": "Analgesics",
        "form": "Tablet",
        "base_daily_demand": 28,
        "demand_std": 6,
        "unit_cost": 2.00,
        "unit_price": 5.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.10,
        "seasonality_monsoon_boost": 1.05,
    },

    # Category 3: Antidiabetics & Cardiovascular (Chronic, steady demand)
    {
        "product_id": "MED011",
        "name": "Metformin 500mg",
        "category": "Chronic_Care",
        "form": "Tablet",
        "base_daily_demand": 85,
        "demand_std": 14,
        "unit_cost": 1.80,
        "unit_price": 4.50,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.02,
        "seasonality_monsoon_boost": 1.01,
    },
    {
        "product_id": "MED012",
        "name": "Glimepiride 2mg",
        "category": "Chronic_Care",
        "form": "Tablet",
        "base_daily_demand": 40,
        "demand_std": 7,
        "unit_cost": 3.20,
        "unit_price": 7.50,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.01,
        "seasonality_monsoon_boost": 1.00,
    },
    {
        "product_id": "MED013",
        "name": "Atorvastatin 20mg",
        "category": "Chronic_Care",
        "form": "Tablet",
        "base_daily_demand": 60,
        "demand_std": 10,
        "unit_cost": 5.50,
        "unit_price": 13.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.02,
        "seasonality_monsoon_boost": 1.01,
    },
    {
        "product_id": "MED014",
        "name": "Amlodipine 5mg",
        "category": "Chronic_Care",
        "form": "Tablet",
        "base_daily_demand": 55,
        "demand_std": 9,
        "unit_cost": 1.50,
        "unit_price": 4.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.03,
        "seasonality_monsoon_boost": 1.00,
    },
    {
        "product_id": "MED015",
        "name": "Telmisartan 40mg",
        "category": "Chronic_Care",
        "form": "Tablet",
        "base_daily_demand": 50,
        "demand_std": 8,
        "unit_cost": 4.20,
        "unit_price": 9.50,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.02,
        "seasonality_monsoon_boost": 1.01,
    },

    # Category 4: Respiratory & Antihistamines
    {
        "product_id": "MED016",
        "name": "Cetirizine 10mg",
        "category": "Respiratory",
        "form": "Tablet",
        "base_daily_demand": 52,
        "demand_std": 13,
        "unit_cost": 1.00,
        "unit_price": 2.80,
        "is_prescription_required": False,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.35,
        "seasonality_monsoon_boost": 1.40,
    },
    {
        "product_id": "MED017",
        "name": "Montelukast 10mg",
        "category": "Respiratory",
        "form": "Tablet",
        "base_daily_demand": 38,
        "demand_std": 8,
        "unit_cost": 7.00,
        "unit_price": 15.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.45,
        "seasonality_monsoon_boost": 1.30,
    },
    {
        "product_id": "MED018",
        "name": "Levocetirizine 5mg",
        "category": "Respiratory",
        "form": "Tablet",
        "base_daily_demand": 34,
        "demand_std": 7,
        "unit_cost": 2.20,
        "unit_price": 5.50,
        "is_prescription_required": False,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.40,
        "seasonality_monsoon_boost": 1.35,
    },
    {
        "product_id": "MED019",
        "name": "Salbutamol Inhaler 100mcg",
        "category": "Respiratory",
        "form": "Inhaler",
        "base_daily_demand": 15,
        "demand_std": 4,
        "unit_cost": 45.00,
        "unit_price": 85.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.65,
        "seasonality_monsoon_boost": 1.25,
    },
    {
        "product_id": "MED020",
        "name": "Dextromethorphan Syrup 100ml",
        "category": "Respiratory",
        "form": "Syrup",
        "base_daily_demand": 24,
        "demand_std": 8,
        "unit_cost": 28.00,
        "unit_price": 55.00,
        "is_prescription_required": False,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.85,
        "seasonality_monsoon_boost": 1.20,
    },

    # Category 5: Gastrointestinal & Special Storage
    {
        "product_id": "MED021",
        "name": "Pantoprazole 40mg",
        "category": "Gastrointestinal",
        "form": "Tablet",
        "base_daily_demand": 72,
        "demand_std": 15,
        "unit_cost": 3.80,
        "unit_price": 8.00,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.05,
        "seasonality_monsoon_boost": 1.15,
    },
    {
        "product_id": "MED022",
        "name": "Omeprazole 20mg",
        "category": "Gastrointestinal",
        "form": "Capsule",
        "base_daily_demand": 48,
        "demand_std": 10,
        "unit_cost": 2.20,
        "unit_price": 5.20,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 1.05,
        "seasonality_monsoon_boost": 1.10,
    },
    {
        "product_id": "MED023",
        "name": "Ondansetron 4mg",
        "category": "Gastrointestinal",
        "form": "Tablet",
        "base_daily_demand": 20,
        "demand_std": 5,
        "unit_cost": 4.00,
        "unit_price": 9.50,
        "is_prescription_required": True,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 24,
        "seasonality_winter_boost": 1.08,
        "seasonality_monsoon_boost": 1.35,
    },
    {
        "product_id": "MED024",
        "name": "ORS Electrolyte Powder 21.8g",
        "category": "Gastrointestinal",
        "form": "Sachet",
        "base_daily_demand": 65,
        "demand_std": 20,
        "unit_cost": 6.00,
        "unit_price": 12.00,
        "is_prescription_required": False,
        "storage_condition": "Room Temperature",
        "shelf_life_months": 36,
        "seasonality_winter_boost": 0.80,
        "seasonality_monsoon_boost": 1.95,  # Strong monsoon spike for gastro infections
    },
    {
        "product_id": "MED025",
        "name": "Insulin Glargine 100 IU/ml",
        "category": "Chronic_Care",
        "form": "Vial",
        "base_daily_demand": 14,
        "demand_std": 3,
        "unit_cost": 210.00,
        "unit_price": 380.00,
        "is_prescription_required": True,
        "storage_condition": "Cold Chain (2-8°C)",
        "shelf_life_months": 18,
        "seasonality_winter_boost": 1.00,
        "seasonality_monsoon_boost": 1.00,
    },
]

