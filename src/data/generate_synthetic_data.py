"""
Synthetic Data Generator for Pharmacy Medicine Demand & Expiry-Risk Forecasting.
ALGIONS AI LABS - Assessment Project

Generates realistic pharmacy datasets:
1. medicines.csv - Master product catalog with therapeutic categories and pricing.
2. daily_sales.csv - Daily sales transactions with seasonal patterns, day-of-week effects,
   price elasticity, and realistic stockout events across 2 full years (2024-2025).
3. inventory_batches.csv - Batch-level inventory tracking manufacturing dates, expiry dates,
   current stock, supplier return windows, and historical expiry outcomes.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

from src.config import (
    PRODUCT_CATALOG,
    START_DATE,
    END_DATE,
    RANDOM_SEED,
    RAW_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_medicines_df() -> pd.DataFrame:
    """Generates the product master table from the configured catalog."""
    df = pd.DataFrame(PRODUCT_CATALOG)
    # Remove synthetic generation helper keys from master export table
    cols_to_keep = [
        "product_id", "name", "category", "form", "unit_cost", "unit_price",
        "is_prescription_required", "storage_condition", "shelf_life_months"
    ]
    return df[cols_to_keep].copy()


def calculate_seasonality_factor(date: pd.Timestamp, prod: dict) -> float:
    """
    Computes a realistic seasonal demand multiplier based on day of year,
    winter respiratory surges, and monsoon gastro/infectious surges.
    """
    month = date.month
    day_of_year = date.dayofyear
    
    # Base smooth annual sinusoidal variation
    annual_cycle = 1.0 + 0.10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    
    winter_multiplier = prod.get("seasonality_winter_boost", 1.0)
    monsoon_multiplier = prod.get("seasonality_monsoon_boost", 1.0)
    
    # Winter surge (December, January, February)
    if month in [12, 1, 2]:
        winter_weight = 1.0 if month == 1 else 0.75
        seasonal_mult = 1.0 + (winter_multiplier - 1.0) * winter_weight
    # Monsoon surge (July, August, September)
    elif month in [7, 8, 9]:
        monsoon_weight = 1.0 if month == 8 else 0.8
        seasonal_mult = 1.0 + (monsoon_multiplier - 1.0) * monsoon_weight
    # Spring allergies (March, April)
    elif month in [3, 4] and prod["category"] == "Respiratory":
        seasonal_mult = 1.25
    else:
        seasonal_mult = annual_cycle
        
    return float(max(0.6, seasonal_mult))


def calculate_day_of_week_factor(date: pd.Timestamp, prod: dict) -> float:
    """
    Computes day-of-week multiplier.
    - OTC and acute analgesics spike on weekends (Saturday, Sunday)
    - Chronic medicines have higher refills on weekdays (especially Monday clinics)
    """
    dow = date.dayofweek  # 0=Monday, 6=Sunday
    is_otc = not prod["is_prescription_required"]
    category = prod["category"]
    
    if is_otc:
        # Weekend spike for OTC
        dow_multipliers = {0: 1.10, 1: 0.95, 2: 0.95, 3: 0.95, 4: 1.05, 5: 1.25, 6: 1.20}
    elif category == "Chronic_Care":
        # Weekday clinics & refills
        dow_multipliers = {0: 1.20, 1: 1.10, 2: 1.05, 3: 1.00, 4: 1.00, 5: 0.85, 6: 0.80}
    else:
        # Acute prescription
        dow_multipliers = {0: 1.15, 1: 1.05, 2: 1.00, 3: 1.00, 4: 1.05, 5: 1.10, 6: 0.95}
        
    return dow_multipliers.get(dow, 1.0)


def generate_daily_sales(
    medicines_df: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates synthetic daily sales records with realistic distributions,
    seasonality, promotions, and realistic inventory stockout events.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []
    
    prod_dict = {p["product_id"]: p for p in PRODUCT_CATALOG}
    
    logger.info("Generating daily sales transactions for %d products over %d days...", len(prod_dict), len(date_range))
    
    for prod_id, prod in prod_dict.items():
        base_demand = prod["base_daily_demand"]
        std_demand = prod["demand_std"]
        unit_price = prod["unit_price"]
        unit_cost = prod["unit_cost"]
        
        # State tracking for stockout simulation
        current_inventory = int(base_demand * np.random.uniform(25, 45))
        
        for dt in date_range:
            season_mult = calculate_seasonality_factor(dt, prod)
            dow_mult = calculate_day_of_week_factor(dt, prod)
            
            # Promotion discount randomly occurring ~5% of days
            is_promo = np.random.rand() < 0.05
            discount_pct = 0.15 if is_promo else 0.0
            price_elasticity = -1.2
            promo_mult = (1.0 - discount_pct) ** price_elasticity if is_promo else 1.0
            
            # Latent unconstrained customer demand
            mean_demand = base_demand * season_mult * dow_mult * promo_mult
            # Ensure mean is positive
            mean_demand = max(1.0, mean_demand)
            
            # Generate sales from negative binomial / poisson
            dispersion = max(1.0, (std_demand ** 2) / mean_demand)
            if dispersion > 1.0:
                p_param = 1.0 / dispersion
                n_param = mean_demand * p_param / (1.0 - p_param)
                latent_demand = int(np.random.negative_binomial(max(1.0, n_param), max(0.01, min(0.99, p_param))))
            else:
                latent_demand = int(np.random.poisson(mean_demand))
                
            # Random restocking event when inventory falls low
            if current_inventory < base_demand * 5:
                # Restock batch arrived
                restock_qty = int(base_demand * np.random.uniform(20, 40))
                current_inventory += restock_qty
                
            # Simulate physical sales constrained by inventory on hand
            if current_inventory <= 0:
                is_stockout = True
                units_sold = 0
            elif current_inventory < latent_demand:
                is_stockout = True
                units_sold = current_inventory
                current_inventory = 0
            else:
                is_stockout = False
                units_sold = latent_demand
                current_inventory -= units_sold
                
            effective_price = round(unit_price * (1.0 - discount_pct), 2)
            revenue = round(units_sold * effective_price, 2)
            
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": prod_id,
                "product_name": prod["name"],
                "category": prod["category"],
                "units_sold": units_sold,
                "latent_demand": latent_demand,
                "effective_price": effective_price,
                "unit_cost": unit_cost,
                "revenue": revenue,
                "discount_pct": discount_pct,
                "is_promotion": int(is_promo),
                "is_weekend": int(dt.dayofweek in [5, 6]),
                "is_stockout": int(is_stockout),
                "inventory_on_hand": max(0, current_inventory),
            })
            
    df_sales = pd.DataFrame(records)
    logger.info("Generated %d daily sales records.", len(df_sales))
    return df_sales


def generate_inventory_batches(
    medicines_df: pd.DataFrame,
    daily_sales_df: pd.DataFrame,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates realistic pharmaceutical batch inventory data.
    - Historical batches: Most sold out cleanly; ~15% expired with unsold stock (for backtesting).
    - Current Active inventory (as of 2025-12-31): Each medicine has 2 to 4 active batches on hand,
      with a realistic spread of Near-Expiry (High Risk), Warning/Return Window (Medium Risk),
      and Fresh Stock (Low Risk).
    """
    np.random.seed(seed)
    batches = []
    batch_counter = 1000
    
    suppliers = [
        {"id": "SUPP_SUN_PHARMA", "name": "Sun Pharma Distributors", "return_window": 90},
        {"id": "SUPP_CIPLA", "name": "Cipla Logistics Ltd", "return_window": 60},
        {"id": "SUPP_DR_REDDY", "name": "Dr. Reddy's Supply Hub", "return_window": 75},
        {"id": "SUPP_LUPIN", "name": "Lupin Healthcare Distribution", "return_window": 60},
        {"id": "SUPP_ABBOTT", "name": "Abbott India Supply Chain", "return_window": 90},
    ]
    
    timeline_end = pd.to_datetime(END_DATE)  # 2025-12-31
    
    for _, med in medicines_df.iterrows():
        prod_id = med["product_id"]
        shelf_life_months = int(med["shelf_life_months"])
        unit_cost = float(med["unit_cost"])
        unit_price = float(med["unit_price"])
        storage_condition = med["storage_condition"]
        
        med_sales = daily_sales_df[daily_sales_df["product_id"] == prod_id]
        avg_daily_demand = float(med_sales["units_sold"].mean())
        
        # 1. Historical batches from 2022 to mid-2025 (completed / resolved)
        num_historical_batches = np.random.randint(8, 12)
        for h_idx in range(num_historical_batches):
            batch_counter += 1
            batch_id = f"BAT-HIST-{batch_counter:05d}"
            
            # Historical arrival between 300 and 1200 days before timeline_end
            days_ago = np.random.randint(350, 1250)
            mfg_date = timeline_end - timedelta(days=days_ago)
            exp_days = int(shelf_life_months * 30.4)
            expiry_date = mfg_date + timedelta(days=exp_days)
            
            supp = suppliers[np.random.randint(0, len(suppliers))]
            initial_qty = int(avg_daily_demand * np.random.uniform(30, 65))
            
            if expiry_date < timeline_end:
                status = "EXPIRED"
                # Realistic pharmacy waste rate: ~25-30% of expired batches had some unsold waste
                has_waste = np.random.rand() < 0.28
                pct_unsold = float(np.random.uniform(0.06, 0.35)) if has_waste else 0.0
                expired_units = int(initial_qty * pct_unsold)
            else:
                status = "SOLD_OUT"
                pct_unsold = 0.0
                expired_units = 0
                
            batches.append({
                "batch_id": batch_id,
                "product_id": prod_id,
                "product_name": med["name"],
                "category": med["category"],
                "mfg_date": mfg_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "initial_quantity": initial_qty,
                "current_stock": 0,
                "unit_cost": unit_cost,
                "unit_price": unit_price,
                "supplier_id": supp["id"],
                "supplier_name": supp["name"],
                "supplier_return_window_days": supp["return_window"],
                "storage_condition": storage_condition,
                "status": status,
                "historical_expired_units": expired_units,
                "as_of_date": END_DATE,
            })
            
        # 2. Currently ACTIVE inventory batches on shelf at 2025-12-31 (2 to 4 active batches per product)
        num_active_batches = np.random.randint(2, 5)
        # Create a realistic spectrum of expiry horizons for active batches:
        # Batch 1 (oldest active): might be near expiry (15-45 days) for some products
        # Batch 2: medium horizon (45-120 days)
        # Batch 3-4: fresh horizon (120-400 days)
        
        for a_idx in range(num_active_batches):
            batch_counter += 1
            batch_id = f"BAT-ACT-{batch_counter:05d}"
            supp = suppliers[np.random.randint(0, len(suppliers))]
            
            if a_idx == 0:
                # Oldest active batch: ~20% of products have an imminent near-expiry batch
                is_near_expiry = np.random.rand() < 0.35
                if is_near_expiry:
                    days_to_exp = int(np.random.randint(12, 35))
                    stock_qty = int(avg_daily_demand * np.random.uniform(20, 50))  # surplus relative to days
                else:
                    days_to_exp = int(np.random.randint(45, 80))
                    stock_qty = int(avg_daily_demand * np.random.uniform(15, 40))
            elif a_idx == 1:
                days_to_exp = int(np.random.randint(60, 150))
                stock_qty = int(avg_daily_demand * np.random.uniform(25, 55))
            else:
                days_to_exp = int(np.random.randint(150, 450))
                stock_qty = int(avg_daily_demand * np.random.uniform(30, 70))
                
            expiry_date = timeline_end + timedelta(days=days_to_exp)
            mfg_date = expiry_date - pd.DateOffset(months=shelf_life_months)
            initial_qty = int(stock_qty * np.random.uniform(1.2, 2.5))
            
            batches.append({
                "batch_id": batch_id,
                "product_id": prod_id,
                "product_name": med["name"],
                "category": med["category"],
                "mfg_date": mfg_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "initial_quantity": initial_qty,
                "current_stock": stock_qty,
                "unit_cost": unit_cost,
                "unit_price": unit_price,
                "supplier_id": supp["id"],
                "supplier_name": supp["name"],
                "supplier_return_window_days": supp["return_window"],
                "storage_condition": storage_condition,
                "status": "ACTIVE",
                "historical_expired_units": 0,
                "as_of_date": END_DATE,
            })
            
    df_batches = pd.DataFrame(batches)
    logger.info("Generated %d inventory batch records (%d currently ACTIVE).",
                len(df_batches), (df_batches['status'] == 'ACTIVE').sum())
    return df_batches


def run_data_generation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes full data generation and saves raw CSV files.
    """
    logger.info("Starting synthetic dataset generation...")
    meds_df = generate_medicines_df()
    sales_df = generate_daily_sales(meds_df)
    batches_df = generate_inventory_batches(meds_df, sales_df)
    
    # Save to disk
    meds_path = RAW_DATA_DIR / "medicines.csv"
    sales_path = RAW_DATA_DIR / "daily_sales.csv"
    batches_path = RAW_DATA_DIR / "inventory_batches.csv"
    
    meds_df.to_csv(meds_path, index=False)
    sales_df.to_csv(sales_path, index=False)
    batches_df.to_csv(batches_path, index=False)
    
    logger.info("Saved raw datasets to:")
    logger.info(" - %s (%d records)", meds_path, len(meds_df))
    logger.info(" - %s (%d records)", sales_path, len(sales_df))
    logger.info(" - %s (%d records)", batches_path, len(batches_df))
    
    return meds_df, sales_df, batches_df


if __name__ == "__main__":
    run_data_generation()
