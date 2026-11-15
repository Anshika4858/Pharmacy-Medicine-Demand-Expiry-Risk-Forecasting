"""
FastAPI REST Inference Service for Pharmacy Medicine Demand & Expiry-Risk Forecasting.
ALGIONS AI LABS - Assessment Project

Provides production REST API endpoints:
- GET  /health            : Health check, active model version, and system status
- GET  /products          : Product catalog master data and therapeutic categories
- POST /predict/demand    : Multi-day demand forecast for specified product and horizon
- POST /predict/expiry-risk: FEFO stock depletion simulation and expiry risk scoring
- GET  /inventory/batches : Live active inventory batch status and operational alerts
"""
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
import joblib

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    PRODUCT_CATALOG,
    ADVISORY_DISCLAIMER,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_MEDIUM,
    RISK_LEVEL_LOW,
    EXPIRY_CRITICAL_DAYS,
    EXPIRY_WARNING_DAYS,
)
from src.models.expiry_risk_engine import (
    forecast_product_demand_horizon,
    simulate_fefo_expiry_risk,
    load_champion_model_and_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Pharmacy Medicine Demand & Expiry-Risk Forecasting API",
    description=(
        "Operational decision-support REST service providing multi-step pharmaceutical "
        "demand projections and FEFO batch expiry-risk evaluations.\n\n"
        f"**IMPORTANT REGULATORY NOTICE**: {ADVISORY_DISCLAIMER}"
    ),
    version="1.0.0",
)

# Global cache for model and static data
MODEL_CACHE = {}


def get_cached_resources():
    """Lazy loader and in-memory cache for models and datasets."""
    if "model" not in MODEL_CACHE:
        logger.info("Loading ML champion model and reference datasets into memory...")
        model, feature_cols = load_champion_model_and_features()
        sales_df = pd.read_csv(RAW_DATA_DIR / "daily_sales.csv")
        meds_df = pd.read_csv(RAW_DATA_DIR / "medicines.csv")
        batches_df = pd.read_csv(RAW_DATA_DIR / "inventory_batches.csv")
        
        with open(MODEL_DIR / "model_metadata.json", "r") as f:
            metadata = json.load(f)
            
        with open(MODEL_DIR / "evaluation_metrics.json", "r") as f:
            eval_metrics = json.load(f)
            
        MODEL_CACHE["model"] = model
        MODEL_CACHE["feature_cols"] = feature_cols
        MODEL_CACHE["sales_df"] = sales_df
        MODEL_CACHE["meds_df"] = meds_df
        MODEL_CACHE["batches_df"] = batches_df
        MODEL_CACHE["metadata"] = metadata
        MODEL_CACHE["eval_metrics"] = eval_metrics
        logger.info("Resources loaded successfully.")
        
    return MODEL_CACHE


# ---------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------

class DemandPredictionRequest(BaseModel):
    product_id: str = Field(..., example="MED001", description="Valid Product SKU ID (e.g. MED001 to MED025)")
    horizon_days: int = Field(30, ge=1, le=90, description="Forecast horizon in days (1 to 90 days)")
    as_of_date: Optional[str] = Field("2025-12-31", description="Cut-off date for temporal point-in-time forecast")
    promotional_discount_pct: Optional[float] = Field(0.0, ge=0.0, le=0.50, description="Hypothetical promotional discount rate (0.0 to 0.50)")


class DailyDemandForecast(BaseModel):
    date: str
    forecasted_demand: float
    lower_bound: float
    upper_bound: float


class DemandPredictionResponse(BaseModel):
    product_id: str
    product_name: str
    category: str
    horizon_days: int
    as_of_date: str
    total_projected_demand: float
    average_daily_demand: float
    daily_forecasts: List[DailyDemandForecast]
    model_used: str
    disclaimer: str


class BatchRiskRequest(BaseModel):
    batch_id: str = Field(..., example="BAT-ACT-01050")
    product_id: str = Field(..., example="MED001")
    current_stock: int = Field(..., ge=1, description="Physical units on hand")
    expiry_date: str = Field(..., example="2026-01-25", description="Batch expiry date (YYYY-MM-DD)")
    unit_cost: float = Field(..., ge=0.01, description="Unit acquisition cost")
    unit_price: float = Field(..., ge=0.01, description="Unit retail price")
    supplier_name: Optional[str] = Field("Sun Pharma Distributors", description="Supplier / distributor name")
    supplier_return_window_days: Optional[int] = Field(90, ge=0, description="Days before expiry returns are accepted")
    as_of_date: Optional[str] = Field("2025-12-31", description="Evaluation date (YYYY-MM-DD)")


class BatchRiskResponse(BaseModel):
    batch_id: str
    product_id: str
    product_name: str
    current_stock: int
    expiry_date: str
    days_to_expiry: int
    cumulative_projected_demand: float
    expected_unsold_units: int
    unsold_ratio: float
    capital_at_risk_cost: float
    potential_revenue_loss: float
    risk_level: str
    risk_score: float
    supplier_return_eligible: bool
    recommended_action_code: str
    recommended_operational_action: str
    disclaimer: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    """Root landing endpoint with system summary and disclaimer."""
    return {
        "service": "Pharmacy Medicine Demand & Expiry-Risk Forecasting API",
        "version": "1.0.0",
        "documentation": "/docs",
        "advisory_notice": ADVISORY_DISCLAIMER
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint exposing model version and training metrics."""
    res = get_cached_resources()
    meta = res["metadata"]
    return {
        "status": "HEALTHY",
        "timestamp": datetime.now().isoformat(),
        "champion_model": meta.get("champion_model_name", "XGBoost_Regressor"),
        "training_timestamp": meta.get("training_timestamp"),
        "features_count": meta.get("feature_count"),
        "test_wape_pct": meta.get("held_out_test_metrics", {}).get(meta.get("champion_model_name"), {}).get("WAPE_pct"),
        "regulatory_disclaimer": ADVISORY_DISCLAIMER
    }


@app.get("/products", tags=["Catalog"])
def list_products(category: Optional[str] = None):
    """Retrieves pharmacy product catalog with optional category filtering."""
    res = get_cached_resources()
    meds_df = res["meds_df"].copy()
    
    if category:
        meds_df = meds_df[meds_df["category"].str.lower() == category.lower()]
        
    return {
        "total_products": len(meds_df),
        "products": meds_df.to_dict(orient="records"),
        "disclaimer": ADVISORY_DISCLAIMER
    }


@app.post("/predict/demand", response_model=DemandPredictionResponse, tags=["Forecasting"])
def predict_product_demand(payload: DemandPredictionRequest):
    """
    Computes day-by-day demand forecasts for a given product across a specified horizon.
    Includes 95% uncertainty intervals and sensitivity to promotional discount rate.
    """
    res = get_cached_resources()
    meds_df = res["meds_df"]
    sales_df = res["sales_df"]
    model = res["model"]
    feature_cols = res["feature_cols"]
    meta = res["metadata"]
    
    # Validate product exists
    match = meds_df[meds_df["product_id"] == payload.product_id]
    if len(match) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product ID '{payload.product_id}' not found in catalog. Valid IDs: MED001 - MED025."
        )
        
    prod_row = match.iloc[0]
    
    # Run multi-step demand forecast
    forecast_df = forecast_product_demand_horizon(
        product_id=payload.product_id,
        recent_sales_df=sales_df,
        meds_df=meds_df,
        model=model,
        feature_cols=feature_cols,
        horizon_days=payload.horizon_days,
        start_date=payload.as_of_date
    )
    
    # Adjust for promotional discount elasticity if specified
    if payload.promotional_discount_pct > 0.0:
        price_elasticity = -1.2
        promo_factor = (1.0 - payload.promotional_discount_pct) ** price_elasticity
        forecast_df["forecasted_demand"] = (forecast_df["forecasted_demand"] * promo_factor).round(2)
        forecast_df["lower_bound"] = (forecast_df["lower_bound"] * promo_factor).round(2)
        forecast_df["upper_bound"] = (forecast_df["upper_bound"] * promo_factor).round(2)
        
    total_demand = float(forecast_df["forecasted_demand"].sum())
    avg_demand = float(forecast_df["forecasted_demand"].mean())
    
    daily_items = [
        DailyDemandForecast(
            date=row["date"],
            forecasted_demand=float(row["forecasted_demand"]),
            lower_bound=float(row["lower_bound"]),
            upper_bound=float(row["upper_bound"])
        )
        for _, row in forecast_df.iterrows()
    ]
    
    return DemandPredictionResponse(
        product_id=payload.product_id,
        product_name=str(prod_row["name"]),
        category=str(prod_row["category"]),
        horizon_days=payload.horizon_days,
        as_of_date=payload.as_of_date,
        total_projected_demand=round(total_demand, 2),
        average_daily_demand=round(avg_demand, 2),
        daily_forecasts=daily_items,
        model_used=meta.get("champion_model_name", "XGBoost_Regressor"),
        disclaimer=ADVISORY_DISCLAIMER
    )


@app.post("/predict/expiry-risk", response_model=BatchRiskResponse, tags=["Expiry Risk Engine"])
def predict_batch_expiry_risk(payload: BatchRiskRequest):
    """
    Evaluates expiry risk for an active medicine batch using FEFO simulation.
    Returns quantitative risk tier, expected unsold quantity, financial capital at risk,
    and actionable operational guidance.
    """
    res = get_cached_resources()
    meds_df = res["meds_df"]
    sales_df = res["sales_df"]
    model = res["model"]
    feature_cols = res["feature_cols"]
    
    match = meds_df[meds_df["product_id"] == payload.product_id]
    if len(match) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product ID '{payload.product_id}' not found in catalog."
        )
    prod_row = match.iloc[0]
    
    as_of_dt = pd.to_datetime(payload.as_of_date)
    exp_dt = pd.to_datetime(payload.expiry_date)
    days_to_expiry = int((exp_dt - as_of_dt).days)
    
    if days_to_expiry < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch expiry date ({payload.expiry_date}) is in the past relative to as-of date ({payload.as_of_date})."
        )
        
    # Generate demand forecast up to expiry date + 14 day buffer
    forecast_horizon = max(30, min(120, days_to_expiry + 14))
    forecast_df = forecast_product_demand_horizon(
        product_id=payload.product_id,
        recent_sales_df=sales_df,
        meds_df=meds_df,
        model=model,
        feature_cols=feature_cols,
        horizon_days=forecast_horizon,
        start_date=payload.as_of_date
    )
    
    # Calculate cumulative demand up to expiry
    forecast_df["dt"] = pd.to_datetime(forecast_df["date"])
    demand_sub = forecast_df[forecast_df["dt"] <= exp_dt]
    
    if len(demand_sub) > 0:
        cum_demand = float(demand_sub["forecasted_demand"].sum())
    else:
        avg_d = float(forecast_df["forecasted_demand"].mean())
        cum_demand = float(days_to_expiry * avg_d)
        
    expected_unsold = max(0.0, float(payload.current_stock) - cum_demand)
    unsold_ratio = min(1.0, max(0.0, expected_unsold / float(payload.current_stock)))
    
    capital_at_risk = round(expected_unsold * payload.unit_cost, 2)
    revenue_loss = round(expected_unsold * payload.unit_price, 2)
    
    return_window = payload.supplier_return_window_days
    is_return_eligible = bool(days_to_expiry <= return_window)
    
    # Risk decision logic
    if (days_to_expiry <= EXPIRY_CRITICAL_DAYS and (unsold_ratio > 0.08 or expected_unsold >= 5)) or \
       (unsold_ratio >= 0.35 and days_to_expiry <= EXPIRY_WARNING_DAYS):
        risk_level = RISK_LEVEL_HIGH
        risk_score = round(min(1.0, 0.70 + 0.30 * unsold_ratio), 2)
        action_code = "FEFO_CLEARANCE_DISCOUNT"
        action_text = (
            f"HIGH RISK CRITICAL: ~{int(round(expected_unsold))} units projected unsold before expiry ({days_to_expiry} days remaining). "
            f"Prioritize for FEFO dispensing and trigger an immediate operational 20-30% clearance discount."
        )
    elif (days_to_expiry <= EXPIRY_WARNING_DAYS and (unsold_ratio > 0.05 or expected_unsold >= 3)) or \
         (days_to_expiry <= return_window and unsold_ratio >= 0.15):
        risk_level = RISK_LEVEL_MEDIUM
        risk_score = round(min(0.69, 0.40 + 0.29 * unsold_ratio), 2)
        if is_return_eligible:
            action_code = "SUPPLIER_RETURN_ELIGIBLE"
            action_text = (
                f"MEDIUM RISK WARNING: {days_to_expiry} days remaining. Batch is within the {return_window}-day return window with "
                f"{payload.supplier_name}. Initiate return RMA for credit note."
            )
        else:
            action_code = "INTER_STORE_TRANSFER"
            action_text = (
                f"MEDIUM RISK: Rotate surplus units ({int(round(expected_unsold))}) to high-volume hospital branch or bundle."
            )
    else:
        risk_level = RISK_LEVEL_LOW
        risk_score = round(max(0.05, unsold_ratio * 0.30), 2)
        action_code = "STANDARD_FEFO_MONITORING"
        action_text = (
            f"LOW RISK (HEALTHY): Cumulative demand ({cum_demand:.0f} units) safely clears stock before expiry in "
            f"{days_to_expiry} days. Maintain standard inventory dispensing."
        )
        
    return BatchRiskResponse(
        batch_id=payload.batch_id,
        product_id=payload.product_id,
        product_name=str(prod_row["name"]),
        current_stock=payload.current_stock,
        expiry_date=payload.expiry_date,
        days_to_expiry=days_to_expiry,
        cumulative_projected_demand=round(cum_demand, 1),
        expected_unsold_units=int(round(expected_unsold)),
        unsold_ratio=round(unsold_ratio, 3),
        capital_at_risk_cost=capital_at_risk,
        potential_revenue_loss=revenue_loss,
        risk_level=risk_level,
        risk_score=risk_score,
        supplier_return_eligible=is_return_eligible,
        recommended_action_code=action_code,
        recommended_operational_action=action_text,
        disclaimer=ADVISORY_DISCLAIMER
    )


@app.get("/inventory/batches", tags=["Inventory Monitoring"])
def get_active_batches_risk_report(risk_filter: Optional[str] = None):
    """Retrieves full active pharmacy batch inventory with risk scores and operational alerts."""
    risk_report_path = PROCESSED_DATA_DIR / "active_batches_risk_report.csv"
    if not risk_report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active batch risk report not found. Run pipeline first."
        )
        
    df = pd.read_csv(risk_report_path)
    if risk_filter:
        df = df[df["risk_level"].str.upper() == risk_filter.upper()]
        
    return {
        "as_of_date": "2025-12-31",
        "total_active_batches": len(df),
        "high_risk_count": int((df["risk_level"] == RISK_LEVEL_HIGH).sum()),
        "medium_risk_count": int((df["risk_level"] == RISK_LEVEL_MEDIUM).sum()),
        "low_risk_count": int((df["risk_level"] == RISK_LEVEL_LOW).sum()),
        "total_capital_at_risk": round(float(df["capital_at_risk_cost"].sum()), 2),
        "batches": df.to_dict(orient="records"),
        "disclaimer": ADVISORY_DISCLAIMER
    }

