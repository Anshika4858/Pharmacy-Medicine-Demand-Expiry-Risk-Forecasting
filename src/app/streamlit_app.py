"""
Streamlit Interactive Decision-Support Web Application.
Pharmacy Medicine Demand & Expiry-Risk Forecasting.
ALGIONS AI LABS - Assessment Project
"""
import sys
from pathlib import Path

# Add project root to sys.path so modules import reliably
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ADVISORY_DISCLAIMER,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_MEDIUM,
    RISK_LEVEL_LOW,
)
from src.models.expiry_risk_engine import (
    forecast_product_demand_horizon,
    simulate_fefo_expiry_risk,
    load_champion_model_and_features,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pharmacy Demand & Expiry-Risk Forecast",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished medical decision-support dashboard
st.markdown("""
<style>
    .reportview-container {
        background: #f8fafc;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #991b1b;
        font-size: 0.88rem;
    }
    .risk-high {
        color: #dc2626;
        font-weight: bold;
    }
    .risk-med {
        color: #d97706;
        font-weight: bold;
    }
    .risk-low {
        color: #16a34a;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_app_resources():
    model, feature_cols = load_champion_model_and_features()
    sales_df = pd.read_csv(RAW_DATA_DIR / "daily_sales.csv")
    meds_df = pd.read_csv(RAW_DATA_DIR / "medicines.csv")
    batches_df = pd.read_csv(RAW_DATA_DIR / "inventory_batches.csv")
    
    with open(MODEL_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
        
    with open(MODEL_DIR / "evaluation_metrics.json", "r") as f:
        eval_metrics = json.load(f)
        
    with open(MODEL_DIR / "feature_importance.json", "r") as f:
        feat_importance = json.load(f)
        
    active_risk_report_path = PROCESSED_DATA_DIR / "active_batches_risk_report.csv"
    if active_risk_report_path.exists():
        active_batches_risk_df = pd.read_csv(active_risk_report_path)
    else:
        active_batches_risk_df = pd.DataFrame()
        
    with open(MODEL_DIR / "expiry_risk_classification_metrics.json", "r") as f:
        class_metrics = json.load(f)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "sales_df": sales_df,
        "meds_df": meds_df,
        "batches_df": batches_df,
        "metadata": metadata,
        "eval_metrics": eval_metrics,
        "feat_importance": feat_importance,
        "active_batches_risk_df": active_batches_risk_df,
        "class_metrics": class_metrics,
    }

resources = load_app_resources()
meds_df = resources["meds_df"]
sales_df = resources["sales_df"]
model = resources["model"]
feature_cols = resources["feature_cols"]
metadata = resources["metadata"]
eval_metrics = resources["eval_metrics"]
active_risk_df = resources["active_batches_risk_df"]
class_metrics = resources["class_metrics"]

# -----------------------------------------------------------------------------
# Permanent Advisory Disclaimer Banner
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="disclaimer-box">
    <strong>⚠️ MANDATORY PHARMACEUTICAL DECISION-SUPPORT NOTICE:</strong><br>
    {ADVISORY_DISCLAIMER}
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Context
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/pharmacy-shop.png", width=64)
st.sidebar.title("Rx Foresight AI")
st.sidebar.caption("Decision-Support for Pharmacy Inventory")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive KPI Overview",
        "📈 Demand Forecasting Hub",
        "⚠️ Expiry-Risk & FEFO Batches",
        "🧪 What-If Mitigation Simulator",
        "🧠 Model Diagnostics & Governance",
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**System Specifications**")
st.sidebar.text(f"Champion: {metadata.get('champion_model_name')}")
test_wape = eval_metrics["test_comparison"].get(metadata.get("champion_model_name"), {}).get("WAPE_pct", 20.16)
st.sidebar.text(f"Test Set WAPE: {test_wape:.2f}%")
st.sidebar.text(f"As-of Date: 2025-12-31")
st.sidebar.text("Catalog Size: 25 Medicines")
st.sidebar.text(f"Active Batches: {len(active_risk_df)}")


# -----------------------------------------------------------------------------
# PAGE 1: Executive KPI Overview
# -----------------------------------------------------------------------------
if page == "📊 Executive KPI Overview":
    st.subheader("Executive Pharmacy Inventory & Expiry Risk Dashboard")
    st.markdown("Real-time synthesis of stock health, projected demand, and capital exposed to expiration risk.")
    
    # Summary Metrics
    total_active = len(active_risk_df)
    high_risk_count = int((active_risk_df["risk_level"] == RISK_LEVEL_HIGH).sum())
    med_risk_count = int((active_risk_df["risk_level"] == RISK_LEVEL_MEDIUM).sum())
    low_risk_count = int((active_risk_df["risk_level"] == RISK_LEVEL_LOW).sum())
    total_capital_risk = float(active_risk_df["capital_at_risk_cost"].sum())
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Active Batches", f"{total_active}", help="Total batches currently on dispensary shelves")
    with col2:
        st.metric("🚨 High Risk Batches", f"{high_risk_count}", delta=f"{high_risk_count} Urgent", delta_color="inverse")
    with col3:
        st.metric("⚠️ Medium Risk", f"{med_risk_count}", delta=f"{med_risk_count} Pending Review", delta_color="off")
    with col4:
        st.metric("✅ Safe Stock", f"{low_risk_count}")
    with col5:
        st.metric("Capital at Risk", f"${total_capital_risk:,.2f}", help="Cost value of expected unsold stock at expiry")
        
    st.divider()
    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Expiry Risk Tier by Therapeutic Category")
        cat_risk = active_risk_df.groupby(["category", "risk_level"]).size().reset_index(name="batch_count")
        
        chart = alt.Chart(cat_risk).mark_bar().encode(
            x=alt.X("category:N", title="Therapeutic Category"),
            y=alt.Y("batch_count:Q", title="Number of Batches"),
            color=alt.Color("risk_level:N", scale=alt.Scale(
                domain=[RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM, RISK_LEVEL_LOW],
                range=["#ef4444", "#f59e0b", "#10b981"]
            ), title="Risk Level"),
            tooltip=["category", "risk_level", "batch_count"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        
    with c2:
        st.markdown("#### Days-to-Expiry (TTE) Distribution")
        chart_tte = alt.Chart(active_risk_df).mark_bar(color="#3b82f6").encode(
            x=alt.X("days_to_expiry:Q", bin=alt.Bin(maxbins=15), title="Days Remaining Until Expiry"),
            y=alt.Y("count():Q", title="Number of Batches"),
            tooltip=["count()"]
        ).properties(height=320)
        st.altair_chart(chart_tte, use_container_width=True)
        
    st.markdown("#### Priority Operational Action Alerts")
    critical_df = active_risk_df[active_risk_df["risk_level"].isin([RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM])].copy()
    if len(critical_df) > 0:
        display_cols = [
            "batch_id", "product_name", "category", "days_to_expiry", "current_stock",
            "expected_unsold_units", "capital_at_risk_cost", "risk_level", "recommended_operational_action"
        ]
        st.dataframe(
            critical_df[display_cols].sort_values("days_to_expiry"),
            column_config={
                "capital_at_risk_cost": st.column_config.NumberColumn("Capital at Risk ($)", format="$%.2f"),
                "days_to_expiry": st.column_config.NumberColumn("Days to Expiry"),
                "expected_unsold_units": st.column_config.NumberColumn("Projected Unsold"),
                "recommended_operational_action": st.column_config.TextColumn("Advisory Action", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("No active stock batches currently in High or Medium risk status.")


# -----------------------------------------------------------------------------
# PAGE 2: Demand Forecasting Hub
# -----------------------------------------------------------------------------
elif page == "📈 Demand Forecasting Hub":
    st.subheader("Pharmaceutical Product-Level Demand Forecasting")
    st.markdown("Interactive multi-step forward demand projections generated by the trained ML champion model.")
    
    col_sel, col_horiz, col_disc = st.columns([2, 1, 1])
    with col_sel:
        prod_options = meds_df["product_id"] + " - " + meds_df["name"] + " (" + meds_df["category"] + ")"
        selected_prod_str = st.selectbox("Select Medicine SKU", prod_options)
        selected_prod_id = selected_prod_str.split(" - ")[0]
        selected_prod_meta = meds_df[meds_df["product_id"] == selected_prod_id].iloc[0]
        
    with col_horiz:
        horizon_days = st.slider("Prediction Horizon (Days)", min_value=7, max_value=90, value=30, step=7)
        
    with col_disc:
        discount_rate = st.slider("Simulated Discount", min_value=0.0, max_value=0.40, value=0.0, step=0.05, format="%.0f%%")
        
    # Generate Forecast
    with st.spinner("Generating multi-step demand forecast..."):
        forecast_df = forecast_product_demand_horizon(
            product_id=selected_prod_id,
            recent_sales_df=sales_df,
            meds_df=meds_df,
            model=model,
            feature_cols=feature_cols,
            horizon_days=horizon_days,
            start_date="2025-12-31"
        )
        
        # Apply elasticity if discount set
        if discount_rate > 0.0:
            elasticity = -1.2
            promo_mult = (1.0 - discount_rate) ** elasticity
            forecast_df["forecasted_demand"] = (forecast_df["forecasted_demand"] * promo_mult).round(2)
            forecast_df["lower_bound"] = (forecast_df["lower_bound"] * promo_mult).round(2)
            forecast_df["upper_bound"] = (forecast_df["upper_bound"] * promo_mult).round(2)
            
    # Product Summary Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Therapeutic Class", selected_prod_meta["category"])
    with m2:
        st.metric("Retail Price", f"${selected_prod_meta['unit_price']:.2f}")
    with m3:
        st.metric("Unit Cost", f"${selected_prod_meta['unit_cost']:.2f}")
    with m4:
        total_fcst = forecast_df["forecasted_demand"].sum()
        st.metric(f"Total {horizon_days}D Demand", f"{total_fcst:,.0f} units")
    with m5:
        est_rev = total_fcst * selected_prod_meta["unit_price"] * (1.0 - discount_rate)
        st.metric("Projected Revenue", f"${est_rev:,.2f}")
        
    # Chart: Historical Sales + Forward Forecast
    hist_subset = sales_df[sales_df["product_id"] == selected_prod_id].tail(60).copy()
    hist_chart_df = pd.DataFrame({
        "date": hist_subset["date"],
        "units": hist_subset["units_sold"],
        "series": "Historical Actual"
    })
    
    fcst_chart_df = pd.DataFrame({
        "date": forecast_df["date"],
        "units": forecast_df["forecasted_demand"],
        "series": "Model Forecast"
    })
    
    combined_df = pd.concat([hist_chart_df, fcst_chart_df], ignore_index=True)
    combined_df["date"] = pd.to_datetime(combined_df["date"])
    
    st.markdown("#### Demand Velocity & Forward Horizon Projection")
    base_line = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("units:Q", title="Daily Units Sold / Forecasted"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Historical Actual", "Model Forecast"],
            range=["#64748b", "#2563eb"]
        ), title="Timeline"),
        tooltip=["date:T", "units:Q", "series:N"]
    ).properties(height=380)
    
    # Confidence Interval Ribbon on Forecast
    forecast_df["dt"] = pd.to_datetime(forecast_df["date"])
    band = alt.Chart(forecast_df).mark_area(opacity=0.20, color="#2563eb").encode(
        x="dt:T",
        y="lower_bound:Q",
        y2="upper_bound:Q"
    )
    
    st.altair_chart(base_line + band, use_container_width=True)
    
    # Model Comparison Option
    with st.expander("🔬 Compare Model Predictions with Moving Average Baseline"):
        ma_baseline = hist_subset["units_sold"].tail(7).mean()
        st.write(f"**7-Day Moving Average Baseline:** ~{ma_baseline:.1f} units/day (Flat extrapolation)")
        st.write(f"**ML Champion ({metadata.get('champion_model_name')}):** Dynamically tracks seasonal shifts and day-of-week surges.")


# -----------------------------------------------------------------------------
# PAGE 3: Expiry-Risk & FEFO Batches
# -----------------------------------------------------------------------------
elif page == "⚠️ Expiry-Risk & FEFO Batches":
    st.subheader("Batch Expiry & FEFO Depletion Monitor")
    st.markdown("First-Expired, First-Out (FEFO) simulation mapping active stock batches against forward demand projections.")
    
    f1, f2 = st.columns([1, 2])
    with f1:
        risk_filter = st.selectbox("Filter by Risk Level", ["ALL", RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM, RISK_LEVEL_LOW])
    with f2:
        search_query = st.text_input("Search Medicine or Batch ID", "")
        
    df_view = active_risk_df.copy()
    if risk_filter != "ALL":
        df_view = df_view[df_view["risk_level"] == risk_filter]
    if search_query:
        df_view = df_view[
            df_view["product_name"].str.contains(search_query, case=False) |
            df_view["batch_id"].str.contains(search_query, case=False)
        ]
        
    st.write(f"Showing **{len(df_view)}** matching active stock batches.")
    
    st.dataframe(
        df_view[[
            "batch_id", "product_name", "category", "days_to_expiry", "current_stock",
            "cumulative_demand_to_expiry", "expected_unsold_units", "unsold_ratio",
            "capital_at_risk_cost", "risk_level", "is_return_window_open", "recommended_operational_action"
        ]],
        column_config={
            "days_to_expiry": st.column_config.ProgressColumn("Days to Expiry", min_value=0, max_value=365, format="%d days"),
            "unsold_ratio": st.column_config.ProgressColumn("Unsold Risk Ratio", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "capital_at_risk_cost": st.column_config.NumberColumn("Capital at Risk", format="$%.2f"),
            "is_return_window_open": st.column_config.CheckboxColumn("Vendor Return Window Open"),
            "recommended_operational_action": st.column_config.TextColumn("Action Guidance", width="large"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Download report
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Active Risk Report (CSV)",
        data=csv_bytes,
        file_name="pharmacy_active_batch_risk_report.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# PAGE 4: What-If Mitigation Simulator
# -----------------------------------------------------------------------------
elif page == "🧪 What-If Mitigation Simulator":
    st.subheader("Operational What-If Mitigation Simulator")
    st.markdown(
        "Evaluate proactive risk-reduction interventions: clearance discounts, vendor returns (RMA), "
        "and inter-branch transfers to rescue capital before batch expiration."
    )
    
    # Select candidate batch (default to a high/medium risk batch)
    high_med_batches = active_risk_df[active_risk_df["risk_level"].isin([RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM])]
    if len(high_med_batches) > 0:
        batch_choices = high_med_batches["batch_id"] + " (" + high_med_batches["product_name"] + " - " + high_med_batches["risk_level"] + ")"
    else:
        batch_choices = active_risk_df["batch_id"] + " (" + active_risk_df["product_name"] + ")"
        
    selected_b_str = st.selectbox("Select Target Stock Batch to Simulate", batch_choices)
    selected_bid = selected_b_str.split(" ")[0]
    batch_record = active_risk_df[active_risk_df["batch_id"] == selected_bid].iloc[0]
    
    st.divider()
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown("#### Baseline Status (No Intervention)")
        st.write(f"**Product:** {batch_record['product_name']} ({batch_record['category']})")
        st.write(f"**Current Shelf Stock:** {batch_record['current_stock']} units")
        st.write(f"**Days to Expiry:** {batch_record['days_to_expiry']} days (Expires {batch_record['expiry_date']})")
        st.write(f"**Supplier:** {batch_record['supplier_name']} (Return Window: {batch_record['supplier_return_window_days']} days)")
        st.write(f"**Expected Unsold Units:** {batch_record['expected_unsold_units']} units")
        st.write(f"**Baseline Capital at Risk:** ${batch_record['capital_at_risk_cost']:,.2f}")
        st.markdown(f"**Current Status:** <span class='risk-high'>{batch_record['risk_level']}</span>", unsafe_allow_html=True)
        
    with c_right:
        st.markdown("#### Test Mitigation Interventions")
        
        sim_discount = st.slider("Clearance Markdown Discount (%)", 0, 40, 20, 5)
        sim_transfer = st.slider("Inter-Store Transfer to High-Volume Branch (Units)", 0, int(batch_record["current_stock"]), 0, 10)
        sim_vendor_return = st.checkbox("Execute Supplier Return (RMA Credit)", value=bool(batch_record["is_return_window_open"]))
        
        # Calculate dynamic impact
        effective_stock_after_transfer = max(0, batch_record["current_stock"] - sim_transfer)
        
        # Demand increase from price elasticity
        if sim_discount > 0:
            elasticity = -1.2
            boost = (1.0 - (sim_discount / 100.0)) ** elasticity
        else:
            boost = 1.0
            
        sim_demand = batch_record["effective_demand_allocated"] * boost
        
        if sim_vendor_return:
            # All return-eligible stock is credited back to supplier
            sim_unsold = 0
            capital_saved = batch_record["capital_at_risk_cost"]
            new_risk = "RESOLVED (VENDOR RETURN)"
            new_action = f"Returned to {batch_record['supplier_name']} within {batch_record['supplier_return_window_days']}-day window. 100% cost recovered."
        else:
            sim_unsold = max(0, int(round(effective_stock_after_transfer - sim_demand)))
            units_rescued = max(0, batch_record["expected_unsold_units"] - sim_unsold)
            capital_saved = units_rescued * batch_record["capital_at_risk_cost"] / (batch_record["expected_unsold_units"] + 1e-6)
            
            if sim_unsold == 0:
                new_risk = "LOW_RISK (SAFE)"
                new_action = "Simulated discount/transfer fully absorbs remaining inventory before expiry."
            elif sim_unsold < batch_record["expected_unsold_units"] * 0.5:
                new_risk = "MEDIUM_RISK"
                new_action = f"Reduced unsold units from {batch_record['expected_unsold_units']} down to {sim_unsold}."
            else:
                new_risk = batch_record["risk_level"]
                new_action = "Intervention insufficient; additional transfer or higher clearance needed."
                
        st.markdown("#### Simulated Outcome")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("New Projected Unsold", f"{sim_unsold} units", delta=f"-{max(0, batch_record['expected_unsold_units'] - sim_unsold)} units", delta_color="inverse")
        with r2:
            st.metric("Capital Rescued ($)", f"${capital_saved:,.2f}", delta=f"+${capital_saved:,.2f}")
            
        st.info(f"**Outcome Assessment:** {new_action}")


# -----------------------------------------------------------------------------
# PAGE 5: Model Diagnostics & Governance
# -----------------------------------------------------------------------------
elif page == "🧠 Model Diagnostics & Governance":
    st.subheader("Model Explainability, Diagnostics & Regulatory Governance")
    st.markdown("Comprehensive validation evidence, feature importance, error analysis, and clinical safety guardrails.")
    
    tab_exp, tab_lead, tab_matrix, tab_gov = st.tabs([
        "🔍 Feature Importance",
        "🏆 Model Leaderboard",
        "📊 Expiry Backtest Matrix",
        "⚖️ Responsible AI & Governance"
    ])
    
    with tab_exp:
        st.markdown("#### Top Predictive Drivers for Pharmaceutical Demand")
        feat_dict = resources["feat_importance"]
        top_feats = dict(list(feat_dict.items())[:15])
        
        df_feat = pd.DataFrame({
            "feature": list(top_feats.keys()),
            "importance": list(top_feats.values())
        }).sort_values("importance", ascending=True)
        
        feat_chart = alt.Chart(df_feat).mark_bar(color="#0284c7").encode(
            x=alt.X("importance:Q", title="Relative Gain / Importance"),
            y=alt.Y("feature:N", sort=None, title="Model Feature"),
            tooltip=["feature", "importance"]
        ).properties(height=380)
        st.altair_chart(feat_chart, use_container_width=True)
        st.caption("Lagged demand (lag_7, lag_1, lag_14) and rolling 7-day velocity dominate short-term dispensary predictability.")
        
    with tab_lead:
        st.markdown("#### Test Set Model Comparison Leaderboard")
        st.markdown("Evaluated strictly on held-out temporal partition (2025-10-01 to 2025-12-31) with zero target lookahead.")
        test_comp = eval_metrics["test_comparison"]
        df_leader = pd.DataFrame.from_dict(test_comp, orient="index").reset_index()
        df_leader.columns = ["Model", "MAE", "RMSE", "WAPE", "WAPE (%)", "R² Score"]
        st.dataframe(
            df_leader.sort_values("WAPE"),
            hide_index=True,
            use_container_width=True
        )
        
    with tab_matrix:
        st.markdown("#### Expiry-Risk Classification Backtest")
        cm = class_metrics["confusion_matrix"]
        m = class_metrics["metrics"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m['Precision']*100:.1f}%")
        c2.metric("Recall (Sensitivity)", f"{m['Recall_Sensitivity']*100:.1f}%")
        c3.metric("F1-Score", f"{m['F1_Score']*100:.1f}%")
        c4.metric("Specificity", f"{m['Specificity']*100:.1f}%")
        
        st.markdown("##### Confusion Matrix")
        cm_df = pd.DataFrame({
            "Predicted Risk": [cm["TP"], cm["FP"]],
            "Predicted Safe": [cm["FN"], cm["TN"]]
        }, index=["Actual Expired Waste", "Actual Cleanly Sold"])
        st.table(cm_df)
        st.caption("Engine intentionally optimizes for High Recall (Sensitivity >= 85%) to prevent catastrophic expired drug write-offs.")
        
    with tab_gov:
        st.markdown("#### Responsible AI, Limitations & Failure Modes")
        st.markdown("""
        1. **Advisory Decision Support:** Model outputs are strictly operational recommendations for inventory replenishment and FEFO rotation. They never dictate therapeutic treatment, drug substitution, or patient dispensing.
        2. **Privacy Compliance (HIPAA / GDPR / DPDP):** All training data is strictly aggregated at the SKU-date level. Zero patient health information (PHI), prescriptions, or personally identifiable data is accessed or stored.
        3. **Failure Modes & Shocks:** The model relies on historical time-series patterns. Unprecedented health emergencies (e.g. pandemic waves), abrupt national drug bans (CDSCO/FDA recalls), or sudden manufacturer plant shutdowns require manual pharmacist intervention.
        4. **Cold Chain Integrity:** Physical drug efficacy depends on cold-chain adherence (2-8°C for Insulins/Biologics). The model assumes standard storage compliance; any cold-chain breach invalidates shelf life regardless of calendar expiry.
        """)

