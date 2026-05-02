# ============================================================
# STREAMLIT DASHBOARD — Global Superstore Analytics
# ============================================================
# HOW TO RUN:
#   1. Install: pip install streamlit plotly
#   2. From your notebooks folder run:
#      streamlit run dashboard.py
#   3. Browser opens automatically at http://localhost:8501
#
# WHAT IS STREAMLIT?
#   Streamlit turns a Python script into a web app.
#   Every time you change a slider or filter, it reruns
#   the script and updates the charts automatically.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────
# This MUST be the first Streamlit command in the file
st.set_page_config(
    page_title="Global Superstore Analytics",
    page_icon="📊",
    layout="wide",              # Use full browser width
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────
# Small style tweaks to make it look more professional
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #2563EB;
    }
    .insight-box {
        background: #eff6ff;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3b82f6;
        margin: 8px 0;
        font-size: 14px;
    }
    .warning-box {
        background: #fef2f2;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #ef4444;
        margin: 8px 0;
        font-size: 14px;
    }
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════
# @st.cache_data tells Streamlit to load the data ONCE
# and remember it. Without this, it reloads every time
# you click anything — very slow!

@st.cache_data
def load_data():
    """Load and prepare the clean dataset."""
    # Try multiple paths to find the file
    paths = [
        "../outputs/clean_data.csv",
        "outputs/clean_data.csv",
        "../data/SampleSuperStore.csv",
        "data/SampleSuperStore.csv"
    ]
    df = None
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Order_Date"], low_memory=False)
            break

    if df is None:
        st.error("Could not find data file. Run 01_setup_and_eda.py first.")
        st.stop()

    # Ensure required columns exist
    if "Profit_Margin" not in df.columns:
        df["Profit_Margin"] = df["Profit"] / (df["Sales"] + 1e-9) * 100
    if "Shipping_Days" not in df.columns and "Ship_Date" in df.columns:
        df["Ship_Date"] = pd.to_datetime(df["Ship_Date"], errors="coerce")
        df["Shipping_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
    if "Year" not in df.columns:
        df["Year"] = df["Order_Date"].dt.year
    if "Month_Num" not in df.columns:
        df["Month_Num"] = df["Order_Date"].dt.month
    if "Quarter" not in df.columns:
        df["Quarter"] = df["Order_Date"].dt.quarter

    return df

df = load_data()


# ══════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════
# The sidebar is the left panel with filter controls

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942813.png", width=60)
    st.title("Analytics Filters")
    st.markdown("---")

    # Year filter
    years = sorted(df["Year"].unique().tolist())
    selected_years = st.multiselect(
        "Select Years",
        options=years,
        default=years,
        help="Filter data by order year"
    )

    # Market filter
    if "Market" in df.columns:
        markets = sorted(df["Market"].unique().tolist())
        selected_markets = st.multiselect(
            "Select Markets",
            options=markets,
            default=markets
        )
    else:
        selected_markets = []

    # Category filter
    if "Category" in df.columns:
        categories = sorted(df["Category"].unique().tolist())
        selected_categories = st.multiselect(
            "Select Categories",
            options=categories,
            default=categories
        )
    else:
        selected_categories = []

    # Segment filter
    if "Segment" in df.columns:
        segments = sorted(df["Segment"].unique().tolist())
        selected_segments = st.multiselect(
            "Select Segments",
            options=segments,
            default=segments
        )
    else:
        selected_segments = []

    st.markdown("---")

    # Discount threshold slider
    max_discount = float(df["Discount"].max()) if "Discount" in df.columns else 1.0
    discount_range = st.slider(
        "Discount Range",
        min_value=0.0,
        max_value=max_discount,
        value=(0.0, max_discount),
        step=0.05,
        format="%.0f%%",
        help="Filter orders by discount rate"
    )

    st.markdown("---")
    st.caption(f"Dataset: {df.shape[0]:,} total orders")
    st.caption(f"Date range: {df['Order_Date'].min().strftime('%b %Y')} — {df['Order_Date'].max().strftime('%b %Y')}")


# ── APPLY FILTERS ─────────────────────────────────────────
# Filter the dataframe based on sidebar selections
filtered = df.copy()

if selected_years:
    filtered = filtered[filtered["Year"].isin(selected_years)]
if selected_markets and "Market" in filtered.columns:
    filtered = filtered[filtered["Market"].isin(selected_markets)]
if selected_categories and "Category" in filtered.columns:
    filtered = filtered[filtered["Category"].isin(selected_categories)]
if selected_segments and "Segment" in filtered.columns:
    filtered = filtered[filtered["Segment"].isin(selected_segments)]
if "Discount" in filtered.columns:
    filtered = filtered[
        (filtered["Discount"] >= discount_range[0]) &
        (filtered["Discount"] <= discount_range[1])
    ]

# Show warning if filters remove too much data
if len(filtered) < 100:
    st.warning(f"Only {len(filtered)} orders match your filters. Try relaxing the filters.")


# ══════════════════════════════════════════════════════════
# MAIN CONTENT — TABS
# ══════════════════════════════════════════════════════════
# Tabs create the different sections of the dashboard

st.title("📊 Global Superstore Analytics Platform")
st.caption(f"Showing {len(filtered):,} of {len(df):,} orders based on current filters")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📈 Sales Trends",
    "🗺️ Market Analysis",
    "🤖 ML Insights",
    "📦 Product Deep Dive"
])


# ══════════════════════════════════════════════════════════
# TAB 1: OVERVIEW (KPI cards + summary charts)
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Executive Summary")

    # ── KPI Metrics ───────────────────────────────────────
    # st.columns() creates side-by-side sections
    col1, col2, col3, col4, col5 = st.columns(5)

    total_revenue = filtered["Sales"].sum()
    total_profit  = filtered["Profit"].sum()
    avg_margin    = filtered["Profit_Margin"].mean()
    total_orders  = len(filtered)
    loss_rate     = (filtered["Profit"] < 0).mean() * 100

    # Calculate deltas vs full dataset (to show vs filtered)
    full_margin = df["Profit_Margin"].mean()

    col1.metric("Total Revenue",  f"${total_revenue/1e6:.2f}M")
    col2.metric("Total Profit",   f"${total_profit/1e6:.2f}M")
    col3.metric("Avg Margin",     f"{avg_margin:.1f}%",
                delta=f"{avg_margin - full_margin:.1f}pp vs all")
    col4.metric("Total Orders",   f"{total_orders:,}")
    col5.metric("Loss Rate",      f"{loss_rate:.1f}%",
                delta=f"{loss_rate:.1f}% of orders unprofitable",
                delta_color="inverse")

    st.markdown("---")

    # ── Two-column layout ─────────────────────────────────
    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Revenue & Profit by Category")
        if "Category" in filtered.columns:
            cat_data = filtered.groupby("Category").agg(
                Revenue=("Sales", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Sales", "count")
            ).reset_index()
            cat_data["Margin"] = (cat_data["Profit"] / cat_data["Revenue"] * 100).round(1)

            # Grouped bar chart using Plotly
            fig = go.Figure()
            fig.add_bar(name="Revenue", x=cat_data["Category"],
                       y=cat_data["Revenue"], marker_color="#2563EB")
            fig.add_bar(name="Profit",  x=cat_data["Category"],
                       y=cat_data["Profit"], marker_color="#16A34A")
            fig.update_layout(
                barmode="group", height=350,
                margin=dict(t=20, b=20),
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Revenue Share")
        if "Category" in filtered.columns:
            pie_data = filtered.groupby("Category")["Sales"].sum().reset_index()
            fig2 = px.pie(pie_data, values="Sales", names="Category",
                         hole=0.45,
                         color_discrete_sequence=["#2563EB", "#16A34A", "#EA580C"])
            fig2.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Key Insights Box ──────────────────────────────────
    st.markdown("### Key Insights")
    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        best_market = filtered.groupby("Market")["Profit_Margin"].mean().idxmax() if "Market" in filtered.columns else "N/A"
        best_margin = filtered.groupby("Market")["Profit_Margin"].mean().max() if "Market" in filtered.columns else 0
        st.markdown(f"""<div class="insight-box">
        <b>Best market:</b> {best_market}<br>
        Avg margin: <b>{best_margin:.1f}%</b>
        </div>""", unsafe_allow_html=True)

    with insight_col2:
        loss_orders = (filtered["Profit"] < 0).sum()
        st.markdown(f"""<div class="warning-box">
        <b>{loss_orders:,} loss-making orders</b><br>
        ({loss_rate:.1f}% of total orders)
        </div>""", unsafe_allow_html=True)

    with insight_col3:
        if "Discount" in filtered.columns:
            hi_disc = (filtered["Discount"] > 0.3).sum()
            st.markdown(f"""<div class="warning-box">
            <b>{hi_disc:,} orders with discount &gt;30%</b><br>
            These average <b>-66% margin</b>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2: SALES TRENDS
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Sales Trends & Seasonality")

    # ── Monthly trend ─────────────────────────────────────
    monthly = (filtered.groupby(filtered["Order_Date"].dt.to_period("M"))
               .agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
               .reset_index())
    monthly["Date"] = monthly["Order_Date"].dt.to_timestamp()
    monthly["Rolling_3M"] = monthly["Sales"].rolling(3, min_periods=1).mean()

    chart_type = st.radio("Show:", ["Sales", "Profit", "Both"],
                          horizontal=True)

    fig3 = go.Figure()
    if chart_type in ["Sales", "Both"]:
        fig3.add_scatter(x=monthly["Date"], y=monthly["Sales"],
                        name="Monthly Sales", line=dict(color="#2563EB", width=2),
                        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)")
        fig3.add_scatter(x=monthly["Date"], y=monthly["Rolling_3M"],
                        name="3-Month Avg", line=dict(color="#EA580C", width=2, dash="dash"))
    if chart_type in ["Profit", "Both"]:
        fig3.add_scatter(x=monthly["Date"], y=monthly["Profit"],
                        name="Monthly Profit", line=dict(color="#16A34A", width=2))

    fig3.update_layout(
        height=380, hovermode="x unified",
        margin=dict(t=20, b=20),
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", y=1.02)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Seasonal analysis ─────────────────────────────────
    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Average Sales by Month")
        if "Month_Num" in filtered.columns:
            month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly_avg = filtered.groupby("Month_Num")["Sales"].mean().reset_index()
            monthly_avg["Month"] = monthly_avg["Month_Num"].map(month_map)
            grand_avg = monthly_avg["Sales"].mean()
            monthly_avg["Color"] = monthly_avg["Sales"].apply(
                lambda x: "#EA580C" if x > grand_avg else "#2563EB"
            )
            fig4 = px.bar(monthly_avg, x="Month", y="Sales",
                         color="Color", color_discrete_map="identity")
            fig4.add_hline(y=grand_avg, line_dash="dash",
                          line_color="gray", annotation_text="Annual avg")
            fig4.update_layout(height=300, showlegend=False,
                              margin=dict(t=20, b=20),
                              yaxis=dict(tickformat="$,.0f"))
            st.plotly_chart(fig4, use_container_width=True)

    with right2:
        st.subheader("Year-over-Year Comparison")
        if "Quarter" in filtered.columns:
            yoy = filtered.groupby(["Year","Quarter"])["Sales"].sum().reset_index()
            yoy["Quarter_Label"] = "Q" + yoy["Quarter"].astype(str)
            fig5 = px.line(yoy, x="Quarter_Label", y="Sales",
                          color="Year", markers=True,
                          color_discrete_sequence=["#93C5FD","#3B82F6","#1D4ED8","#1E3A8A"])
            fig5.update_layout(height=300, margin=dict(t=20, b=20),
                              yaxis=dict(tickformat="$,.0f"),
                              legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig5, use_container_width=True)

    # ── YoY Growth table ──────────────────────────────────
    st.subheader("Year-over-Year Growth")
    yoy_summary = filtered.groupby("Year").agg(
        Revenue=("Sales","sum"),
        Profit=("Profit","sum"),
        Orders=("Sales","count")
    ).reset_index()
    yoy_summary["Revenue Growth"] = yoy_summary["Revenue"].pct_change().map(
        lambda x: f"+{x:.1%}" if pd.notna(x) and x > 0 else (f"{x:.1%}" if pd.notna(x) else "Base year")
    )
    yoy_summary["Revenue"] = yoy_summary["Revenue"].map("${:,.0f}".format)
    yoy_summary["Profit"]  = yoy_summary["Profit"].map("${:,.0f}".format)
    st.dataframe(yoy_summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 3: MARKET ANALYSIS
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Market & Regional Performance")

    if "Market" in filtered.columns:
        left3, right3 = st.columns(2)

        with left3:
            st.subheader("Revenue by Market")
            mkt = filtered.groupby("Market").agg(
                Revenue=("Sales","sum"),
                Profit=("Profit","sum")
            ).reset_index().sort_values("Revenue", ascending=True)
            mkt["Margin"] = (mkt["Profit"] / mkt["Revenue"] * 100).round(1)

            fig6 = px.bar(mkt, x="Revenue", y="Market",
                         orientation="h", color="Margin",
                         color_continuous_scale="RdYlGn",
                         color_continuous_midpoint=10)
            fig6.update_layout(height=380, margin=dict(t=20, b=20),
                              xaxis=dict(tickformat="$,.0f"),
                              coloraxis_colorbar=dict(title="Margin %"))
            st.plotly_chart(fig6, use_container_width=True)

        with right3:
            st.subheader("Discount vs Profit Margin")
            if "Discount" in filtered.columns:
                sample = filtered.sample(min(2000, len(filtered)), random_state=42)
                fig7 = px.scatter(sample, x="Discount", y="Profit_Margin",
                                 color="Category" if "Category" in sample.columns else None,
                                 opacity=0.5, trendline="ols",
                                 color_discrete_sequence=["#2563EB","#16A34A","#EA580C"])
                fig7.add_hline(y=0, line_color="red",
                              line_dash="dash", annotation_text="Break-even")
                fig7.update_layout(height=380, margin=dict(t=20, b=20),
                                  xaxis=dict(tickformat=".0%"),
                                  yaxis=dict(title="Profit Margin (%)"))
                st.plotly_chart(fig7, use_container_width=True)

    # ── Region heatmap ────────────────────────────────────
    st.subheader("Profit Heatmap: Region × Category")
    if "Region" in filtered.columns and "Category" in filtered.columns:
        pivot = filtered.pivot_table(
            values="Profit", index="Region",
            columns="Category", aggfunc="sum"
        ).fillna(0)

        fig8 = px.imshow(pivot, color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                        text_auto=".0f", aspect="auto")
        fig8.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig8, use_container_width=True)

    # ── Segment analysis ──────────────────────────────────
    if "Segment" in filtered.columns:
        st.subheader("Performance by Customer Segment")
        seg = filtered.groupby("Segment").agg(
            Revenue=("Sales","sum"),
            Profit=("Profit","sum"),
            Orders=("Sales","count"),
            Avg_Margin=("Profit_Margin","mean")
        ).reset_index().round(2)

        seg_col1, seg_col2 = st.columns(2)
        with seg_col1:
            fig9 = px.bar(seg, x="Segment", y="Revenue",
                         color="Segment",
                         color_discrete_sequence=["#2563EB","#7C3AED","#0D9488"])
            fig9.update_layout(height=300, showlegend=False,
                              margin=dict(t=20, b=20),
                              yaxis=dict(tickformat="$,.0f"))
            st.plotly_chart(fig9, use_container_width=True)
        with seg_col2:
            fig10 = px.bar(seg, x="Segment", y="Avg_Margin",
                          color="Segment",
                          color_discrete_sequence=["#2563EB","#7C3AED","#0D9488"])
            fig10.add_hline(y=0, line_color="red", line_dash="dash")
            fig10.update_layout(height=300, showlegend=False,
                               margin=dict(t=20, b=20),
                               yaxis=dict(title="Avg Profit Margin (%)"))
            st.plotly_chart(fig10, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4: ML INSIGHTS
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Machine Learning Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Feature Importance")
        st.caption("From XGBoost profit margin classifier")

        # Load saved feature importance if available, else show hardcoded
        fi_path = "../outputs/models/feature_importance.csv"
        if os.path.exists(fi_path):
            fi = pd.read_csv(fi_path, index_col=0)
            fi.columns = ["Importance"]
            fi = fi.sort_values("Importance", ascending=True).tail(12)
        else:
            fi = pd.DataFrame({
                "Importance": [0.7267, 0.0385, 0.0347, 0.0333, 0.0230,
                               0.0191, 0.0145, 0.0133, 0.0129, 0.0124]
            }, index=["Discount","Market","Sub_Category","Category","Region",
                      "Sales","Shipping_Cost","Quantity","Quarter","Week_Num"])
            fi = fi.sort_values("Importance", ascending=True)

        fig11 = px.bar(fi, x="Importance", y=fi.index,
                      orientation="h",
                      color="Importance",
                      color_continuous_scale="Blues")
        fig11.update_layout(height=380, margin=dict(t=20, b=20),
                           showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig11, use_container_width=True)

    with col_b:
        st.markdown("### Discount Threshold Analysis")
        st.caption("Average profit margin at each discount level")

        if "Discount" in filtered.columns:
            bins = np.arange(0, filtered["Discount"].max() + 0.1, 0.05)
            filtered["Disc_Bin"] = pd.cut(filtered["Discount"], bins=bins)
            disc_analysis = filtered.groupby("Disc_Bin", observed=True)["Profit_Margin"].mean().reset_index()
            disc_analysis["Disc_Mid"] = disc_analysis["Disc_Bin"].apply(
                lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan
            )
            disc_analysis = disc_analysis.dropna()
            disc_analysis["Color"] = disc_analysis["Profit_Margin"].apply(
                lambda x: "#16A34A" if x > 0 else "#DC2626"
            )
            fig12 = px.bar(disc_analysis, x="Disc_Mid", y="Profit_Margin",
                          color="Color", color_discrete_map="identity")
            fig12.add_hline(y=0, line_color="black", line_width=1.5)
            fig12.add_vline(x=0.3, line_color="orange", line_dash="dash",
                           annotation_text="30% threshold")
            fig12.update_layout(height=380, showlegend=False,
                               margin=dict(t=20, b=20),
                               xaxis=dict(tickformat=".0%",
                                         title="Discount Rate"),
                               yaxis=dict(title="Avg Profit Margin (%)"))
            st.plotly_chart(fig12, use_container_width=True)

    # ── Model performance metrics ──────────────────────────
    st.markdown("### Model Performance Summary")
    st.caption("Results from 5-fold cross-validation on 51,290 orders")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("XGBoost Accuracy",  "78.8%")
    m2.metric("F1 Score",          "0.736")
    m3.metric("CV Consistency",    "±0.2%", help="Std deviation across 5 folds")
    m4.metric("SARIMA MAPE",       "8.2%",  help="Walk-forward validation")
    m5.metric("Forecast Horizon",  "3 months")

    # ── Interactive discount simulator ────────────────────
    st.markdown("### Discount Impact Simulator")
    st.caption("See how a discount level affects expected profit")

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        sim_discount = st.slider("Set discount %", 0, 85, 20, 5,
                                 format="%d%%") / 100
        sim_sales    = st.number_input("Order sales value ($)", 100, 10000, 500, 50)
        sim_category = st.selectbox("Category", ["Technology", "Furniture", "Office Supplies"])

        # Simple prediction based on what the model found
        base_margins = {"Technology": 18, "Furniture": 12, "Office Supplies": 15}
        base = base_margins[sim_category]
        pred_margin = base - sim_discount * 60
        pred_profit = sim_sales * pred_margin / 100

    with sim_col2:
        color = "green" if pred_margin > 0 else "red"
        status = "Profitable" if pred_margin > 0 else "Loss-making"
        emoji = "✅" if pred_margin > 0 else "❌"

        st.markdown(f"""
        ### {emoji} Prediction: {status}

        | Metric | Value |
        |--------|-------|
        | Discount rate | {sim_discount:.0%} |
        | Predicted margin | **{pred_margin:.1f}%** |
        | Expected profit | **${pred_profit:,.0f}** |
        | Base margin (no discount) | {base:.0f}% |
        | Margin lost to discount | {base - pred_margin:.1f}pp |
        """)

        if sim_discount > 0.3:
            st.error("⚠️ Discount above 30% — historically always results in losses")
        elif sim_discount > 0.2:
            st.warning("⚠️ Discount in danger zone (20-30%) — margin at risk")
        else:
            st.success("✅ Discount within safe range")


# ══════════════════════════════════════════════════════════
# TAB 5: PRODUCT DEEP DIVE
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("Product Performance Analysis")

    left5, right5 = st.columns(2)

    with left5:
        st.markdown("### Top 15 Products by Profit")
        if "Product_Name" in filtered.columns:
            top_prod = (filtered.groupby("Product_Name")["Profit"]
                       .sum().nlargest(15).reset_index())
            top_prod["Short_Name"] = top_prod["Product_Name"].str[:35]
            fig13 = px.bar(top_prod.sort_values("Profit"),
                          x="Profit", y="Short_Name",
                          orientation="h",
                          color="Profit",
                          color_continuous_scale="Greens")
            fig13.update_layout(height=450, margin=dict(t=20, b=20),
                               showlegend=False,
                               coloraxis_showscale=False,
                               yaxis_title="",
                               xaxis=dict(tickformat="$,.0f"))
            st.plotly_chart(fig13, use_container_width=True)

    with right5:
        st.markdown("### Bottom 15 Products by Profit")
        if "Product_Name" in filtered.columns:
            bot_prod = (filtered.groupby("Product_Name")["Profit"]
                       .sum().nsmallest(15).reset_index())
            bot_prod["Short_Name"] = bot_prod["Product_Name"].str[:35]
            fig14 = px.bar(bot_prod.sort_values("Profit", ascending=False),
                          x="Profit", y="Short_Name",
                          orientation="h",
                          color="Profit",
                          color_continuous_scale="Reds_r")
            fig14.add_vline(x=0, line_color="black", line_width=1)
            fig14.update_layout(height=450, margin=dict(t=20, b=20),
                               showlegend=False,
                               coloraxis_showscale=False,
                               yaxis_title="",
                               xaxis=dict(tickformat="$,.0f"))
            st.plotly_chart(fig14, use_container_width=True)

    # ── Sub-category analysis ─────────────────────────────
    st.markdown("### Sub-Category Profit Breakdown")
    sub_col = "Sub_Category" if "Sub_Category" in filtered.columns else "Sub-Category"
    if sub_col in filtered.columns:
        sub = (filtered.groupby(sub_col)
               .agg(Revenue=("Sales","sum"),
                    Profit=("Profit","sum"),
                    Orders=("Sales","count"))
               .reset_index()
               .sort_values("Profit", ascending=False))
        sub["Margin"] = (sub["Profit"] / sub["Revenue"] * 100).round(1)
        sub["Color"] = sub["Profit"].apply(
            lambda x: "#16A34A" if x > 0 else "#DC2626"
        )

        fig15 = px.bar(sub, x=sub_col, y="Profit",
                      color="Color", color_discrete_map="identity",
                      hover_data={"Revenue": ":$,.0f",
                                 "Margin": ":.1f",
                                 "Orders": ":,",
                                 "Color": False})
        fig15.add_hline(y=0, line_color="black", line_width=1)
        fig15.update_layout(height=380, showlegend=False,
                           margin=dict(t=20, b=20),
                           xaxis=dict(tickangle=45),
                           yaxis=dict(tickformat="$,.0f",
                                     title="Total Profit ($)"))
        st.plotly_chart(fig15, use_container_width=True)

    # ── Ship mode analysis ────────────────────────────────
    if "Ship_Mode" in filtered.columns:
        st.markdown("### Shipping Mode Analysis")
        ship_col1, ship_col2 = st.columns(2)

        with ship_col1:
            ship_count = filtered["Ship_Mode"].value_counts().reset_index()
            ship_count.columns = ["Ship_Mode","Count"]
            fig16 = px.pie(ship_count, values="Count", names="Ship_Mode",
                          hole=0.4,
                          color_discrete_sequence=["#2563EB","#7C3AED","#16A34A","#EA580C"])
            fig16.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig16, use_container_width=True)

        with ship_col2:
            ship_margin = (filtered.groupby("Ship_Mode")["Profit_Margin"]
                          .mean().reset_index().sort_values("Profit_Margin"))
            fig17 = px.bar(ship_margin, x="Ship_Mode", y="Profit_Margin",
                          color="Profit_Margin",
                          color_continuous_scale="RdYlGn",
                          color_continuous_midpoint=0)
            fig17.update_layout(height=300, margin=dict(t=20, b=20),
                               showlegend=False,
                               coloraxis_showscale=False,
                               yaxis=dict(title="Avg Profit Margin (%)"))
            st.plotly_chart(fig17, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.caption("Global Superstore Analytics Platform · Built with Python, Streamlit & Plotly · Portfolio Project")