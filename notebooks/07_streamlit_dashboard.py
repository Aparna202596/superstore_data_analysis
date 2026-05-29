import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Global Superstore Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────
# Makes KPI cards look professional
st.markdown("""
<style>
    .kpi-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #2563EB;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 12px; color: #6B7280; font-weight: 500; text-transform: uppercase; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #111827; }
    .kpi-delta { font-size: 13px; margin-top: 2px; }
    .kpi-up   { color: #16A34A; }
    .kpi-down { color: #DC2626; }
    .section-header { font-size: 18px; font-weight: 600; color: #111827;
                        margin: 24px 0 12px; padding-bottom: 6px;
                        border-bottom: 2px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING (cached so it only runs once) ────────────
@st.cache_data
def load_data():
    data_path = Path("../outputs/clean_data.csv")
    if not data_path.exists():
        return None, "Run notebooks/01_setup_and_eda.py first to create outputs/clean_data.csv"
    df = pd.read_csv(data_path, parse_dates=["Order_Date"])
    # Ensure derived columns exist
    if "Profit_Margin" not in df.columns:
        df["Profit_Margin"] = df["Profit"] / (df["Sales"] + 1e-9) * 100
    if "Year" not in df.columns:
        df["Year"] = df["Order_Date"].dt.year
    if "Month_Num" not in df.columns:
        df["Month_Num"] = df["Order_Date"].dt.month
    if "Quarter" not in df.columns:
        df["Quarter"] = df["Order_Date"].dt.quarter
    return df, None

df_raw, load_error = load_data()

# ── HEADER ────────────────────────────────────────────────
st.title("📊 Global Superstore Analytics Dashboard")
st.caption("Interactive portfolio project — 51,290 orders across 7 markets (2011–2014)")

if load_error:
    st.error(f"❌ {load_error}")
    st.stop()

# ── SIDEBAR FILTERS ───────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filters")

    # Market filter
    all_markets = sorted(df_raw["Market"].unique()) if "Market" in df_raw.columns else []
    selected_markets = st.multiselect(
        "Markets", all_markets,
        default=all_markets,
        help="Select one or more markets to filter the dashboard"
    )

    # Category filter
    all_cats = sorted(df_raw["Category"].unique()) if "Category" in df_raw.columns else []
    selected_cats = st.multiselect(
        "Categories", all_cats,
        default=all_cats,
    )

    # Year range filter
    all_years = sorted(df_raw["Year"].unique())
    year_range = st.select_slider(
        "Year Range",
        options=all_years,
        value=(min(all_years), max(all_years))
    )

    st.divider()
    st.header("🤖 LLM Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Optional. Get one free at platform.openai.com"
    )

    st.divider()
    st.caption("Built with Python, Streamlit & Plotly")
    st.caption("Notebooks: EDA → Viz → Forecast → ML → Pipeline")

# ── APPLY FILTERS ─────────────────────────────────────────
df = df_raw.copy()
if selected_markets:
    df = df[df["Market"].isin(selected_markets)]
if selected_cats:
    df = df[df["Category"].isin(selected_cats)]
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

if df.empty:
    st.warning("No data matches your filters. Adjust the sidebar selections.")
    st.stop()

# ── KPI CARDS ─────────────────────────────────────────────
st.markdown('<p class="section-header">Key Performance Indicators</p>', unsafe_allow_html=True)

total_rev = df["Sales"].sum()
total_profit = df["Profit"].sum()
avg_margin = df["Profit_Margin"].mean()
loss_pct = (df["Profit"] < 0).mean() * 100
avg_ship = df["Shipping_Days"].mean() if "Shipping_Days" in df.columns else 0

# YoY growth (compare last two years in filtered data)
yoy_growth = 0.0
if "Year" in df.columns and df["Year"].nunique() >= 2:
    yoy = df.groupby("Year")["Sales"].sum().sort_index()
    yoy_growth = (yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2] * 100

col1, col2, col3, col4, col5 = st.columns(5)

def kpi_card(col, label, value, delta=None, delta_label="", color="#2563EB"):
    """Render a styled KPI card."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        cls   = "kpi-up" if delta >= 0 else "kpi-down"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {abs(delta):.1f}% {delta_label}</div>'
    col.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

kpi_card(col1, "Total Revenue", f"${total_rev/1e6:.2f}M", yoy_growth, "YoY", "#2563EB")
kpi_card(col2, "Total Profit", f"${total_profit/1e6:.2f}M", None, "", "#16A34A")
kpi_card(col3, "Avg Margin", f"{avg_margin:.1f}%", None, "", "#7C3AED")
kpi_card(col4, "Loss-making %", f"{loss_pct:.1f}%", None, "", "#DC2626")
kpi_card(col5, "Avg Ship Days", f"{avg_ship:.1f}d", None, "", "#0D9488")

st.divider()

# ── CHART ROW 1: Sales Trend + Category Performance ───────
st.markdown('<p class="section-header">Sales & Category Analysis</p>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])

with c1:
    # Monthly sales trend with rolling average
    monthly = (df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"]
                .sum().reset_index())
    monthly.columns = ["Period", "Sales"]
    monthly["Date"] = monthly["Period"].dt.to_timestamp()
    monthly["Rolling"] = monthly["Sales"].rolling(3).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Sales"],
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        line=dict(color="#2563EB", width=2),
        name="Monthly Sales"
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Rolling"],
        line=dict(color="#EA580C", width=2, dash="dash"),
        name="3-Month Rolling Avg"
    ))
    fig_trend.update_layout(
        title="Monthly Sales Trend",
        xaxis_title="", yaxis_title="Sales ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", y=1.1),
        height=350, margin=dict(t=60, b=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig_trend, width="stretch")

with c2:
    # Category donut chart
    cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
    fig_donut = px.pie(
        cat_sales, values="Sales", names="Category",
        hole=0.5,
        color_discrete_sequence=["#2563EB", "#16A34A", "#EA580C"],
        title="Revenue Share by Category"
    )
    fig_donut.update_layout(height=350, margin=dict(t=60, b=20))
    st.plotly_chart(fig_donut, width="stretch")

# ── CHART ROW 2: Market Performance + Discount Impact ─────
st.markdown('<p class="section-header">Market Intelligence</p>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    # Market revenue + margin bubble chart
    mkt = df.groupby("Market").agg(
        Revenue=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Sales", "count")
    ).reset_index()
    mkt["Margin"] = (mkt["Profit"] / mkt["Revenue"] * 100).round(1)

    fig_mkt = px.bar(
        mkt.sort_values("Revenue"),
        x="Revenue", y="Market", orientation="h",
        color="Margin",
        color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"],
        color_continuous_midpoint=10,
        text=mkt.sort_values("Revenue")["Margin"].apply(lambda m: f"{m:.1f}%"),
        title= "Revenue by Market (Color = Profit Margin %)"
    )
    fig_mkt.update_traces(textposition="outside")
    fig_mkt.update_layout(
        height=380, xaxis_tickformat="$,.0f",
        margin=dict(t=60, b=20), coloraxis_showscale=False
    )
    st.plotly_chart(fig_mkt, width="stretch")

with c4:
    # Discount impact bar chart
    bands  = [0, 0.10, 0.20, 0.30, 0.50, 1.0]
    labels = ["0%", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
    margins, counts = [], []
    for i in range(len(bands) - 1):
        mask = (df["Discount"] >= bands[i]) & (df["Discount"] < bands[i+1])
        margins.append(round(float(df.loc[mask, "Profit_Margin"].mean()), 1))
        counts.append(int(mask.sum()))

    disc_df = pd.DataFrame({"Band": labels[:len(margins)], "Margin": margins, "Orders": counts})
    fig_disc = px.bar(
        disc_df, x="Band", y="Margin",
        color="Margin",
        color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"],
        color_continuous_midpoint=0,
        text=disc_df["Margin"].apply(lambda m: f"{m:.1f}%"),
        title="Avg Profit Margin by Discount Band"
    )
    fig_disc.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
    fig_disc.update_traces(textposition="outside")
    fig_disc.update_layout(
        height= 380, yaxis_title="Avg Margin (%)",
        margin= dict(t=60, b=20), coloraxis_showscale=False
    )
    st.plotly_chart(fig_disc, width="stretch")

# ── CHART ROW 3: Sub-category Profit + Heatmap ────────────
st.markdown('<p class="section-header">Granular Analysis</p>', unsafe_allow_html=True)
c5, c6 = st.columns(2)

with c5:
    sub_col = "Sub_Category" if "Sub_Category" in df.columns else None
    if sub_col:
        sub_profit = df.groupby(sub_col)["Profit"].sum().sort_values().reset_index()
        sub_profit.columns = ["Sub_Category", "Profit"]
        sub_profit["Color"] = sub_profit["Profit"].apply(lambda p: "#DC2626" if p < 0 else "#16A34A")
        fig_sub = px.bar(
            sub_profit, x="Profit", y="Sub_Category", orientation="h",
            color="Color", color_discrete_map="identity",
            title="Profit by Sub-Category (Red = Loss-Making)"
        )
        fig_sub.add_vline(x=0, line_dash="dot", line_color="black", opacity=0.5)
        fig_sub.update_layout(
            height=420, xaxis_tickformat="$,.0f",
            showlegend=False, margin=dict(t=60, b=20)
        )
        st.plotly_chart(fig_sub, width="stretch")

with c6:
    # Seasonal heatmap
    if "Month_Num" in df.columns and "Year" in df.columns:
        heat_df = df.groupby(["Year", "Month_Num"])["Sales"].sum().reset_index()
        heat_pivot = heat_df.pivot(index="Year", columns="Month_Num", values="Sales")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        heat_pivot.columns = [month_names[m-1] for m in heat_pivot.columns]

        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale="Blues",
            title="Monthly Sales Heatmap by Year",
            text_auto=".0s",
            aspect="auto"
        )
        fig_heat.update_layout(
            height=420, margin=dict(t=60, b=20),
            coloraxis_showscale=True
        )
        st.plotly_chart(fig_heat, width="stretch")

# ── SCATTER: Discount vs Margin ───────────────────────────
st.markdown('<p class="section-header">Discount vs Profit Margin</p>', unsafe_allow_html=True)
sample_df = df[(df["Profit_Margin"] > -200) & (df["Profit_Margin"] < 200)].sample(
    min(3000, len(df)), random_state=42
)
fig_scatter = px.scatter(
    sample_df, x="Discount", y="Profit_Margin",
    color="Category" if "Category" in sample_df.columns else None,
    color_discrete_sequence=["#2563EB", "#16A34A", "#EA580C"],
    opacity=0.4, size_max=6,
    trendline="ols",
    title="Discount Rate vs Profit Margin (sample of 3,000 orders)",
    labels={"Discount": "Discount Rate", "Profit_Margin": "Profit Margin (%)"}
)
fig_scatter.add_hline(y=0, line_dash="dot", line_color="red",
                        annotation_text="Break-even", annotation_position="right")
fig_scatter.update_layout(
    height=400, margin=dict(t=60, b=20),
    xaxis_tickformat=".0%"
)
st.plotly_chart(fig_scatter, width="stretch")

# ── SARIMA FORECAST SECTION ───────────────────────────────
st.markdown('<p class="section-header">Time-Series Forecast (SARIMA)</p>', unsafe_allow_html=True)

with st.expander("▶ Generate SARIMA Forecast (takes ~30 seconds)", expanded=False):
    if st.button("Run SARIMA Model", type="primary"):
        with st.spinner("Training SARIMA(1,1,1)(1,1,1,12) — please wait..."):
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                monthly_ts = (df_raw.groupby(df_raw["Order_Date"].dt.to_period("M"))["Sales"]
                                .sum().reset_index())
                monthly_ts.columns = ["Period", "Sales"]
                monthly_ts["Date"] = monthly_ts["Period"].dt.to_timestamp()
                monthly_ts = monthly_ts.sort_values("Date").set_index("Date")
                monthly_ts = monthly_ts.asfreq("MS")
                ts = monthly_ts["Sales"]

                train_size = int(len(ts) * 0.8)
                train, test = ts[:train_size], ts[train_size:]

                model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                                enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit(disp=False)
                fc = fitted.get_forecast(steps=len(test))
                pred = fc.predicted_mean
                ci   = fc.conf_int()

                # Future 3-month forecast
                final = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                                enforce_stationarity=False, enforce_invertibility=False)
                ffit = final.fit(disp=False)
                future = ffit.get_forecast(steps=3)
                fp  = future.predicted_mean
                fci = future.conf_int()

                rmse_val = float(np.sqrt(np.mean((test.values - pred.values)**2)))
                mape_val = float(np.mean(np.abs((test.values - pred.values) / (test.values + 1e-9))) * 100)

                # Build forecast chart
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=train.index, y=train.values,
                    line=dict(color="#6B7280", width=1.5), name="Training"))
                fig_fc.add_trace(go.Scatter(x=test.index, y=test.values,
                    line=dict(color="#2563EB", width=2), name="Actual (Test)"))
                fig_fc.add_trace(go.Scatter(x=pred.index, y=pred.values,
                    line=dict(color="#EA580C", width=2, dash="dash"), name="SARIMA Forecast"))
                fig_fc.add_trace(go.Scatter(
                    x=list(ci.index) + list(ci.index[::-1]),
                    y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                    fill="toself", fillcolor="rgba(234,88,12,0.1)",
                    line=dict(color="rgba(234,88,12,0)"), name="95% CI"
                ))
                fig_fc.add_trace(go.Scatter(x=fp.index, y=fp.values,
                    line=dict(color="#16A34A", width=2), mode="lines+markers",
                    marker=dict(size=8), name="Future Forecast"))
                fig_fc.update_layout(
                    title=f"SARIMA Forecast — RMSE: ${rmse_val:,.0f} | MAPE: {mape_val:.1f}%",
                    yaxis_tickformat="$,.0f", height=420,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_fc, width="stretch")

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("RMSE", f"${rmse_val:,.0f}")
                col_r2.metric("MAPE", f"{mape_val:.1f}%")
                col_r3.metric("Walk-forward R²",
                    f"{max(0, 1 - (rmse_val/ts.std())**2):.3f}")

                st.info(f"📅 Next 3-month forecast: " +
                    ", ".join([f"{d.strftime('%b %Y')}: ${v:,.0f}" for d, v in zip(fp.index, fp.values)]))

            except ImportError:
                st.error("Install statsmodels: `pip install statsmodels`")
            except Exception as e:
                st.error(f"Forecast failed: {e}")

# ── SHAP SECTION ──────────────────────────────────────────
st.markdown('<p class="section-header">SHAP Feature Importance</p>', unsafe_allow_html=True)

with st.expander("▶ Compute SHAP Explanations (takes ~20 seconds)", expanded=False):
    if st.button("Run SHAP Analysis", type="primary"):
        with st.spinner("Training XGBoost + computing SHAP values..."):
            try:
                import shap
                from sklearn.preprocessing import LabelEncoder
                import xgboost as xgb

                df_shap = df_raw.copy()
                if "Profit_Margin" not in df_shap.columns:
                    df_shap["Profit_Margin"] = df_shap["Profit"] / (df_shap["Sales"] + 1e-9) * 100

                df_shap["Margin_Class"] = pd.cut(df_shap["Profit_Margin"],
                    bins=[-999, 0, 10, 999], labels=["Low/Loss", "Medium", "High"])

                cat_cols = [c for c in ["Category","Sub_Category","Market","Region",
                                        "Segment","Ship_Mode","Order_Priority"] if c in df_shap.columns]
                num_cols = [c for c in ["Discount","Quantity","Shipping_Cost",
                                        "Sales","Year","Month_Num","Quarter"] if c in df_shap.columns]
                all_cols = cat_cols + num_cols

                df_ml = df_shap[all_cols + ["Margin_Class"]].dropna().copy()
                for col in cat_cols:
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

                le_y = LabelEncoder()
                y = le_y.fit_transform(df_ml["Margin_Class"].astype(str))
                X = df_ml[all_cols]

                sample_idx = np.random.choice(len(X), min(3000, len(X)), replace=False)
                X_s, y_s = X.iloc[sample_idx], y[sample_idx]

                model = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                                random_state=42, verbosity=0,
                                                eval_metric="mlogloss")
                model.fit(X_s, y_s)

                explainer = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X_s[:500])

                if isinstance(shap_vals, list):
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
                else:
                    mean_shap = np.abs(shap_vals).mean(axis=0)

                feat_imp = pd.Series(mean_shap, index=all_cols).sort_values(ascending=True)

                fig_shap = px.bar(
                    x=feat_imp.values, y=feat_imp.index, orientation="h",
                    color=feat_imp.values,
                    color_continuous_scale=["#E5E7EB", "#2563EB"],
                    title="SHAP Feature Importance — Profit Margin Classifier",
                    labels={"x": "Mean |SHAP Value|", "y": "Feature"}
                )
                fig_shap.update_layout(
                    height=420, margin=dict(t=60, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_shap, width="stretch")

                top_feat = feat_imp.index[-1]
                st.success(f"Top driver: **{top_feat}** (SHAP = {feat_imp.iloc[-1]:.4f}). "
                            f"Discount rate explains {feat_imp.iloc[-1]/feat_imp.sum()*100:.0f}% "
                            f"of model decisions.")

            except ImportError as e:
                missing = "shap" if "shap" in str(e) else "xgboost"
                st.error(f"Missing package: `pip install {missing}`")
            except Exception as e:
                st.error(f"SHAP failed: {e}")

# ── LLM INSIGHT SECTION ───────────────────────────────────
st.markdown('<p class="section-header">AI-Generated Insights</p>', unsafe_allow_html=True)

with st.expander("▶ Generate LLM Insight Report", expanded=False):
    if api_key:
        if st.button("Ask GPT-4o-mini for insights", type="primary"):
            with st.spinner("Calling OpenAI API..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)

                    kpi_summary = {
                        "revenue": int(total_rev),
                        "profit":  round(float(total_profit), 0),
                        "margin":  round(float(avg_margin), 1),
                        "loss_pct": round(float(loss_pct), 1),
                        "orders":  len(df),
                        "filter": {
                            "markets": selected_markets,
                            "categories": selected_cats,
                            "years": list(year_range)
                        }
                    }

                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "system",
                            "content": "You are a Senior Data Scientist presenting to a trading desk. Be concise and quantitative."
                        }, {
                            "role": "user",
                            "content": f"Analyze this filtered retail data and give 5 sharp insights:\n{json.dumps(kpi_summary, indent=2)}"
                        }],
                        max_tokens=400, temperature=0.2
                    )
                    insight = resp.choices[0].message.content
                    st.markdown(insight)
                    st.caption(f"*{resp.usage.total_tokens} tokens used*")

                except Exception as e:
                    st.error(f"LLM call failed: {e}")
    else:
        st.info("Enter your OpenAI API key in the sidebar to enable LLM insights.")
        st.code("# Or set in PowerShell:\n$env:OPENAI_API_KEY = 'sk-...'")

# ── FOOTER ────────────────────────────────────────────────
st.divider()
st.caption(
    f"Data: {len(df):,} orders after filters | "
    f"Revenue: ${total_rev/1e6:.2f}M | "
    f"Profit: ${total_profit/1e6:.2f}M | "
    f"Avg Margin: {avg_margin:.1f}%"
)
st.caption("Global Superstore Analytics Platform — Portfolio Project")