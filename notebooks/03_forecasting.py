import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")   
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("=" * 60)
print("MODULE 3: Time-Series Forecasting")
print("=" * 60)

# LOAD DATA
try:
    df = pd.read_csv("../outputs/clean_data.csv", parse_dates=["Order_Date"])
    print(f"\n✓ Loaded: {df.shape[0]:,} rows")
except FileNotFoundError:
    print("✗ Run 01_setup_and_eda.py first!")
    exit()

print("\n[1] Building monthly time series...")

monthly = (df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"]
                .sum()
                .reset_index())
monthly.columns = ["Period", "Sales"]

# Convert period to timestamp (required for plotting and modeling)
monthly["Date"] = monthly["Period"].dt.to_timestamp()
monthly = monthly.set_index("Date").sort_index()

# Keep only the "Sales" column as our time series
ts = monthly["Sales"]

print(f"    Time range: {ts.index[0].strftime('%b %Y')} → {ts.index[-1].strftime('%b %Y')}")
print(f"    Months of data: {len(ts)}")
print(f"    Monthly avg: ${ts.mean():,.0f}")
print(f"    Min: ${ts.min():,.0f}  |  Max: ${ts.max():,.0f}")

# STEP 2: VISUALISE THE TIME SERIES

print("\n[2] Plotting raw time series...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Time Series Analysis", fontsize=15, fontweight="bold")

# --- Plot 1: Raw series ---
axes[0, 0].plot(ts.index, ts.values, color="#2563EB", linewidth=2)
axes[0, 0].set_title("Monthly Sales (Raw)")
axes[0, 0].set_ylabel("Sales ($)")
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0, 0].grid(True, alpha=0.3)

rolling_mean = ts.rolling(window=3).mean()
rolling_std = ts.rolling(window=3).std()
# --- Plot 2: Rolling mean ± std dev ---
axes[0, 1].plot(ts.index, ts.values, color="#6B7280", alpha=0.5, label="Actual")
axes[0, 1].plot(ts.index, rolling_mean, color="#2563EB", linewidth=2, label="3-Month Avg")
axes[0, 1].fill_between(ts.index,
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    alpha=0.15, color="#2563EB", label="±1 Std Dev")
axes[0, 1].set_title("Rolling Mean ± Std Dev")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# --- Plot 3: Month-over-month growth rate ---
growth = ts.pct_change() * 100   # pct_change() = percentage change vs previous
axes[1, 0].bar(ts.index, growth.values,
                color=["#16A34A" if g > 0 else "#DC2626" for g in growth.values],
                alpha=0.7)
axes[1, 0].axhline(y=0, color="black", linewidth=0.8)
axes[1, 0].set_title("Month-over-Month Growth Rate (%)")
axes[1, 0].set_ylabel("Growth (%)")
axes[1, 0].grid(True, alpha=0.3)

# --- Plot 4: Seasonal box plot ---
if "Month_Num" in df.columns:
    month_sales = df.groupby(["Year", "Month_Num"])["Sales"].sum().reset_index()
    month_sales_pivot = month_sales.pivot(index="Year", columns="Month_Num", values="Sales")
    axes[1, 1].boxplot([month_sales_pivot[m].dropna() for m in range(1, 13)],
                        tick_labels=["J","F","M","A","M","J","J","A","S","O","N","D"])
    axes[1, 1].set_title("Sales Distribution by Month")
    axes[1, 1].set_ylabel("Sales ($)")
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("../outputs/charts/08_time_series_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✓ Saved: outputs/charts/08_time_series_overview.png")

# STEP 3: DECOMPOSITION

print("\n[3] Decomposing the time series...")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose

    # period=12 because we have monthly data with annual seasonality
    decomp = seasonal_decompose(ts, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Time Series Decomposition (Trend + Seasonality + Noise)",
                    fontsize=14, fontweight="bold")

    axes[0].plot(decomp.observed, color="#2563EB"); axes[0].set_ylabel("Observed")
    axes[1].plot(decomp.trend, color="#EA580C"); axes[1].set_ylabel("Trend")
    axes[2].plot(decomp.seasonal, color="#7C3AED"); axes[2].set_ylabel("Seasonality")
    axes[3].plot(decomp.resid, color="#6B7280"); axes[3].set_ylabel("Residual")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig("../outputs/charts/09_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ Decomposition complete & saved")

    # Print seasonality insights
    seasonal_strength = decomp.seasonal.max() - decomp.seasonal.min()
    print(f"    Seasonal amplitude: ${seasonal_strength:,.0f}")
    best_month = decomp.seasonal.groupby(decomp.seasonal.index.month).mean().idxmax()
    worst_month = decomp.seasonal.groupby(decomp.seasonal.index.month).mean().idxmin()
    months = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"    Best seasonal month : {months[best_month]}")
    print(f"    Worst seasonal month: {months[worst_month]}")

except ImportError:
    print("    ⚠ statsmodels not installed. Run: pip install statsmodels")

# STEP 4: BASELINE FORECAST
print("\n[4] Baseline forecast (3-month simple moving average)...")

window = 3
train_size = int(len(ts) * 0.8)   
train = ts[:train_size]
test  = ts[train_size:]

# Moving average forecast: predict next value = avg of last N values
ma_predictions = []
ts_values = ts.values.copy()
for i in range(len(test)):
    pred = np.mean(ts_values[train_size + i - window : train_size + i])
    ma_predictions.append(pred)

ma_predictions = np.array(ma_predictions)
actuals = test.values

# Calculate errors
# RMSE = Root Mean Squared Error
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# MAPE = Mean Absolute Percentage Error
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100

baseline_rmse = rmse(actuals, ma_predictions)
baseline_mape = mape(actuals, ma_predictions)
print(f"    Baseline RMSE: ${baseline_rmse:,.0f}")
print(f"    Baseline MAPE: {baseline_mape:.1f}%")

# STEP 5: SARIMA MODEL

print("\n[5] SARIMA Forecasting Model...")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    print("    Training SARIMA(1,1,1)(1,1,1,12)...")
    model = SARIMAX(train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False)

    # Forecast on the test period
    forecast_obj = fitted.get_forecast(steps=len(test))
    sarima_pred  = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()  

    sarima_rmse = rmse(test.values, sarima_pred.values)
    sarima_mape = mape(test.values, sarima_pred.values)

    print(f"    SARIMA RMSE: ${sarima_rmse:,.0f}  (vs baseline ${baseline_rmse:,.0f})")
    print(f"    SARIMA MAPE: {sarima_mape:.1f}%  (vs baseline {baseline_mape:.1f}%)")
    improvement = (baseline_rmse - sarima_rmse) / baseline_rmse * 100
    print(f"    Improvement over baseline: {improvement:.1f}%")

    print("\n[6] Walk-forward validation...")

    wf_actual = []
    wf_predicted = []
    wf_start = train_size

    for i in range(len(test)):
        # Train on everything we know so far
        current_train = ts[:wf_start + i]
        try:
            wf_model = SARIMAX(current_train, order=(1,1,1),
                                seasonal_order=(1,1,1,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            wf_fit = wf_model.fit(disp=False)
            wf_pred = wf_fit.forecast(steps=1)
            wf_predicted.append(float(wf_pred.iloc[0]))
            wf_actual.append(float(ts.iloc[wf_start + i]))
        except Exception:
            # If model fails, fall back to moving average
            wf_predicted.append(float(ts.iloc[wf_start + i - 1]))
            wf_actual.append(float(ts.iloc[wf_start + i]))

    wf_rmse = rmse(np.array(wf_actual), np.array(wf_predicted))
    wf_mape = mape(np.array(wf_actual), np.array(wf_predicted))
    print(f"    Walk-forward RMSE: ${wf_rmse:,.0f}")
    print(f"    Walk-forward MAPE: {wf_mape:.1f}%")

    print("\n[7] Generating 3-month future forecast...")

    # Retrain on ALL data for final forecast
    final_model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                            enforce_stationarity=False, enforce_invertibility=False)
    final_fit = final_model.fit(disp=False)
    future_steps = 3
    future_forecast = final_fit.get_forecast(steps=future_steps)
    future_pred = future_forecast.predicted_mean
    future_ci   = future_forecast.conf_int()

    print(f"    Future forecast (next {future_steps} months):")
    for date, pred in future_pred.items():
        print(f"    {date.strftime('%b %Y')}: ${pred:,.0f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("SARIMA Forecasting Model — Results", fontsize=14, fontweight="bold")

    # --- Top: Full forecast view ---
    ax = axes[0]
    ax.plot(train.index, train.values, color="#6B7280", linewidth=1.5, label="Training Data")
    ax.plot(test.index,  test.values,  color="#2563EB", linewidth=2,   label="Actual (Test)")
    ax.plot(test.index,  sarima_pred,  color="#EA580C", linewidth=2,
            linestyle="--", label="SARIMA Forecast")
    ax.fill_between(test.index,
                    conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    alpha=0.15, color="#EA580C", label="95% Confidence Interval")
    ax.plot(future_pred.index, future_pred.values, color="#16A34A", linewidth=2,
            marker="o", label="Future Forecast")
    ax.fill_between(future_pred.index,
                    future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                    alpha=0.15, color="#16A34A")
    ax.axvline(test.index[0], color="black", linestyle=":", alpha=0.5, label="Train/Test Split")
    ax.set_title("Sales Forecast vs Actual")
    ax.set_ylabel("Sales ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add model performance text box
    textstr = f"SARIMA Performance\nRMSE: ${sarima_rmse:,.0f}\nMAPE: {sarima_mape:.1f}%\nR²: {1 - (sarima_rmse/ts.std())**2:.3f}"
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Bottom: Walk-forward residuals ---
    residuals = np.array(wf_actual) - np.array(wf_predicted)
    ax2 = axes[1]
    ax2.bar(test.index[:len(residuals)], residuals,
            color=["#16A34A" if r > 0 else "#DC2626" for r in residuals], alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_title("Walk-Forward Forecast Residuals (Actual − Predicted)")
    ax2.set_ylabel("Residual ($)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/charts/10_sarima_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓ Saved: outputs/charts/10_sarima_forecast.png")

    # Save results summary
    results = {
        "model": "SARIMA(1,1,1)(1,1,1,12)",
        "train_months": train_size,
        "test_months": len(test),
        "baseline_rmse": round(baseline_rmse, 2),
        "sarima_rmse": round(sarima_rmse, 2),
        "sarima_mape": round(sarima_mape, 2),
        "walkforward_rmse": round(wf_rmse, 2),
        "walkforward_mape": round(wf_mape, 2),
        "improvement_pct": round(improvement, 2)
    }
    pd.Series(results).to_csv("../outputs/models/sarima_results.csv", header=["Value"])
    print("    ✓ Saved: outputs/models/sarima_results.csv")

except ImportError:
    print("    ⚠ statsmodels not installed.")
    print("    Run: pip install statsmodels")

# STEP 9: SIMPLE LINEAR TREND MODEL (Backup / Extra)

print("\n[8] Linear trend model (benchmark comparison)...")

from sklearn.linear_model import LinearRegression

ts_arr = ts.values
X = np.arange(len(ts_arr)).reshape(-1, 1)   # -1 means "figure out the rows"
y = ts_arr

# Split
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_rmse = rmse(y_test, lr_pred)
lr_mape = mape(y_test, lr_pred)
print(f"    Linear Regression RMSE: ${lr_rmse:,.0f}")
print(f"    Linear Regression MAPE: {lr_mape:.1f}%")

# ── MODEL COMPARISON TABLE ────────────────────────────────
print("\n" + "─" * 55)
print(f"  MODEL COMPARISON")
print("─" * 55)
print(f"  {'Model':<30} {'RMSE':>10} {'MAPE':>8}")
print("─" * 55)
print(f"  {'Moving Avg (Baseline)':<30} ${baseline_rmse:>9,.0f} {baseline_mape:>7.1f}%")
print(f"  {'Linear Regression':<30} ${lr_rmse:>9,.0f} {lr_mape:>7.1f}%")
try:
    print(f"  {'SARIMA (walk-forward)':<30} ${wf_rmse:>9,.0f} {wf_mape:>7.1f}%")
except NameError:
    pass
print("─" * 55)

print("\n" + "=" * 60)
print("✓ MODULE 3 COMPLETE — Time-Series Forecasting Done!")
print("=" * 60)