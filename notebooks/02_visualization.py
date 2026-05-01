import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs("../outputs/charts", exist_ok=True)

print("=" * 60)
print("MODULE 2: Data Visualization")
print("=" * 60)

try:
    df = pd.read_csv("../outputs/clean_data.csv", parse_dates=["Order_Date"])
    print(f"\n  Loaded clean data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns available: {list(df.columns)}")
except FileNotFoundError:
    print("  ERROR: Run 01_setup_and_eda.py first to create clean_data.csv!")
    exit()

sns.set_theme(style="whitegrid", palette="muted")

# Custom color palette — professional, like you'd see in finance
COLORS = {
    "blue":"#2563EB",
    "green":"#16A34A",
    "red":"#DC2626",
    "orange":"#EA580C",
    "purple":"#7C3AED",
    "gray":"#6B7280",
    "teal":"#0D9488",
}

# Color list for multi-category charts
CAT_COLORS = [COLORS["blue"], COLORS["green"], COLORS["orange"],
                COLORS["purple"], COLORS["red"], COLORS["teal"]]

def save_chart(filename, title=""):
    path = f"../outputs/charts/{filename}"
    plt.tight_layout()        
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()                  
    print(f"    Saved: {path}")

cat_col = "Category" if "Category" in df.columns else None
sub_col = "Sub_Category" if "Sub_Category" in df.columns else (
            "Sub-Category" if "Sub-Category" in df.columns else None)
market_col = "Market" if "Market" in df.columns else None
region_col = "Region" if "Region" in df.columns else None
ship_col = "Ship_Mode" if "Ship_Mode" in df.columns else None

print(f"\n  Key columns found:")
print(f" Category : {cat_col}")
print(f" Sub-Cat :{sub_col}")
print(f" Market :{market_col}")
print(f" Region  : {region_col}")
print(f" Ship Mode : {ship_col}")

if "Month_Num" not in df.columns and "Order_Date" in df.columns:
    df["Month_Num"] = df["Order_Date"].dt.month
    print("    Added Month_Num column")

# Quarter (1–4) from Order_Date
if "Quarter" not in df.columns and "Order_Date" in df.columns:
    df["Quarter"] = df["Order_Date"].dt.quarter
    print("    Added Quarter column")

# Year from Order_Date
if "Year" not in df.columns and "Order_Date" in df.columns:
    df["Year"] = df["Order_Date"].dt.year
    print("    Added Year column")

# Shipping_Days = difference between Ship_Date and Order_Date
if "Shipping_Days" not in df.columns:
    if "Ship_Date" in df.columns and "Order_Date" in df.columns:
        df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
        df["Shipping_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
        print("    Added Shipping_Days column")

# Profit_Margin — safe version (avoids division by zero)
if "Profit_Margin" not in df.columns:
    df["Profit_Margin"] = df.apply(
        lambda row: (row["Profit"] / row["Sales"] * 100) if row["Sales"] != 0 else 0,
        axis=1
    )
    print("    Added Profit_Margin column")

print(f"\n  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

#1. Monthly Sales Trend (Line Chart)
print("\n[1] Monthly Sales Trend...")

# Group sales by month
monthly = (df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"]
            .sum()
            .reset_index())
monthly.columns = ["Month", "Sales"]

# Convert Period back to timestamp for plotting (matplotlib needs datetime)
monthly["Month_dt"] = monthly["Month"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 5)) 

# Plot the main line
ax.plot(monthly["Month_dt"], monthly["Sales"],
        color=COLORS["blue"], linewidth=2, label="Monthly Sales")

# Fill the area under the line (looks more professional)
ax.fill_between(monthly["Month_dt"], monthly["Sales"],
                alpha=0.1, color=COLORS["blue"])

monthly["Rolling_Avg"] = monthly["Sales"].rolling(3).mean()
ax.plot(monthly["Month_dt"], monthly["Rolling_Avg"],
        color=COLORS["orange"], linewidth=2, linestyle="--", label="3-Month Rolling Avg")

# Labels and formatting
ax.set_title("Monthly Sales Trend (2011-2014)", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Sales ($)", fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)
save_chart("01_monthly_sales_trend.png")

# 2. Category & Sub-Category Performance (Bar Charts)
print("\n[2] Sales by Category & Sub-Category...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if cat_col:
    cat_sales = df.groupby(cat_col)["Sales"].sum().sort_values(ascending=True)
    cat_sales.plot(kind="barh", ax=axes[0], color=COLORS["blue"], edgecolor="white")
    axes[0].set_title("Revenue by Category", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Total Sales ($)")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    # Add value labels on each bar
    for i, v in enumerate(cat_sales):
        axes[0].text(v + 5000, i, f"${v/1e6:.2f}M", va="center", fontsize=10)
else:
    axes[0].text(0.5, 0.5, "Category column not found", ha="center", va="center")

# Right panel: Sub-Category profit (red = loss, green = profit) 
if sub_col:
    sub_profit = df.groupby(sub_col)["Profit"].sum().sort_values()
    # Color each bar based on whether profit is positive or negative
    colors = [COLORS["red"] if x < 0 else COLORS["green"] for x in sub_profit]
    sub_profit.plot(kind="barh", ax=axes[1], color=colors, edgecolor="white")
    axes[1].set_title("Profit by Sub-Category\n(Red = Loss-Making)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Total Profit ($)")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[1].axvline(x=0, color="black", linewidth=0.8)  # Vertical line at zero
else:
    axes[1].text(0.5, 0.5, "Sub_Category column not found", ha="center", va="center")

plt.suptitle("Category Performance Analysis", fontsize=15, fontweight="bold", y=1.02)
save_chart("02_category_performance.png")

# 3. Discount vs Profit Margin Scatter Plot
print("\n[3] Discount vs Profit Margin Scatter Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

if "Discount" in df.columns and "Profit_Margin" in df.columns:
    # Clip extreme outliers for cleaner display (keep -200% to +200%)
    plot_df = df[(df["Profit_Margin"] > -200) & (df["Profit_Margin"] < 200)].copy()

    if cat_col:
        categories = plot_df[cat_col].unique()
        for i, cat in enumerate(categories):
            mask = plot_df[cat_col] == cat
            # Sample 500 points max to avoid overcrowding the chart
            sample = plot_df[mask].sample(min(500, mask.sum()), random_state=42)
            ax.scatter(sample["Discount"], sample["Profit_Margin"],
                       color=CAT_COLORS[i], alpha=0.5, s=30, label=cat)
    else:
        ax.scatter(plot_df["Discount"], plot_df["Profit_Margin"],
                   color=COLORS["blue"], alpha=0.4, s=30)

    z = np.polyfit(plot_df["Discount"], plot_df["Profit_Margin"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, plot_df["Discount"].max(), 100)
    ax.plot(x_line, p(x_line), color="black", linewidth=2,
            linestyle="--", label="Trend Line")

    # Horizontal line at 0 = break-even
    ax.axhline(y=0, color=COLORS["red"], linewidth=1.5, alpha=0.7, label="Break-even (0%)")

    ax.set_title("Discount Rate vs Profit Margin\n(Higher discounts destroy profit)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Discount Rate", fontsize=12)
    ax.set_ylabel("Profit Margin (%)", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate the key insight directly on the chart
    ax.annotate("Orders above 30% discount\nare typically unprofitable",
                xy=(0.35, -30), fontsize=10, color=COLORS["red"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor=COLORS["red"]))

save_chart("03_discount_vs_profit.png")

#4. Heatmap: Profit by Category x Region
print("\n[4] Heatmap: Profit by Category x Region...")

if cat_col and region_col:
    # pivot_table creates a 2D grid: rows = Region, columns = Category, values = Profit
    pivot = df.pivot_table(values="Profit", index=region_col,
                           columns=cat_col, aggfunc="sum")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot,
                annot=True,           
                fmt=".0f",           
                cmap="RdYlGn",        
                center=0,             
                linewidths=0.5,
                ax=ax,
                cbar_kws={"label": "Profit ($)"})
    ax.set_title("Profit Heatmap: Region x Category\n(Red = Loss, Green = Profit)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Region", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    save_chart("04_heatmap_region_category.png")
else:
    print("    Skipped — Category or Region column not found")

#5. Seasonality Analysis: Average Sales by Month & Quarterly Sales by Year
print("\n[5] Seasonality Analysis...")

if "Month_Num" in df.columns:
    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",  6:"Jun",
                   7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    monthly_avg = df.groupby("Month_Num")["Sales"].mean()
    # Map month numbers to names using the dictionary above
    monthly_avg.index = monthly_avg.index.map(month_names)
    grand_avg = monthly_avg.mean()

    # Orange = above average month, Blue = below average month
    colors_bar = [COLORS["orange"] if v > grand_avg else COLORS["blue"]
                  for v in monthly_avg]

    monthly_avg.plot(kind="bar", ax=axes[0], color=colors_bar, edgecolor="white")
    axes[0].axhline(grand_avg, color="red", linestyle="--", linewidth=1.5, label="Annual Avg")
    axes[0].set_title("Average Sales by Month\n(Orange = above average)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Avg Sales ($)")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Right panel: Quarterly sales by year
    if "Quarter" in df.columns and "Year" in df.columns:
        qtr_sales = df.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
        for yr in sorted(qtr_sales["Year"].unique()):
            subset = qtr_sales[qtr_sales["Year"] == yr]
            axes[1].plot(subset["Quarter"], subset["Sales"],
                         marker="o", label=str(yr), linewidth=2)
        axes[1].set_title("Quarterly Sales by Year", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Quarter")
        axes[1].set_ylabel("Sales ($)")
        axes[1].set_xticks([1, 2, 3, 4])
        axes[1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        axes[1].legend(title="Year")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    else:
        axes[1].text(0.5, 0.5, "Quarter/Year column not found", ha="center", va="center")

    plt.suptitle("Seasonality Analysis", fontsize=15, fontweight="bold")
    save_chart("05_seasonality.png")
else:
    print("    Skipped — Month_Num column not found")

#6. Shipping Analysis: Shipping Days Distribution & Profit Margin by Ship Mode
print("\n[6] Shipping Analysis...")

if ship_col and "Shipping_Days" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: Distribution of shipping days by ship mode ---
    ship_modes = df[ship_col].unique()
    for i, mode in enumerate(ship_modes):
        data = df[df[ship_col] == mode]["Shipping_Days"].dropna()
        axes[0].hist(data, bins=15, alpha=0.6, label=mode,
                     color=CAT_COLORS[i % len(CAT_COLORS)])
    axes[0].set_title("Shipping Days Distribution by Mode", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Shipping Days")
    axes[0].set_ylabel("Number of Orders")
    axes[0].legend()

    # --- Right panel: Average profit margin by shipping mode ---
    ship_profit = df.groupby(ship_col)["Profit_Margin"].mean().sort_values()
    colors_ship = [COLORS["red"] if v < 0 else COLORS["teal"] for v in ship_profit]
    ship_profit.plot(kind="barh", ax=axes[1], color=colors_ship, edgecolor="white")
    axes[1].set_title("Average Profit Margin by Ship Mode", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Avg Profit Margin (%)")
    axes[1].axvline(x=0, color="black", linewidth=0.8)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.suptitle("Shipping Analysis", fontsize=15, fontweight="bold")
    save_chart("06_shipping_analysis.png")
else:
    print("    Skipped — Ship_Mode or Shipping_Days column not found")

#7. Executive Summary Dashboard
print("\n[7] Executive Summary Dashboard...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Global Superstore — Executive Analytics Dashboard",
             fontsize=18, fontweight="bold", y=1.01)

# GridSpec lets us arrange charts in a custom grid layout
# 2 rows, 3 columns
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
ax1 = fig.add_subplot(gs[0, :2])   # Row 0, spans first 2 columns (wide chart)
ax2 = fig.add_subplot(gs[0, 2])    # Row 0, column 2 (narrow)
ax3 = fig.add_subplot(gs[1, 0])    # Row 1, column 0
ax4 = fig.add_subplot(gs[1, 1])    # Row 1, column 1
ax5 = fig.add_subplot(gs[1, 2])    # Row 1, column 2

# Panel 1: Sales trend line
ax1.plot(monthly["Month_dt"], monthly["Sales"], color=COLORS["blue"], linewidth=2)
ax1.fill_between(monthly["Month_dt"], monthly["Sales"], alpha=0.1, color=COLORS["blue"])
ax1.set_title("Monthly Sales Trend", fontweight="bold")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.grid(True, alpha=0.3)

#  Panel 2: Revenue share donut chart 
if cat_col:
    cat_s = df.groupby(cat_col)["Sales"].sum()
    wedges, texts, autotexts = ax2.pie(
        cat_s, labels=cat_s.index, autopct="%1.0f%%",
        colors=CAT_COLORS[:len(cat_s)], startangle=90,
        wedgeprops=dict(width=0.5)   # width < 1 creates the donut hole
    )
    ax2.set_title("Revenue Share by Category", fontweight="bold")

# Panel 3: Revenue by market
if market_col:
    mkt = df.groupby(market_col)["Sales"].sum().sort_values(ascending=True)
    mkt.plot(kind="barh", ax=ax3, color=COLORS["teal"], edgecolor="white")
    ax3.set_title("Revenue by Market", fontweight="bold")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

# Panel 4: Quarterly sales grouped by year
if "Quarter" in df.columns and "Year" in df.columns:
    qtr = df.groupby(["Year", "Quarter"])["Sales"].sum().unstack()
    qtr.plot(kind="bar", ax=ax4, colormap="Blues", edgecolor="white")
    ax4.set_title("Quarterly Sales by Year", fontweight="bold")
    ax4.set_xticklabels([str(y) for y in qtr.index], rotation=0)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax4.legend(title="Quarter", fontsize=8)

# Panel 5: Top 8 products by profit
if "Product_Name" in df.columns:
    top_prod = df.groupby("Product_Name")["Profit"].sum().nlargest(8)
    top_prod.plot(kind="barh", ax=ax5, color=COLORS["green"], edgecolor="white")
    ax5.set_title("Top 8 Products by Profit", fontweight="bold")
    ax5.set_xlabel("Profit ($)")
    # Truncate long product names so they fit neatly
    labels = [name[:25] + "..." if len(name) > 25 else name
              for name in top_prod.index]
    ax5.set_yticklabels(labels, fontsize=8)

save_chart("07_executive_dashboard.png")

# FINAL SUMMARY 
print("\n" + "=" * 60)
print("MODULE 2 COMPLETE — All charts saved to outputs/charts/")
print("  Charts created:")
charts = [
    "01_monthly_sales_trend.png",
    "02_category_performance.png",
    "03_discount_vs_profit.png",
    "04_heatmap_region_category.png",
    "05_seasonality.png",
    "06_shipping_analysis.png",
    "07_executive_dashboard.png",
]
for c in charts:
    print(f"    {c}")
print("\n  Run 03_forecasting.py next")
print("=" * 60)