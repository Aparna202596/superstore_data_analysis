import pandas as pd
import numpy as np
import os

os.makedirs("../outputs", exist_ok=True)

# STEP 1: LOAD THE DATASET
print("=" * 60)
print("MODULE 1: Setup & Exploratory Data Analysis")
print("=" * 60)

print("\n[1] Loading dataset...")

possible_paths = [
    "../data/SampleSuperStore.csv",
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        print(f"    Found and loaded: {path}")
        break

if df is None:
    print("    ERROR: CSV file not found!")
    print("    Looked in these locations:")
    for p in possible_paths:
        print(f"      {p}")
    print("\n    Fix: copy your CSV into the data/ folder.")
    print("    Download: https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset")
    exit()

print(f"    Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# STEP 2: FIRST LOOK AT THE DATA
print("\n[2] First look at the data...")

print("--- First 5 rows ---")
print(df.head())

print("\n--- Column names (raw, before cleaning) ---")
print(list(df.columns))

print("\n--- Column types ---")
print(df.dtypes)

print("\n--- Summary statistics ---")
print(df.describe())

# STEP 3: DATA QUALITY CHECK
print("\n[3] Checking for data quality issues...")

missing = df.isnull().sum()
if missing.sum() == 0:
    print("    No missing values found!")
else:
    print("    Missing values:")
    print(missing[missing > 0])

dupes = df.duplicated().sum()
print(f"    Duplicate rows: {dupes}")

sales_col_raw = None
for col in df.columns:
    if col.lower().strip() in ["sales", "sale"]:
        sales_col_raw = col
        break
if sales_col_raw:
    neg = (df[sales_col_raw] < 0).sum()
    print(f"    Negative Sales values: {neg}")

# STEP 4: CLEAN AND STANDARDISE THE DATA
print("\n[4] Cleaning the data...")

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "_", regex=False)
    .str.replace("-", "_", regex=False)
)
print("    Standardised column names (spaces/dots -> underscores)")

# Attempt to parse Order_Date and Ship_Date with multiple formats
def safe_parse_dates(series):
    text = series.astype(str).str.strip()

    formats = [
        "%d/%m/%Y",    
        "%m/%d/%Y",    
        "%Y-%m-%d",    
        "%d-%m-%Y",    
        "%m-%d-%Y",    
        "%Y/%m/%d",    
        "%d %b %Y",    
    ]

    for fmt in formats:
        try:
            parsed = pd.to_datetime(text, format=fmt, errors="raise")
            print(f"      Parsed using format: {fmt}  (sample: {parsed.iloc[0]})")
            return parsed
        except (ValueError, TypeError):
            continue

    parsed = pd.to_datetime(text, errors="coerce")
    n_null = parsed.isnull().sum()
    if n_null < len(parsed) * 0.05:   
        print(f"      Parsed using auto-detect  ({n_null} failed)")
        return parsed

    print(f"      WARNING: date parsing failed — column stays as text")
    return series  

if "Order_Date" in df.columns:
    print("    Parsing Order_Date...")
    df["Order_Date"] = safe_parse_dates(df["Order_Date"])

# Apply to Ship_Date
if "Ship_Date" in df.columns:
    print("    Parsing Ship_Date...")
    df["Ship_Date"] = safe_parse_dates(df["Ship_Date"])

if "Order_Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Order_Date"]):
    df["Year"] = df["Order_Date"].dt.year
    df["Month_Num"] = df["Order_Date"].dt.month
    df["Quarter"] = df["Order_Date"].dt.quarter
    df["Week_Num"] = df["Order_Date"].dt.isocalendar().week.astype(int)
    df["Day_Name"] = df["Order_Date"].dt.day_name()
    print("    Extracted: Year, Month_Num, Quarter, Week_Num, Day_Name from Order_Date")
else:
    print("    WARNING: Order_Date not parsed — date features skipped")
    for col in ["Year", "Month_Num", "Quarter", "Week_Num"]:
        if col not in df.columns:
            df[col] = 0

if ("Order_Date" in df.columns and "Ship_Date" in df.columns and
        pd.api.types.is_datetime64_any_dtype(df["Order_Date"]) and
        pd.api.types.is_datetime64_any_dtype(df["Ship_Date"])):
    df["Shipping_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
    print("    Calculated: Shipping_Days (Ship_Date - Order_Date)")

if "Profit" in df.columns and "Sales" in df.columns:
    df["Profit_Margin"] = df.apply(
        lambda r: round((r["Profit"] / r["Sales"]) * 100, 2) if r["Sales"] != 0 else 0.0,
        axis=1
    )
    print("    Calculated: Profit_Margin (%)")

before = len(df)
df = df.drop_duplicates()
print(f"    Removed {before - len(df)} duplicate rows")
print(f"    Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

# STEP 5: KEY BUSINESS INSIGHTS
print("\n[5] Key Business Insights...")
print("-" * 50)

if "Sales" in df.columns and "Profit" in df.columns:
    total_rev    = df["Sales"].sum()
    total_profit = df["Profit"].sum()

    overall_pct  = (total_profit / total_rev * 100) if total_rev else 0

    print(f"\n  FINANCIAL SUMMARY")
    print(f"  Total Revenue  : ${total_rev:>12,.0f}")
    print(f"  Total Profit   : ${total_profit:>12,.0f}")
    print(f"  Overall Margin : {overall_pct:>11.1f}%")
    print(f"  Total Orders   : {len(df):>12,}")

# Performance by Category
if "Category" in df.columns:
    print(f"\n  CATEGORY PERFORMANCE")
    cat = df.groupby("Category").agg(
        Revenue=("Sales",  "sum"),
        Profit =("Profit", "sum"),
        Orders =("Sales",  "count"),
    )
    cat["Margin_%"] = (cat["Profit"] / cat["Revenue"] * 100).round(1)
    print(cat.to_string())

# Market Performance
if "Market" in df.columns:
    print(f"\n  MARKET PERFORMANCE")
    mkt = df.groupby("Market").agg(
        Revenue=("Sales",  "sum"),
        Profit =("Profit", "sum"),
    )
    mkt["Margin_%"] = (mkt["Profit"] / mkt["Revenue"] * 100).round(1)
    print(mkt.sort_values("Revenue", ascending=False).to_string())

# Discount Impact on Profit Margin
if "Discount" in df.columns and "Profit_Margin" in df.columns:
    print(f"\n  DISCOUNT IMPACT ON PROFIT MARGIN")
    bins = [-0.001, 0.001, 0.10, 0.20, 0.30, 0.50, 1.01]
    labels = ["None", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
    df["Discount_Band"] = pd.cut(df["Discount"], bins=bins, labels=labels)
    disc = df.groupby("Discount_Band", observed=True).agg(
        Avg_Margin = ("Profit_Margin", "mean"),
        Orders = ("Sales", "count"),
    ).round(1)
    for band, row in disc.iterrows():
        flag = "  <- LOSS" if row["Avg_Margin"] < 0 else ""
        print(f"  {str(band):<8}: {row['Avg_Margin']:>7.1f}%   ({int(row['Orders']):,} orders){flag}")

# Year-over-Year Growth
if "Year" in df.columns and df["Year"].nunique() > 1:
    print(f"\n  YEAR-OVER-YEAR SALES GROWTH")
    yoy = df.groupby("Year")["Sales"].sum().sort_index()
    prev = None
    for yr, sales in yoy.items():
        if prev is None:
            print(f"    {yr}: ${sales:>10,.0f}  (base year)")
        else:
            growth = (sales - prev) / prev * 100
            arrow = "+" if growth >= 0 else ""
            print(f"    {yr}: ${sales:>10,.0f}  ({arrow}{growth:.1f}% vs prior year)")
        prev = sales

# Top 5 Most Profitable and Loss-Making Products
if "Product_Name" in df.columns:
    print(f"\n  TOP 5 MOST PROFITABLE PRODUCTS")
    top5 = df.groupby("Product_Name")["Profit"].sum().nlargest(5)
    for name, profit in top5.items():
        print(f"    ${profit:>8,.0f}  {name[:55]}")

    print(f"\n  TOP 5 LOSS-MAKING PRODUCTS")
    bot5 = df.groupby("Product_Name")["Profit"].sum().nsmallest(5)
    for name, profit in bot5.items():
        print(f"    ${profit:>8,.0f}  {name[:55]}")

# STEP 6: SAVE CLEAN DATA
print("\n[6] Saving clean data...")

output_path = "../outputs/clean_data.csv"
df.to_csv(output_path, index=False)
print(f"    Saved to: {output_path}")
print(f"    Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\n  All columns in clean_data.csv:")
for col in df.columns:
    print(f"    {col:<25} ({df[col].dtype})")

print("\n" + "=" * 60)
print("MODULE 1 COMPLETE - Setup & EDA done!")
print("=" * 60)