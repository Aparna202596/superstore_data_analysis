import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

warnings.filterwarnings("ignore")

os.makedirs("../outputs/reports", exist_ok=True)
os.makedirs("../outputs/synthetic", exist_ok=True)
os.makedirs("../outputs/charts", exist_ok=True)

try:
    df = pd.read_csv("../outputs/clean_data.csv", parse_dates=["Order_Date"])
    print(f"Loaded: {df.shape[0]:,} rows")
except FileNotFoundError as e:
    raise FileNotFoundError("Missing required input file. Run 01_setup_and_eda.py first.") from e

def generate_eda_summary(data_df: pd.DataFrame) -> dict:
    summary = {
        "shape": {"rows": data_df.shape[0], "columns": data_df.shape[1]},
        "missing_values": int(data_df.isnull().sum().sum()),
        "duplicates": int(data_df.duplicated().sum()),
        "financials": {},
        "category_breakdowns": {},
        "anomaly_flags": []
    }

    fin_cols = ["Sales", "Profit", "Profit_Margin", "Discount", "Shipping_Cost"]
    for col in fin_cols:
        if col in data_df.columns:
            summary["financials"][col] = {
                "mean": round(float(data_df[col].mean()), 2),
                "median": round(float(data_df[col].median()), 2),
                "std": round(float(data_df[col].std()), 2),
                "min": round(float(data_df[col].min()), 2),
                "max": round(float(data_df[col].max()), 2),
                "pct_negative": round(float((data_df[col] < 0).mean() * 100), 1)
            }

    cat_cols = ["Category", "Market", "Segment", "Ship_Mode"]
    for col in cat_cols:
        if col in data_df.columns:
            top_sales = data_df.groupby(col)["Sales"].sum().sort_values(ascending=False)
            summary["category_breakdowns"][col] = {str(k): round(float(v), 0) for k, v in top_sales.to_dict().items()}

    if "Discount" in data_df.columns and "Profit_Margin" in data_df.columns:
        high_disc_loss = ((data_df["Discount"] > 0.3) & (data_df["Profit_Margin"] < 0)).sum()
        summary["anomaly_flags"].append(f"{high_disc_loss:,} orders: discount >30% AND loss-making")

    sub_col = "Sub_Category" if "Sub_Category" in data_df.columns else "Sub-Category"
    if sub_col in data_df.columns:
        losing_subs = data_df.groupby(sub_col)["Profit"].sum()
        for name, val in losing_subs[losing_subs < 0].sort_values().items():
            summary["anomaly_flags"].append(f"Sub-category '{name}' total loss: ${val:,.0f}")

    if "Order_Date" in data_df.columns:
        summary["date_range"] = {
            "start": str(data_df["Order_Date"].min().date()),
            "end": str(data_df["Order_Date"].max().date()),
            "years": int(data_df["Year"].nunique()) if "Year" in data_df.columns else None
        }
    return summary

eda_summary = generate_eda_summary(df)

with open("../outputs/reports/eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=2, default=str)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"DATA SUMMARY:\n{json.dumps(eda_summary, indent=2, default=str)}\n\nWrite an executive insight report with sections: Executive Summary (3 sentences), Key Findings (5 numerical bullets), Risk Flags (top 3 issues), Recommended Actions (3 concrete steps), and Model Readiness (suited ML approaches). Be concise and quantitative."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        report_text = response.choices[0].message.content
        
        with open("../outputs/reports/llm_insight_report.txt", "w") as f:
            f.write("GLOBAL SUPERSTORE — AUTO-GENERATED INSIGHT REPORT\n============================================================\n\n")
            f.write(report_text)
    except Exception as e:
        print(f"LLM generation bypassed: {e}")
else:
    print("Skipping LLM Report Generation: OPENAI_API_KEY not found in environment variables.")

def tool_top_products(n: int = 5) -> str:
    if "Product_Name" not in df.columns:
        return "No product data available."
    top = df.groupby("Product_Name")["Profit"].sum().nlargest(n)
    return f"Top {n} products by profit:\n" + "".join([f"  - {prod[:40]}: ${profit:,.0f}\n" for prod, profit in top.items()])

def tool_discount_analysis() -> str:
    if "Discount" not in df.columns:
        return "No discount data available."
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    result = "Discount impact on profit margin:\n"
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i+1]
        mask = (df["Discount"] >= lo) & (df["Discount"] < hi)
        avg_margin = df.loc[mask, "Profit_Margin"].mean() if "Profit_Margin" in df.columns else 0
        result += f"  Discount {lo:.0%}–{hi:.0%}: {mask.sum():,} orders, avg margin {avg_margin:.1f}%\n"
    return result

def tool_market_summary() -> str:
    if "Market" not in df.columns:
        return "No market data available."
    mkt = df.groupby("Market").agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Sales", "count")).sort_values("Revenue", ascending=False)
    result = "Market performance summary:\n"
    for market, row in mkt.iterrows():
        margin = (row["Profit"] / row["Revenue"] * 100) if row["Revenue"] > 0 else 0
        result += f"  {market}: ${row['Revenue']:,.0f} revenue, {margin:.1f}% margin, {row['Orders']:,} orders\n"
    return result

def tool_detect_anomalies() -> str:
    flags = []
    if "Discount" in df.columns and "Profit" in df.columns:
        flags.append(f"High-risk orders (discount>50% AND loss>$100): {len(df[(df['Discount'] > 0.5) & (df['Profit'] < -100)]):,}")
    if "Shipping_Days" in df.columns:
        flags.append(f"Slow shipping (>7 days): {len(df[df['Shipping_Days'] > 7]):,} orders")
    sub_col = "Sub_Category" if "Sub_Category" in df.columns else "Sub-Category"
    if sub_col in df.columns:
        loss_cats = df.groupby(sub_col)["Profit"].sum()
        flags.append(f"Loss-making sub-categories: {loss_cats[loss_cats < 0].index.tolist()}")
    return "Anomaly Detection Results:\n" + "\n".join(f"  • {f}" for f in flags)

def simple_agent(question: str) -> str:
    q_lower = question.lower()
    results = []
    if any(w in q_lower for w in ["product", "best", "top", "item"]):
        results.append(tool_top_products(5))
    if any(w in q_lower for w in ["discount", "price", "cut"]):
        results.append(tool_discount_analysis())
    if any(w in q_lower for w in ["market", "region", "country", "geographic"]):
        results.append(tool_market_summary())
    if any(w in q_lower for w in ["anomal", "unusual", "outlier", "problem", "risk"]):
        results.append(tool_detect_anomalies())
    
    if not results:
        margin_str = f"- Avg profit margin: {df['Profit_Margin'].mean():.1f}%" if 'Profit_Margin' in df.columns else ""
        results.append(f"No specific tool matched. Generic Summary:\n- {df.shape[0]:,} orders, ${df['Sales'].sum():,.0f} revenue\n"
                       f"- {df['Market'].nunique() if 'Market' in df.columns else 'N/A'} markets\n{margin_str}")
    return "\n\n".join(results)

questions = [
    "What are the top performing products and which markets have the best revenue?",
    "Are there any anomalies or unusual risks in the data I should know about?",
    "What is the impact of discounts on profit?"
]

agent_output = [{"question": q, "answer": simple_agent(q)} for q in questions]
agent_log = "\n".join([f"Q: {a['question']}\nA: {a['answer']}\n{'-'*40}" for a in agent_output])
with open("../outputs/reports/agent_qa_log.txt", "w", encoding="utf-8") as f:
    f.write("AUTOMATED AGENT Q&A LOG\n========================================\n\n" + agent_log)

def generate_synthetic_orders(source_df: pd.DataFrame, n: int = 1000, scenario: str = "normal") -> pd.DataFrame:
    np.random.seed(42)
    synthetic = pd.DataFrame()

    log_mean = np.log(source_df["Sales"].mean())
    synthetic["Sales"] = np.random.lognormal(log_mean, 0.8, n).astype(int).clip(10, 10000)

    if scenario == "high_discount":
        synthetic["Discount"] = np.random.uniform(0.3, 0.6, n).round(2)
    else:
        disc_probs = source_df["Discount"].value_counts(normalize=True)
        synthetic["Discount"] = np.random.choice(disc_probs.index, n, p=disc_probs.values)

    q_mean = source_df["Quantity"].mean() if "Quantity" in source_df.columns else 3
    q_std = source_df["Quantity"].std() if "Quantity" in source_df.columns else 2
    synthetic["Quantity"] = np.random.normal(q_mean, max(q_std, 0.1), n).clip(1, 20).round().astype(int)

    sc_mean = source_df["Shipping_Cost"].mean() if "Shipping_Cost" in source_df.columns else 20
    sc_std = source_df["Shipping_Cost"].std() if "Shipping_Cost" in source_df.columns else 15
    synthetic["Shipping_Cost"] = np.random.normal(sc_mean, max(sc_std, 0.1), n).clip(0).round(2)

    for col in ["Category", "Market", "Segment", "Ship_Mode", "Region"]:
        if col in source_df.columns:
            probs = source_df[col].value_counts(normalize=True)
            synthetic[col] = np.random.choice(probs.index, n, p=probs.values)

    if scenario == "peak_season":
        synthetic["Month_Num"] = np.random.choice([10, 11, 12], n, p=[0.2, 0.3, 0.5])
        synthetic["Quarter"] = 4
    else:
        if "Month_Num" in source_df.columns:
            m_probs = source_df["Month_Num"].value_counts(normalize=True).sort_index()
            synthetic["Month_Num"] = np.random.choice(m_probs.index, n, p=m_probs.values)
            synthetic["Quarter"] = synthetic["Month_Num"].apply(lambda m: (m - 1) // 3 + 1)

    base_margins = {"Technology": 18, "Furniture": 12, "Office Supplies": 15}
    synthetic["Profit_Margin"] = (synthetic["Category"].map(base_margins).fillna(14) - synthetic["Discount"] * 60 + np.random.normal(0, 3, n))
    synthetic["Profit"] = (synthetic["Sales"] * synthetic["Profit_Margin"] / 100).round(2)
    synthetic["Scenario"] = scenario

    return synthetic

syn_normal = generate_synthetic_orders(df, 1000, "normal")
syn_stress = generate_synthetic_orders(df, 500, "high_discount")
syn_seasonal = generate_synthetic_orders(df, 500, "peak_season")
all_synthetic = pd.concat([syn_normal, syn_stress, syn_seasonal], ignore_index=True)
all_synthetic.to_csv("../outputs/synthetic/synthetic_orders.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Real Data vs Synthetic Data — Distribution Comparison", fontsize=13, fontweight="bold")
for i, col in enumerate(["Sales", "Profit_Margin", "Discount"]):
    if col in df.columns and col in syn_normal.columns:
        real_data = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
        syn_data = syn_normal[col].clip(syn_normal[col].quantile(0.01), syn_normal[col].quantile(0.99))
        axes[i].hist(real_data, bins=40, alpha=0.6, color="#2563EB", label="Real", density=True)
        axes[i].hist(syn_data, bins=40, alpha=0.6, color="#EA580C", label="Synthetic", density=True)
        axes[i].set_title(col)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/charts/12_synthetic_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()

def run_pipeline(data_df: pd.DataFrame) -> dict:
    pipeline_log = []
    start = time.time()

    def log_step(step, message, status="OK"):
        elapsed = time.time() - start
        entry = {"step": step, "message": message, "status": status, "elapsed_sec": round(elapsed, 2)}
        pipeline_log.append(entry)
        return entry

    missing_pct = data_df.isnull().mean().max() * 100
    log_step(1, f"Validation complete. Missing Max: {missing_pct:.1f}%", "WARN" if missing_pct > 10 else "OK")

    kpis = {
        "total_revenue": round(data_df["Sales"].sum(), 0),
        "total_profit": round(data_df["Profit"].sum(), 0),
        "avg_margin": round(data_df["Profit_Margin"].mean(), 2) if "Profit_Margin" in data_df.columns else None,
        "orders": len(data_df),
        "loss_rate": round((data_df["Profit"] < 0).mean() * 100, 1)
    }
    log_step(2, f"KPIs computed. Revenue: ${kpis['total_revenue']:,.0f}")

    anomalies = []
    if "Discount" in data_df.columns and (data_df["Discount"] > 0.4).sum() > 100:
        anomalies.append(f"{(data_df['Discount'] > 0.4).sum():,} orders with discount >40%")
    log_step(3, f"Anomaly evaluation done. Issues found: {len(anomalies)}", "WARN" if anomalies else "OK")

    summary = generate_eda_summary(data_df)
    log_step(4, "EDA summary extraction completed")

    report = {"pipeline_run": True, "kpis": kpis, "anomalies": anomalies, "eda_summary": summary}
    with open("../outputs/reports/pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log_step(5, "Pipeline metrics exported successfully")

    return {"log": pipeline_log, "kpis": kpis, "anomalies": anomalies}

pipeline_result = run_pipeline(df)
print("Pipeline execution complete. Artifacts written to environment workspace partitions.")