import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("../outputs/reports", exist_ok=True)
os.makedirs("../outputs/synthetic", exist_ok=True)

print("=" * 60)
print("MODULE 5: AI-Accelerated Data Pipeline")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────
try:
    df = pd.read_csv("../outputs/clean_data.csv", parse_dates=["Order_Date"])
    print(f"\n✓ Loaded: {df.shape[0]:,} rows")
except FileNotFoundError:
    print("✗ Run 01_setup_and_eda.py first!")
    exit()

print("\n─────────────────────────────────────────────────────")
print("PART A: Automated EDA Summary Generator")
print("─────────────────────────────────────────────────────")

def generate_eda_summary(df: pd.DataFrame) -> dict:

    summary = {}

    # --- Basic info ---
    summary["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    summary["missing_values"] = int(df.isnull().sum().sum())
    summary["duplicates"] = int(df.duplicated().sum())

    # --- Financial KPIs ---
    fin = {}
    for col in ["Sales", "Profit", "Profit_Margin", "Discount", "Shipping_Cost"]:
        if col in df.columns:
            fin[col] = {
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
                "std": round(float(df[col].std()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "pct_negative": round(float((df[col] < 0).mean() * 100), 1)
            }
    summary["financials"] = fin

    # --- Categorical breakdowns ---
    cats = {}
    for col in ["Category", "Market", "Segment", "Ship_Mode"]:
        if col in df.columns:
            top = df.groupby(col)["Sales"].sum().sort_values(ascending=False)
            cats[col] = {str(k): round(float(v), 0) for k, v in top.items()}
    summary["category_breakdowns"] = cats

    # --- Anomalies / flags ---
    flags = []
    if "Discount" in df.columns and "Profit_Margin" in df.columns:
        high_disc_loss = ((df["Discount"] > 0.3) & (df["Profit_Margin"] < 0)).sum()
        flags.append(f"{high_disc_loss:,} orders: discount >30% AND loss-making")

    sub_col = "Sub_Category" if "Sub_Category" in df.columns else "Sub-Category"
    if sub_col in df.columns:
        losing = df.groupby(sub_col)["Profit"].sum()
        losing = losing[losing < 0].sort_values()
        for name, val in losing.items():
            flags.append(f"Sub-category '{name}' total loss: ${val:,.0f}")

    summary["anomaly_flags"] = flags

    # --- Temporal ---
    if "Order_Date" in df.columns:
        summary["date_range"] = {
            "start": str(df["Order_Date"].min().date()),
            "end":   str(df["Order_Date"].max().date()),
            "years": int(df["Year"].nunique()) if "Year" in df.columns else None
        }

    return summary


eda_summary = generate_eda_summary(df)
print("\n✓ EDA summary generated:")
print(json.dumps(eda_summary, indent=2, default=str))

# Save the summary
with open("../outputs/reports/eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(eda_summary, f, indent=2, default=str)
print("\n✓ Saved: outputs/reports/eda_summary.json")


# ══════════════════════════════════════════════════════════
# PART B: LLM-POWERED INSIGHT REPORT (Requires API key)
# ══════════════════════════════════════════════════════════
print("\n─────────────────────────────────────────────────────")
print("PART B: LLM-Powered Insight Report")
print("─────────────────────────────────────────────────────")

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("\n⚠ No OPENAI_API_KEY found in environment variables.")
    print("  To enable LLM features:")
    print("  1. Get a free API key: https://platform.openai.com/api-keys")
    print("  2. Run in terminal: export OPENAI_API_KEY=sk-your-key-here")
    print("  3. Re-run this file")
    print("\n  Showing the prompt template you would use:\n")

    # Show the prompt we would send to the LLM
    EXAMPLE_PROMPT = f"""
I am a Data analyst presenting findings to a trading desk.
Analyze this dataset summary and written a professional executive report.

DATA SUMMARY:
{json.dumps(eda_summary, indent=2, default=str)}

Created my report with these sections:
1. EXECUTIVE SUMMARY (3 sentences max)
2. KEY FINDINGS (5 bullet points, quantified)
3. RISK FLAGS (items that need immediate attention)
4. RECOMMENDED ACTIONS (3 concrete steps)
5. MODEL READINESS (what ML models would work best on this data)

Be concise, quantitative, and direct. Use the actual numbers from the data.
"""
    print("  ─── EXAMPLE PROMPT ───")
    print(EXAMPLE_PROMPT[:800] + "...")
    print("  ─────────────────────────────────────────────────")
    llm_available = False
else:
    llm_available = True

if llm_available:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print("\n  Calling OpenAI GPT-4 for insight report...")

        prompt = f"""
I am a Data Analyst presenting findings to a trading desk.
Analyze this retail dataset summary and written a professional executive report.

DATA SUMMARY:
{json.dumps(eda_summary, indent=2, default=str)}

Created my report with:
1. EXECUTIVE SUMMARY (3 sentences)
2. KEY FINDINGS (5 bullet points with numbers)
3. RISK FLAGS (top 3 issues)
4. RECOMMENDED ACTIONS (3 concrete steps)
5. MODEL READINESS (what ML approaches suit this data)

Be concise, quantitative, and direct.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )

        report_text = response.choices[0].message.content
        print("\n  ─── LLM-GENERATED REPORT ───")
        print(report_text)
        print("  ─────────────────────────────")

        # Save the report
        with open("../outputs/reports/llm_insight_report.txt", "w", encoding="utf-8") as f:
            f.write("GLOBAL SUPERSTORE — AUTO-GENERATED INSIGHT REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_text)
        print("\n  ✓ Saved: outputs/reports/llm_insight_report.txt")

    except Exception as e:
        print(f"\n  ✗ LLM call failed: {e}")
        llm_available = False


# ══════════════════════════════════════════════════════════
# PART C: LANGCHAIN AGENT (Requires langchain + API key)
# ══════════════════════════════════════════════════════════

print("\n─────────────────────────────────────────────────────")
print("PART C: LangChain Agent for Automated Analysis")
print("─────────────────────────────────────────────────────")

def tool_top_products(n: int = 5) -> str:
    if "Product_Name" not in df.columns:
        return "No product data available."
    top = df.groupby("Product_Name")["Profit"].sum().nlargest(n)
    result = f"Top {n} products by profit:\n"
    for prod, profit in top.items():
        result += f"  - {prod[:40]}: ${profit:,.0f}\n"
    return result


def tool_discount_analysis() -> str:
    if "Discount" not in df.columns:
        return "No discount data."
    
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    result = "Discount impact on profit margin:\n"
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i+1]
        mask = (df["Discount"] >= lo) & (df["Discount"] < hi)
        avg_margin = df.loc[mask, "Profit_Margin"].mean() if "Profit_Margin" in df.columns else 0
        count = mask.sum()
        result += f"  Discount {lo:.0%}–{hi:.0%}: {count:,} orders, avg margin {avg_margin:.1f}%\n"
    return result


def tool_market_summary() -> str:
    if "Market" not in df.columns:
        return "No market data."
    mkt = df.groupby("Market").agg(
        Revenue=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Sales", "count")
    ).sort_values("Revenue", ascending=False)
    result = "Market performance summary:\n"
    for market, row in mkt.iterrows():
        margin = row["Profit"] / row["Revenue"] * 100 if row["Revenue"] > 0 else 0
        result += f"  {market}: ${row['Revenue']:,.0f} revenue, {margin:.1f}% margin, {row['Orders']:,} orders\n"
    return result


def tool_detect_anomalies() -> str:
    flags = []
    
    # Flag 1: Very high discounts with losses
    if "Discount" in df.columns and "Profit" in df.columns:
        risky = df[(df["Discount"] > 0.5) & (df["Profit"] < -100)]
        flags.append(f"High-risk orders (discount>50% AND loss>$100): {len(risky):,}")
    
    # Flag 2: Unusually long shipping
    if "Shipping_Days" in df.columns:
        long_ship = df[df["Shipping_Days"] > 7]
        flags.append(f"Slow shipping (>7 days): {len(long_ship):,} orders")
    
    # Flag 3: Loss-making categories
    sub_col = "Sub_Category" if "Sub_Category" in df.columns else "Sub-Category"
    if sub_col in df.columns:
        loss_cats = df.groupby(sub_col)["Profit"].sum()
        loss_cats = loss_cats[loss_cats < 0].index.tolist()
        flags.append(f"Loss-making sub-categories: {loss_cats}")
    
    return "Anomaly Detection Results:\n" + "\n".join(f"  • {f}" for f in flags)


# Simple agent without LangChain (works without API key)
def simple_agent(question: str) -> str:
    question_lower = question.lower()
    
    results = []
    
    if any(w in question_lower for w in ["product", "best", "top", "item"]):
        results.append(tool_top_products(5))
    
    if any(w in question_lower for w in ["discount", "price", "cut"]):
        results.append(tool_discount_analysis())
    
    if any(w in question_lower for w in ["market", "region", "country", "geographic"]):
        results.append(tool_market_summary())
    
    if any(w in question_lower for w in ["anomal", "unusual", "outlier", "problem", "risk"]):
        results.append(tool_detect_anomalies())
    
    if not results:
        results.append(f"No specific tool matched. Here is the EDA summary:\n"
                        f"- {df.shape[0]:,} orders, ${df['Sales'].sum():,.0f} revenue\n"
                        f"- {df['Market'].nunique() if 'Market' in df.columns else 'N/A'} markets\n"
                        f"- Avg profit margin: {df['Profit_Margin'].mean():.1f}%" if 'Profit_Margin' in df.columns else "")
    
    return "\n\n".join(results)


# ── Run the agent on example questions ────────────────────
print("\n  Running Agent on 3 sample questions...\n")

questions = [
    "What are the top performing products and which markets have the best revenue?",
    "Are there any anomalies or unusual risks in the data I should know about?",
    "What is the impact of discounts on profit?"
]

agent_output = []
for i, q in enumerate(questions, 1):
    print(f"  Q{i}: {q}")
    answer = simple_agent(q)
    print(f"  Agent response:\n{answer}\n")
    agent_output.append({"question": q, "answer": answer})

# Save agent Q&A
agent_log = "\n".join([f"Q: {a['question']}\nA: {a['answer']}\n" + "-"*40
                        for a in agent_output])
with open("../outputs/reports/agent_qa_log.txt", "w", encoding="utf-8") as f:
    f.write("AUTOMATED AGENT Q&A LOG\n" + "="*40 + "\n\n" + agent_log)
print("  ✓ Saved: ../outputs/reports/agent_qa_log.txt")

# LangChain version (shown as code pattern, runs if installed)
LANGCHAIN_EXAMPLE = '''
# ── LANGCHAIN AGENT (Full version with real LLM) ─────────
# This is what the production version looks like:

from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Wrap your functions as LangChain tools
@tool
def analyze_discounts(query: str) -> str:
    """Analyzes the impact of discounts on profit margins."""
    return tool_discount_analysis()

@tool  
def find_anomalies(query: str) -> str:
    """Detects anomalous orders and data quality issues."""
    return tool_detect_anomalies()

# Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize the agent with your tools
agent = initialize_agent(
    tools=[analyze_discounts, find_anomalies],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# The agent decides which tools to call automatically!
result = agent.run("Which sub-categories are losing money and why?")
print(result)
'''
print("\n  LangChain code pattern (runs when langchain is installed):")
print(LANGCHAIN_EXAMPLE)


# ══════════════════════════════════════════════════════════
# PART D: SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════

print("\n─────────────────────────────────────────────────────")
print("PART D: Synthetic Data Generation")
print("─────────────────────────────────────────────────────")

def generate_synthetic_orders(df: pd.DataFrame, n: int = 1000, 
                                scenario: str = "normal") -> pd.DataFrame:
    """
    Generate synthetic orders using statistical sampling.
    
    We learn the distribution of each column from real data,
    then sample from those distributions.
    
    scenario options:
        "normal"       — matches real data distribution
        "high_discount"— stress test: all orders have heavy discounts
        "peak_season"  — Q4 only, high volume
    """
    np.random.seed(42)
    synthetic = pd.DataFrame()

    # --- Sales: log-normal distribution (many small, few large) ---
    log_mean = np.log(df["Sales"].mean())
    log_std  = np.log(df["Sales"].std() + 1)
    synthetic["Sales"] = np.random.lognormal(log_mean, 0.8, n).astype(int)
    synthetic["Sales"] = synthetic["Sales"].clip(10, 10000)  # Realistic bounds

    # --- Discount ---
    if scenario == "high_discount":
        synthetic["Discount"] = np.random.uniform(0.3, 0.6, n).round(2)
    else:
        # Sample from real discount distribution
        disc_probs = df["Discount"].value_counts(normalize=True)
        synthetic["Discount"] = np.random.choice(disc_probs.index, n, p=disc_probs.values)

    # --- Quantity ---
    q_mean = df["Quantity"].mean() if "Quantity" in df.columns else 3
    q_std  = df["Quantity"].std()  if "Quantity" in df.columns else 2
    synthetic["Quantity"] = np.random.normal(q_mean, q_std, n).clip(1, 20).round().astype(int)

    # --- Shipping Cost ---
    sc_mean = df["Shipping_Cost"].mean() if "Shipping_Cost" in df.columns else 20
    sc_std  = df["Shipping_Cost"].std()  if "Shipping_Cost" in df.columns else 15
    synthetic["Shipping_Cost"] = np.random.normal(sc_mean, sc_std, n).clip(0).round(2)

    # --- Categorical columns (sample from real distribution) ---
    for col in ["Category", "Market", "Segment", "Ship_Mode", "Region"]:
        if col in df.columns:
            probs = df[col].value_counts(normalize=True)
            synthetic[col] = np.random.choice(probs.index, n, p=probs.values)

    # --- Time features ---
    if scenario == "peak_season":
        synthetic["Month_Num"] = np.random.choice([10, 11, 12], n, p=[0.2, 0.3, 0.5])
        synthetic["Quarter"] = 4
    else:
        if "Month_Num" in df.columns:
            m_probs = df["Month_Num"].value_counts(normalize=True).sort_index()
            synthetic["Month_Num"] = np.random.choice(m_probs.index, n, p=m_probs.values)
            synthetic["Quarter"]   = synthetic["Month_Num"].apply(lambda m: (m-1)//3 + 1)

    # --- Derive Profit from business rules ---
    # Base margin depends on category, then discount reduces it
    base_margins = {"Technology": 18, "Furniture": 12, "Office Supplies": 15}
    if "Category" in synthetic.columns:
        synthetic["Base_Margin"] = synthetic["Category"].map(base_margins).fillna(14)
    else:
        synthetic["Base_Margin"] = 14

    # Discount impact: each 10% discount ≈ 6pp margin reduction
    synthetic["Profit_Margin"] = (synthetic["Base_Margin"]
                                    - synthetic["Discount"] * 60
                                    + np.random.normal(0, 3, n))

    synthetic["Profit"] = (synthetic["Sales"] * synthetic["Profit_Margin"] / 100).round(2)

    synthetic["Scenario"] = scenario
    synthetic.drop(columns=["Base_Margin"], inplace=True)

    return synthetic


# Generate 3 scenarios
print("\n  Generating synthetic datasets...")

syn_normal   = generate_synthetic_orders(df, 1000, "normal")
syn_stress   = generate_synthetic_orders(df, 500,  "high_discount")
syn_seasonal = generate_synthetic_orders(df, 500,  "peak_season")

all_synthetic = pd.concat([syn_normal, syn_stress, syn_seasonal], ignore_index=True)

print(f"  ✓ Normal scenario      : {len(syn_normal):,} orders")
print(f"  ✓ High-discount stress : {len(syn_stress):,} orders  (avg discount: {syn_stress['Discount'].mean():.0%})")
print(f"  ✓ Peak season          : {len(syn_seasonal):,} orders")
print(f"  ✓ Total synthetic rows : {len(all_synthetic):,}")

# Compare real vs synthetic
print("\n  Real vs Synthetic comparison:")
for col in ["Sales", "Profit_Margin", "Discount"]:
    if col in df.columns and col in syn_normal.columns:
        real_mean = df[col].mean()
        syn_mean  = syn_normal[col].mean()
        print(f"  {col:<20}: Real={real_mean:>8.2f}  Synthetic={syn_mean:>8.2f}")

all_synthetic.to_csv("../outputs/synthetic/synthetic_orders.csv", index=False)
print("\n  ✓ Saved: outputs/synthetic/synthetic_orders.csv")

# Visualise synthetic vs real
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Real Data vs Synthetic Data — Distribution Comparison", fontsize=13, fontweight="bold")

for i, col in enumerate(["Sales", "Profit_Margin", "Discount"]):
    if col in df.columns and col in syn_normal.columns:
        # Clip for readability
        real_data = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
        syn_data  = syn_normal[col].clip(syn_normal[col].quantile(0.01), syn_normal[col].quantile(0.99))

        axes[i].hist(real_data, bins=40, alpha=0.6, color="#2563EB", label="Real", density=True)
        axes[i].hist(syn_data,  bins=40, alpha=0.6, color="#EA580C", label="Synthetic", density=True)
        axes[i].set_title(col)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../outputs/charts/12_synthetic_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/charts/12_synthetic_vs_real.png")


# ══════════════════════════════════════════════════════════
# PART E: PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════

print("\n─────────────────────────────────────────────────────")
print("PART E: Automated Pipeline (Simulated)")
print("─────────────────────────────────────────────────────")

import time

def run_pipeline(df: pd.DataFrame) -> dict:
    """
    Simulate an end-to-end automated data pipeline.
    
    Steps:
    1. Ingest new data
    2. Validate data quality
    3. Compute KPIs
    4. Detect anomalies
    5. Generate report
    
    In production this would be scheduled to run daily/weekly.
    """
    pipeline_log = []
    start = time.time()

    def log(step, message, status="OK"):
        elapsed = time.time() - start
        entry = {"step": step, "message": message,
                "status": status, "elapsed_sec": round(elapsed, 2)}
        pipeline_log.append(entry)
        icon = "✓" if status == "OK" else "✗" if status == "ERROR" else "⚠"
        print(f"  [{elapsed:>5.1f}s] {icon} Step {step}: {message}")
        return entry

    # Step 1: Validate
    missing_pct = df.isnull().mean().max() * 100
    if missing_pct > 10:
        log(1, f"High missing data: {missing_pct:.1f}%", "WARN")
    else:
        log(1, f"Data validation passed. Missing: {missing_pct:.1f}%")

    # Step 2: Compute KPIs
    kpis = {
        "total_revenue":  round(df["Sales"].sum(), 0),
        "total_profit":   round(df["Profit"].sum(), 0),
        "avg_margin":     round(df["Profit_Margin"].mean(), 2) if "Profit_Margin" in df.columns else None,
        "orders":         len(df),
        "loss_rate":      round((df["Profit"] < 0).mean() * 100, 1)
    }
    log(2, f"KPIs computed: Revenue=${kpis['total_revenue']:,.0f}, Margin={kpis['avg_margin']}%")

    # Step 3: Anomaly detection
    anomalies = []
    if "Discount" in df.columns:
        hi_disc = (df["Discount"] > 0.4).sum()
        if hi_disc > 100:
            anomalies.append(f"{hi_disc:,} orders with discount >40%")
    if anomalies:
        log(3, f"Anomalies detected: {len(anomalies)}", "WARN")
    else:
        log(3, "No anomalies detected")

    # Step 4: Auto-generate summary
    summary = generate_eda_summary(df)
    log(4, f"EDA summary generated ({len(str(summary))} chars)")

    # Step 5: Save report
    report = {"pipeline_run": True, "kpis": kpis,
                "anomalies": anomalies, "eda_summary": summary}
    with open("../outputs/reports/pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log(5, "Pipeline report saved to outputs/reports/pipeline_report.json")

    total_time = time.time() - start
    print(f"\n  Pipeline completed in {total_time:.1f}s")
    return {"log": pipeline_log, "kpis": kpis, "anomalies": anomalies}


print("\n  Running automated pipeline...\n")
pipeline_result = run_pipeline(df)

print("\n" + "=" * 60)
print("✓ MODULE 5 COMPLETE")
print()
print("  Files created:")
print("  outputs/reports/eda_summary.json")
print("  outputs/reports/agent_qa_log.txt")
print("  outputs/reports/pipeline_report.json")
print("  outputs/synthetic/synthetic_orders.csv")
print("  outputs/charts/12_synthetic_vs_real.png")
print()
print("=" * 60)
