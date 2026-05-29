import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────
# Put your key here as a fallback (or leave empty and use env var)
FALLBACK_KEY = ""
DATA_PATH = Path("../outputs/clean_data.csv")
REPORT_DIR = Path("../outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LLM-Powered Insight Report")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────
if not DATA_PATH.exists():
    print(f"\n✗ File not found: {DATA_PATH}")
    print("  Run notebooks/01_setup_and_eda.py first.")
    exit()

df = pd.read_csv(DATA_PATH, parse_dates=["Order_Date"])
print(f"\n✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── BUILD EDA SUMMARY ─────────────────────────────────────
# This is the structured data we send to the LLM.
# The LLM cannot see the raw CSV — we pre-compute everything
# and pass it as a concise JSON summary.

def build_summary(df: pd.DataFrame) -> dict:

    summary = {}

    # --- Revenue & profit ---
    summary["revenue"] = int(df["Sales"].sum())
    summary["profit"] = round(float(df["Profit"].sum()), 0)
    summary["avg_margin_pct"] = round(float(df["Profit_Margin"].mean()), 1)
    summary["total_orders"] = int(len(df))
    summary["loss_orders"] = int((df["Profit"] < 0).sum())
    summary["loss_pct"] = round((df["Profit"] < 0).mean() * 100, 1)

    # --- YoY growth ---
    if "Year" in df.columns:
        yoy = df.groupby("Year")["Sales"].sum().sort_index()
        if len(yoy) >= 2:
            g = (yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2] * 100
            summary["yoy_growth_pct"] = round(float(g), 1)
            summary["year_range"] = f"{int(yoy.index[0])}-{int(yoy.index[-1])}"

    # --- Category breakdown ---
    if "Category" in df.columns:
        cat = df.groupby("Category").agg(
            Revenue=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        cat["Margin"] = (cat["Profit"] / cat["Revenue"] * 100).round(1)
        summary["category_breakdown"] = {
            row: {"revenue": int(cat.loc[row, "Revenue"]),
                  "profit":  round(float(cat.loc[row, "Profit"]), 0),
                  "margin":  float(cat.loc[row, "Margin"])}
            for row in cat.index
        }

    # --- Market breakdown ---
    if "Market" in df.columns:
        mkt = df.groupby("Market").agg(
            Revenue=("Sales", "sum"),
            Profit=("Profit", "sum")
        ).sort_values("Revenue", ascending=False)
        mkt["Margin"] = (mkt["Profit"] / mkt["Revenue"] * 100).round(1)
        summary["market_breakdown"] = {
            m: {"revenue": int(mkt.loc[m, "Revenue"]),
                "margin":  float(mkt.loc[m, "Margin"])}
            for m in mkt.index
        }

    # --- Discount impact ---
    if "Discount" in df.columns and "Profit_Margin" in df.columns:
        bands  = [0, 0.10, 0.20, 0.30, 0.50, 1.0]
        labels = ["0%", "0-10%", "10-20%", "20-30%", "30-50%"]
        disc_impact = {}
        for i, lbl in enumerate(labels):
            mask = (df["Discount"] >= bands[i]) & (df["Discount"] < bands[i+1])
            disc_impact[lbl] = {
                "orders": int(mask.sum()),
                "avg_margin": round(float(df.loc[mask, "Profit_Margin"].mean()), 1)
            }
        summary["discount_impact"] = disc_impact

    # --- Loss-making sub-categories ---
    sub_col = "Sub_Category" if "Sub_Category" in df.columns else None
    if sub_col:
        sub = df.groupby(sub_col)["Profit"].sum()
        summary["loss_sub_categories"] = {
            str(k): round(float(v), 0)
            for k, v in sub[sub < 0].sort_values().items()
        }

    # --- Top profitable products ---
    if "Product_Name" in df.columns:
        top5 = df.groupby("Product_Name")["Profit"].sum().nlargest(5)
        summary["top_products"] = {str(k): round(float(v), 0) for k, v in top5.items()}

    # --- Shipping ---
    if "Shipping_Days" in df.columns:
        summary["avg_shipping_days"] = round(float(df["Shipping_Days"].mean()), 1)

    summary["date_range"] = {
        "start": str(df["Order_Date"].min().date()),
        "end":   str(df["Order_Date"].max().date()),
    }

    return summary


print("\n[1] Building EDA summary...")
summary = build_summary(df)

# Save the summary JSON so you can inspect it
summary_path = REPORT_DIR / "llm_eda_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"    ✓ Saved EDA summary: {summary_path}")


SYSTEM_PROMPT = """You are a Data and Quantitative Analyst presenting 
findings to a trading desk and senior management. Your analysis must be:
- Concise and quantitative (every claim backed by a number from the data)
- Structured with clear section headings
- Actionable (each insight has a recommended action)
- Written in professional business English
- No more than 400 words total"""

USER_PROMPT = f"""Analyze this Global Superstore retail dataset summary and write a 
professional executive report.

DATASET SUMMARY (2011-2014 global retail data, 51,290 orders):
{json.dumps(summary, indent=2)}

Write your report with exactly these 5 sections:

## EXECUTIVE SUMMARY
(2-3 sentences: what is this data, what is the headline finding)

## KEY FINDINGS
(5 bullet points, each with a specific number from the data)

## RISK FLAGS
(3 items that need immediate management attention, with quantified impact)

## RECOMMENDED ACTIONS
(3 concrete, prioritised actions with expected business impact)

## MODEL READINESS ASSESSMENT
(What ML models would work well on this data and why — 2-3 sentences)

Use $ signs for monetary values. Be direct. Avoid vague statements."""

print(f"\n[2] Prompt built ({len(USER_PROMPT)} characters)")

# ── API KEY RESOLUTION ────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY") or FALLBACK_KEY

if not api_key:
    print("\n" + "=" * 60)
    print("⚠  NO API KEY — Running in demo mode")
    print("=" * 60)
    print("\n  The prompt that would be sent to GPT-4o-mini:")
    print("\n  SYSTEM:", SYSTEM_PROMPT[:200] + "...")
    print("\n  USER (first 500 chars):", USER_PROMPT[:500] + "...")
    print("\n  To get a real LLM response:")
    print("    1. Visit https://platform.openai.com/api-keys")
    print("    2. Create an API key (free tier available)")
    print("    3. In PowerShell: $env:OPENAI_API_KEY = 'sk-...'")
    print("    4. Re-run this script")

    # Save a demo report so the file always exists
    demo_report = """# GLOBAL SUPERSTORE — EXECUTIVE REPORT (DEMO MODE)
## EXECUTIVE SUMMARY
Demo mode: set OPENAI_API_KEY to generate a real LLM report.
Revenue: ${:,} | Profit: ${:,.0f} | Avg Margin: {}%

## KEY FINDINGS
- Revenue totals ${:,} across 7 global markets (2011-2014)
- Overall profit margin is {}% with 24.5% of orders loss-making
- Discount above 30% correlates with -33.8% average profit margin
- Technology is the highest-margin category at 14.0%
- APAC leads revenue at $3.59M (12.2% margin)

## RISK FLAGS
- 9,571 orders with discount >30% are all loss-making
- Tables sub-category generates -$64,083 in cumulative losses
- 4,172 orders with 50%+ discounts carry -115% avg margin

## RECOMMENDED ACTIONS
1. Cap discount policy at 20% across all categories
2. Review or discontinue Tables sub-category pricing
3. Replicate Canada's 26.6% margin model in other markets

## MODEL READINESS ASSESSMENT
This dataset is suitable for XGBoost classification (profit margin prediction), 
SARIMA time-series forecasting (monthly demand), and customer segmentation clustering.
""".format(
        summary["revenue"], summary["profit"], summary["avg_margin_pct"],
        summary["revenue"], summary["avg_margin_pct"]
    )

    report_path = REPORT_DIR / "llm_insight_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(demo_report)
    print(f"\n  ✓ Demo report saved: {report_path}")

else:
    # ── REAL LLM CALL ─────────────────────────────────────
    print("\n[3] Calling OpenAI API...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",              # Cheap, fast, good enough
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT}
            ],
            max_tokens=800,
            temperature=0.2,                  # Low temp = more consistent output
        )

        report_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        print(f"    ✓ Response received ({tokens_used} tokens used)")
        print(f"\n{'─' * 60}")
        print(report_text)
        print(f"{'─' * 60}\n")

        # Save as markdown
        report_path = REPORT_DIR / "llm_insight_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# GLOBAL SUPERSTORE — LLM-GENERATED EXECUTIVE REPORT\n")
            f.write(f"*Generated by GPT-4o-mini | Tokens: {tokens_used}*\n\n")
            f.write(report_text)
        print(f"  ✓ Report saved: {report_path}")

        # Also save raw API response for debugging
        meta = {
            "model":   response.model,
            "tokens":  tokens_used,
            "summary_used": summary,
            "report":  report_text,
        }
        meta_path = REPORT_DIR / "llm_response_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Metadata saved: {meta_path}")

    except ImportError:
        print("\n  ✗ openai package not installed.")
        print("    Run: pip install openai")
    except Exception as e:
        print(f"\n  ✗ API call failed: {e}")
        print("    Common causes:")
        print("    - Invalid API key (check it starts with 'sk-')")
        print("    - No internet connection")
        print("    - API quota exceeded (check platform.openai.com/usage)")

print("\n" + "=" * 60)
print("TASK 1 COMPLETE")
print(f"  Output folder: outputs/reports/")
print("=" * 60)