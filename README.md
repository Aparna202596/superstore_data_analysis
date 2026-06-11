# Global Superstore Analytics Project

An end-to-end data analytics pipeline covering exploratory data analysis, visualisation, time-series forecasting, machine learning, AI-powered automation, SHAP explainability, and interactive dashboards — built on the Global Superstore dataset (2011–2014, 51,290 orders across 7 markets).

---

## What This Project Covers

| Module | File | Skills Demonstrated |
|--------|------|---------------------|
| Setup & EDA | `01_setup_and_eda.py` | Pandas, data cleaning, profiling, business KPIs |
| Visualisation | `02_visualization.py` | Matplotlib, Seaborn, chart storytelling |
| Forecasting | `03_forecasting.py` | Time-series analysis, SARIMA, walk-forward validation |
| ML Modelling | `04_ml_modeling.py` | XGBoost, feature engineering, cross-validation |
| AI Pipeline | `05_ai_pipeline.py` | OpenAI API, LangChain agent, synthetic data generation |
| LLM Insights | `06_llm_insights.py` | GPT-powered executive report generation |
| Streamlit Dashboard | `07_streamlit_dashboard.py` | Interactive web dashboard |
| Django Dashboard | `08_django_setup.py` | Role-based web application (Admin / Analyst / Viewer) |
| SHAP Explainability | `09_shap_explainability.py` | Model transparency, feature attribution |

---

## Project Structure

```
global_superstore_analysis/
├── README.md
├── data/
│   └── SampleSuperStore.csv
├── notebooks/
│   ├── 01_setup_and_eda.py
│   ├── 02_visualization.py
│   ├── 03_forecasting.py
│   ├── 04_ml_modeling.py
│   ├── 05_ai_pipeline.py
│   ├── 06_llm_insights.py
│   ├── 07_streamlit_dashboard.py
│   ├── 08_django_setup.py
│   └── 09_shap_explainability.py
└── outputs/
    ├── clean_data.csv
    ├── charts/
    │   ├── 01_monthly_sales_trend.png
    │   ├── 02_category_performance.png
    │   ├── 03_discount_vs_profit.png
    │   ├── 04_heatmap_region_category.png
    │   ├── 05_seasonality.png
    │   ├── 06_shipping_analysis.png
    │   ├── 07_executive_dashboard.png
    │   ├── 08_time_series_overview.png
    │   ├── 09_decomposition.png
    │   ├── 10_sarima_forecast.png
    │   ├── 11_ml_results.png
    │   ├── 12_synthetic_vs_real.png
    │   ├── 13_shap_summary.png
    │   ├── 14_shap_bar.png
    │   ├── 15_shap_waterfall.png
    │   ├── 16_shap_dependence.png
    │   └── 17_shap_heatmap.png
    ├── models/
    │   └── sarima_results.csv
    ├── reports/
    │   ├── eda_summary.json
    │   ├── llm_eda_summary.json
    │   ├── agent_qa_log.txt
    │   ├── pipeline_report.json
    │   └── llm_insight_report.md
    └── synthetic/
        └── synthetic_orders.csv
```

---

## How to Run This Project

### Step 1 — Install Python

Download Python 3.10+ from [https://python.org](https://python.org)

### Step 2 — Create a virtual environment

```powershell
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels \
            openai langchain langchain-openai streamlit plotly django shap
```

### Step 4 — Get the dataset

Download the Global Superstore dataset from Kaggle:
[https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset](https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset)

Save it as `SampleSuperStore.csv` inside the `data/` folder.

### Step 5 — Run each module in order

```bash
cd notebooks

python 01_setup_and_eda.py        # Clean data, EDA, business KPIs
python 02_visualization.py         # Generate all charts
python 03_forecasting.py           # SARIMA forecast
python 04_ml_modeling.py           # XGBoost profit classifier
python 05_ai_pipeline.py           # AI pipeline, LangChain agent, synthetic data
python 06_llm_insights.py          # LLM executive report (requires OpenAI key)
streamlit run 07_streamlit_dashboard.py   # Interactive dashboard → http://localhost:8501
python 08_django_setup.py          # Scaffold Django role-based dashboard
python 09_shap_explainability.py   # SHAP feature attribution charts
```

---

## Module Summaries & Key Outputs

### Module 1 — Setup & EDA (`01_setup_and_eda.py`)

Loads the raw CSV, standardises column names, parses dates, engineers features (shipping days, profit margin, discount bands), and surfaces key business KPIs.

**Dataset after cleaning:** 51,290 rows × 33 columns

**Financial summary:**

| Metric | Value |
|--------|-------|
| Total Revenue | $12,642,905 |
| Total Profit | $1,467,457 |
| Overall Margin | 11.6% |
| Total Orders | 51,290 |

**Year-over-year sales growth:**

| Year | Revenue | Growth |
|------|---------|--------|
| 2011 | $2,259,511 | Base year |
| 2012 | $2,677,493 | +18.5% |
| 2013 | $3,405,860 | +27.2% |
| 2014 | $4,300,041 | +26.3% |

**Discount impact on profit margin:**

| Discount Band | Avg Margin | Orders |
|---------------|------------|--------|
| 0% | +26.5% | 29,009 |
| 0–10% | +17.1% | 4,679 |
| 10–20% | +13.8% | 6,274 |
| 20–30% | −3.9% ⚠ | 967 |
| 30–50% | −33.8% ⚠ | 6,189 |
| 50%+ | −115.0% ⚠ | 4,172 |

**Output:** `outputs/clean_data.csv`

---

### Module 2 — Visualisation (`02_visualization.py`)

Generates seven publication-quality charts covering category performance, discount vs profit scatter, regional heatmaps, seasonality, shipping analysis, and an executive dashboard.

**Charts saved to `outputs/charts/`:**
- `01_monthly_sales_trend.png`
- `02_category_performance.png`
- `03_discount_vs_profit.png`
- `04_heatmap_region_category.png`
- `05_seasonality.png`
- `06_shipping_analysis.png`
- `07_executive_dashboard.png`

To regenerate charts after data or code changes, re-run `python 02_visualization.py`.

---

### Module 3 — Time-Series Forecasting (`03_forecasting.py`)

Builds a monthly sales time series (Jan 2011 – Dec 2014), decomposes it, and fits three models.

**Model comparison:**

| Model | RMSE | MAPE |
|-------|------|------|
| Moving Average (baseline) | $82,468 | 16.6% |
| Linear Regression | $106,566 | 25.0% |
| **SARIMA (walk-forward)** | **$44,093** | **8.2%** |

**3-month forecast (2015):**

| Month | Forecast |
|-------|---------|
| Jan 2015 | $350,878 |
| Feb 2015 | $306,276 |
| Mar 2015 | $350,220 |

Seasonal amplitude: $218,378 — peak month December, trough month February.

**Outputs:** `outputs/charts/08–10_*.png`, `outputs/models/sarima_results.csv`

---

### Module 4 — ML Modelling (`04_ml_modeling.py`)

Trains a profit-margin classifier (High / Medium / Low/Loss) using XGBoost with 5-fold cross-validation.

**Results:**

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Balanced Random Forest | 78.4% | 0.733 |
| **Tuned XGBoost** | **78.3%** | **0.741** |
| 5-fold CV (XGBoost) | 78.7% ± 0.2% | — |

**Top feature importances (SHAP weight):**

| Feature | Importance |
|---------|------------|
| Discount | 0.7136 |
| Market | 0.0456 |
| Sub_Category | 0.0393 |
| Category | 0.0354 |
| Region | 0.0243 |

**Business rule derived:** Hard-cap discounts at 30% — above this threshold the model consistently predicts Low/Loss outcomes.

**Output:** `outputs/charts/11_ml_results.png`

---

### Module 5 — AI Pipeline (`05_ai_pipeline.py`)

Five sub-components:

**Part A — Automated EDA Summary Generator**
Produces a structured JSON summary of dataset health, financials, category breakdowns, and anomaly flags. Saved to `outputs/reports/eda_summary.json`.

**Part B — LLM-Powered Insight Report**
Builds an executive prompt for GPT-4o-mini. Requires an OpenAI API key:
```powershell
$env:OPENAI_API_KEY = 'sk-...'
```
Without a key the prompt template is displayed for review.

**Part C — LangChain Agent**
A rule-based agent (upgradeable to a real LLM) answers business questions automatically:
- Top 5 products by profit: Canon imageCLASS 2200 Copier ($25,200), Cisco Smart Phone ($17,239), Motorola Smart Phone ($17,027)
- High-risk orders (discount >50% AND loss >$100): 912 orders
- Loss-making sub-categories: Tables
- Discount >40% cohort average margin: −28.9%

Saved to `outputs/reports/agent_qa_log.txt`.

**Part D — Synthetic Data Generation**
Generates 2,000 synthetic orders across three scenarios (normal, high-discount stress, peak season) for model stress-testing. Saved to `outputs/synthetic/synthetic_orders.csv`.

**Part E — Automated Pipeline (Simulated)**
End-to-end pipeline run with anomaly detection, KPI computation, and report generation in under 0.2 seconds.

---

### Module 6 — LLM Insights (`06_llm_insights.py`)

Builds a structured EDA summary and formats an executive prompt for GPT-4o-mini. Runs in demo mode without an API key and saves the prompt/report template.

**Outputs:** `outputs/reports/llm_eda_summary.json`, `outputs/reports/llm_insight_report.md`

**To enable live LLM generation:**
```powershell
$env:OPENAI_API_KEY = 'sk-...'
python 06_llm_insights.py
```

---

### Module 7 — Streamlit Dashboard (`07_streamlit_dashboard.py`)

Interactive web dashboard with filters, charts, and KPI cards.

```bash
streamlit run 07_streamlit_dashboard.py
```

Opens at [http://localhost:8501](http://localhost:8501)

---

### Module 8 — Django Dashboard (`08_django_setup.py`)

Scaffolds a role-based Django 5.x web application with three access levels.

**Role access:**

| Role | Access |
|------|--------|
| Admin | All pages + pipeline trigger + SHAP |
| Analyst | All pages + SHAP, no pipeline trigger |
| Viewer | Dashboard, Forecast, ML, Market only |

**After running the setup script:**
```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Admin panel: [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/)
Dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Assign roles via Admin → Auth → Users → Groups.

---

### Module 9 — SHAP Explainability (`09_shap_explainability.py`)

Retrains XGBoost on 12 features and computes SHAP values (TreeExplainer) on 2,000 samples across all three profit-margin classes.

**Top features by mean |SHAP|:**

| Feature | Mean |SHAP| |
|---------|---------------|
| Discount | 1.1665 |
| Market | 0.1762 |
| Sub_Category | 0.1329 |
| Sales | 0.0954 |
| Shipping_Cost | 0.0680 |

**SHAP business insights:**
1. **Dominant driver — Discount (1.1665):** Discount alone accounts for most of the model's classification power. Every percentage point above 30% sharply increases the probability of a Low/Loss outcome.
2. **Secondary drivers — Market & Sub_Category:** Geography and product mix provide secondary signal, confirming that pricing discipline is the primary lever.
3. **Policy recommendation:** A hard cap at 30% discount is supported by a sharp SHAP inflection point consistent across all markets and sub-categories.
4. **Model transparency:** Waterfall charts satisfy explainability requirements for AI-assisted pricing decisions.

**Charts saved to `outputs/charts/`:** `13_shap_summary.png`, `14_shap_bar.png`, `15_shap_waterfall.png`, `16_shap_dependence.png`, `17_shap_heatmap.png`

**Report:** `outputs/charts/shap_feature_summary.csv`

---

## Key Business Insights

| Finding | Detail |
|---------|--------|
| Discounts above 30% always create losses | Average margin drops to −33.8% at 30–50% discount; −115% above 50% |
| Tables sub-category is loss-making | Total loss: −$64,083 |
| 9,571 orders: discount >30% AND loss-making | Largest addressable profit leakage |
| Q4 drives peak revenue | December is the best seasonal month; February the worst |
| Canada has the highest profit margin | 26.6% margin despite the lowest absolute revenue ($66,932) |
| APAC is the largest market | $3.59M revenue, 12.2% margin, 11,002 orders |
| Strong YoY growth | Revenue grew from $2.26M (2011) to $4.30M (2014) — +90% over four years |
| SARIMA outperforms all baselines | 8.2% MAPE vs 16.6% for moving average |

---

## Optional: Enable LLM Features

Modules 5 and 6 call the OpenAI API for natural-language reports and agent-based Q&A. Without a key they run in demo mode and display the prompt templates.

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = 'sk-your-key-here'

# Mac / Linux
export OPENAI_API_KEY='sk-your-key-here'
```

Get a free key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
