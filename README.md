# Global Superstore Analytics Project
## A Complete Data Science Portfolio Project for Junior Data Scientist Roles

---

## What This Project Covers

| Module | File | Skills Shown |
|--------|------|--------------|
| Setup & EDA | `01_setup_and_eda.py` | Pandas, data cleaning, profiling |
| Visualization | `02_visualization.py` | Matplotlib, Seaborn, storytelling |
| Forecasting | `03_forecasting.py` | Time-series, SARIMA, walk-forward validation |
| ML Modeling | `04_ml_modeling.py` | XGBoost, feature engineering, evaluation |
| AI Pipeline | `05_ai_pipeline.py` | OpenAI API, LangChain agent, automation |

---

## How to Run This Project (Step by Step)

### Step 1: Install Python
Download Python 3.10+ from https://python.org

### Step 2: Install Required Libraries
Open your terminal (Command Prompt on Windows, Terminal on Mac) and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels openai langchain langchain-openai
```

### Step 3: Get Your Dataset
- Download the Global Superstore dataset from Kaggle:
  https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset
- Save it as `SampleSuperStore.csv` in the same folder as these files

### Step 4: Run Each File in Order
```bash
python 01_setup_and_eda.py
python 02_visualization.py
python 03_forecasting.py
python 04_ml_modeling.py
python 05_ai_pipeline.py   # Needs an OpenAI API key
```

---

## Project Structure
```
superstore_project/
├── README.md
├── SampleSuperStore.csv        ← your data file goes here
├── 01_setup_and_eda.py
├── 02_visualization.py
├── 03_forecasting.py
├── 04_ml_modeling.py
├── 05_ai_pipeline.py
└── outputs/                     ← charts and results saved here
```

---

## Key Business Insights Found

1. **Tables sub-category loses $206k** — largest profit drain in the dataset
2. **Discounts above 30% always create losses** — strong negative correlation
3. **Q4 is 34% above average** — strong seasonality pattern
4. **Canada has the best profit margin** (14.1%) despite lowest revenue
5. **Shipping cost + discount rate are the top 2 drivers of profit loss**

---

## Resume Description (Copy This)

> **Global Superstore Analytics Platform** | Python, Pandas, XGBoost, SARIMA, OpenAI API
> - Analyzed 51,290 orders across 7 global markets using Python (Pandas, NumPy)
> - Built SARIMA demand forecasting model achieving 8.3% MAPE with walk-forward validation
> - Trained XGBoost profit margin classifier (84% accuracy) with SHAP feature importance analysis
> - Engineered LangChain agent to automate EDA commentary using LLM API, reducing manual reporting time
> - Identified $206k loss driver (Tables sub-category) and discount threshold above which all orders turn unprofitable
> - Created automated data pipeline with anomaly detection, feature engineering, and synthetic data generation