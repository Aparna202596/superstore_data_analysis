import pandas as pd
import numpy as np
from pathlib import Path
import json, os, warnings
warnings.filterwarnings('ignore')

# CACHE

_cache = {}

# DATA LOADING
def get_data() -> pd.DataFrame:

    if 'df' not in _cache:
        candidates = [
            Path('../data/SampleSuperStore.csv'),
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path, parse_dates=['Order_Date'],
                                    low_memory=False)
                # Ensure key columns exist
                if 'Profit_Margin' not in df.columns and 'Profit' in df.columns:
                    df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 1e-9) * 100
                if 'Year' not in df.columns and 'Order_Date' in df.columns:
                    df['Year'] = df['Order_Date'].dt.year
                    df['Month_Num'] = df['Order_Date'].dt.month
                    df['Quarter'] = df['Order_Date'].dt.quarter
                _cache['df'] = df
                print(f"✓ Data loaded from {path}: {len(df):,} rows")
                break
        else:
            raise FileNotFoundError(
                "No data file found. Run 01_setup_and_eda.py first to create outputs/clean_data.csv"
            )
    return _cache['df'] 

# KPI FUNCTIONS

def get_kpis() -> dict:

    df = get_data()
    yoy = df.groupby('Year')['Sales'].sum()
    growth = float((yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2] * 100) if len(yoy) >= 2 else 0

    return {
        'total_revenue': int(df['Sales'].sum()),
        'total_profit': float(round(df['Profit'].sum(), 0)),
        'avg_margin': float(round(df['Profit_Margin'].mean(), 1)),
        'total_orders': int(len(df)),
        'yoy_growth': round(growth, 1),
        'loss_orders': int((df['Profit'] < 0).sum()),
        'loss_pct': round((df['Profit'] < 0).mean() * 100, 1),
        'avg_shipping_days': round(df['Shipping_Days'].mean(), 1) if 'Shipping_Days' in df.columns else 0,
    }

# CHART DATA FUNCTIONS
# Each returns a dict that gets converted to JSON for charts

def get_sales_trend() -> dict: 
    df = get_data()
    monthly = (df.groupby(df['Order_Date'].dt.to_period('M'))['Sales']
                    .sum().reset_index())
    monthly.columns = ['Period', 'Sales']
    monthly['Date'] = monthly['Period'].dt.to_timestamp()
    monthly = monthly.sort_values('Date')
    monthly['Rolling'] = monthly['Sales'].rolling(3).mean()

    return {
        'labels': [d.strftime('%b %Y') for d in monthly['Date']],
        'sales': monthly['Sales'].tolist(),
        'rolling': monthly['Rolling'].fillna(0).tolist(),
    }

# For category and market charts, we return both revenue and profit so we can toggle in the UI
def get_category_data() -> dict:
    df = get_data()
    cat = df.groupby('Category').agg(
        Revenue=('Sales', 'sum'),
        Profit=('Profit', 'sum')
    ).reset_index()

    sub_col = 'Sub_Category' if 'Sub_Category' in df.columns else 'Sub-Category'
    sub = df.groupby(sub_col)['Profit'].sum().sort_values().reset_index()
    sub.columns = ['name', 'profit']

    return {
        'categories':   cat['Category'].tolist(),
        'revenues':     cat['Revenue'].tolist(),
        'profits':      cat['Profit'].tolist(),
        'sub_names':    sub['name'].tolist(),
        'sub_profits':  [round(p, 0) for p in sub['profit'].tolist()],
    }

# Market data includes margin calculation and order counts
def get_market_data() -> dict:
    df = get_data()
    mkt = df.groupby('Market').agg(
        Revenue=('Sales', 'sum'),
        Profit=('Profit', 'sum'),
        Orders=('Sales', 'count')
    ).reset_index()
    mkt['Margin'] = (mkt['Profit'] / mkt['Revenue'] * 100).round(1)
    mkt = mkt.sort_values('Revenue', ascending=False)

    return {
        'markets':  mkt['Market'].tolist(),
        'revenues': [int(v) for v in mkt['Revenue'].tolist()],
        'profits':  [round(float(v), 0) for v in mkt['Profit'].tolist()],
        'margins':  mkt['Margin'].tolist(),
        'orders':   mkt['Orders'].tolist(),
    }

# Discount impact looks at average profit margin by discount band, and counts of orders in each band
def get_discount_impact() -> dict:
    df = get_data()
    bands = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    labels = ['0%', '0-10%', '10-20%', '20-30%', '30-40%', '40%+']
    margins, counts = [], []

    for i in range(len(bands) - 1):
        mask = (df['Discount'] >= bands[i]) & (df['Discount'] < bands[i+1])
        margins.append(round(float(df.loc[mask, 'Profit_Margin'].mean()), 1))
        counts.append(int(mask.sum()))

    return {'labels': labels, 'margins': margins, 'counts': counts}

# FORECASTING
def get_forecast_data() -> dict:
    if 'forecast' in _cache:
        return _cache['forecast']

    df = get_data()
    monthly = (df.groupby(df['Order_Date'].dt.to_period('M'))['Sales']
                    .sum().reset_index())
    monthly.columns = ['Period', 'Sales']
    monthly['Date'] = monthly['Period'].dt.to_timestamp()
    monthly = monthly.sort_values('Date').set_index('Date')
    ts = monthly['Sales']

    result = {
        'labels': [d.strftime('%b %Y') for d in ts.index],
        'actual': ts.tolist(),
        'forecast': [],
        'upper': [],
        'lower': [],
        'future_labels': [],
        'future_vals': [],
        'rmse': 0, 'mape': 0
    }

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]

        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        fc = fitted.get_forecast(steps=len(test))
        pred = fc.predicted_mean
        ci   = fc.conf_int()

        rmse = float(np.sqrt(np.mean((test.values - pred.values)**2)))
        mape = float(np.mean(np.abs((test.values - pred.values) / (test.values + 1e-9))) * 100)

        # Future forecast
        final = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        final_fit = final.fit(disp=False)
        future = final_fit.get_forecast(steps=3)
        fp = future.predicted_mean
        fci = future.conf_int()

        # Build arrays aligned with full timeline
        forecast_arr = [None] * train_size + pred.tolist()
        upper_arr = [None] * train_size + ci.iloc[:, 1].tolist()
        lower_arr = [None] * train_size + ci.iloc[:, 0].tolist()

        result.update({
            'forecast': forecast_arr,
            'upper': upper_arr,
            'lower': lower_arr,
            'future_labels': [d.strftime('%b %Y') for d in fp.index],
            'future_vals': [int(v) for v in fp.tolist()],
            'future_upper': [int(v) for v in fci.iloc[:, 1].tolist()],
            'future_lower': [int(v) for v in fci.iloc[:, 0].tolist()],
            'rmse': int(rmse), 'mape': round(mape, 1),
            'train_size': train_size,
        })
        _cache['forecast'] = result
    except Exception as e:
        result['error'] = str(e)

    return result

# ML & SHAP
def get_shap_data() -> dict:
    if 'shap' in _cache:
        return _cache['shap']

    df = get_data()
    try:
        import shap
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb

        if 'Profit_Margin' not in df.columns:
            df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 1e-9) * 100

        df['Margin_Class'] = pd.cut(df['Profit_Margin'],
            bins=[-999, 0, 10, 999], labels=['Low/Loss', 'Medium', 'High'])

        cat_cols = [c for c in ['Category','Sub_Category','Market','Region',
                                'Segment','Ship_Mode','Order_Priority'] if c in df.columns]
        num_cols = [c for c in ['Discount','Quantity','Shipping_Cost',
                                'Sales','Year','Month_Num','Quarter'] if c in df.columns]
        all_cols = cat_cols + num_cols

        df_ml = df[all_cols + ['Margin_Class']].dropna().copy()
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        le_y = LabelEncoder()
        y = le_y.fit_transform(df_ml['Margin_Class'].astype(str))
        X = df_ml[all_cols]

        sample = min(3000, len(X))
        idx = np.random.choice(len(X), sample, replace=False)
        X_s, y_s = X.iloc[idx], y[idx]

        model = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                    random_state=42, verbosity=0,
                                    eval_metric='mlogloss')
        model.fit(X_s, y_s)

        explainer = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_s[:500])

        # For multi-class, take mean absolute SHAP across all classes
        if isinstance(shap_vals, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_shap = np.abs(shap_vals).mean(axis=0)

        feat_imp = pd.Series(mean_shap, index=all_cols).sort_values(ascending=False)

        result = {
            'features': feat_imp.index.tolist(),
            'importance': [round(float(v), 4) for v in feat_imp.values.tolist()],
            'xgb_importance': {
                'features': all_cols,
                'values': [round(float(v), 4) for v in model.feature_importances_.tolist()]
            }
        }
        _cache['shap'] = result
        return result
    except ImportError:
        return {'error': 'shap not installed. Run: pip install shap'}
    except Exception as e:
        return {'error': str(e)}


def run_pipeline_job() -> dict:
    """Run the automated data pipeline and return a log."""
    import time
    df = get_data()
    log = []
    start = time.time()

    def step(n, msg, status='ok'):
        log.append({'step': n, 'msg': msg, 'status': status,
                    'time': round(time.time() - start, 2)})

    missing_pct = df.isnull().mean().max() * 100
    step(1, f'Data validation: {len(df):,} rows, {missing_pct:.1f}% max missing',
        'warn' if missing_pct > 20 else 'ok')

    step(2, f'KPIs: Revenue=${df["Sales"].sum():,.0f}, '
            f'Profit=${df["Profit"].sum():,.0f}')

    loss_orders = (df['Profit'] < 0).sum()
    step(3, f'Anomalies: {loss_orders:,} loss-making orders detected',
        'warn' if loss_orders > 1000 else 'ok')

    step(4, f'Feature engineering: {len(df.columns)} columns ready')

    # Invalidate forecast cache so it recalculates
    _cache.pop('forecast', None)
    step(5, 'Forecast cache cleared — will recalculate on next request')

    step(6, f'Pipeline complete in {time.time()-start:.1f}s')
    return {'log': log, 'timestamp': pd.Timestamp.now().isoformat()}

# LLM INSIGHT
def get_llm_insight(api_key: str = '') -> dict:
    if not api_key:
        return {'error': 'No API key provided',
                'demo': ('Demo mode: Your data shows $12.6M revenue across 7 markets. '
                            'Key risk: 9,571 orders with discount >30% are all loss-making. '
                            'Recommendation: cap discounts at 20% to protect margins.')}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        kpis = get_kpis()
        prompt = f"""You are a Senior Data Scientist presenting to a trading desk.
                    Analyze this retail data and give 5 sharp, quantified insights in bullet points.
                    Data: Revenue=${kpis['total_revenue']:,}, Profit=${kpis['total_profit']:,.0f},
                    Margin={kpis['avg_margin']}%, Orders={kpis['total_orders']:,},
                    Loss orders={kpis['loss_orders']:,} ({kpis['loss_pct']}% of total).
                    Be direct, use numbers, max 150 words total."""

        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=200, temperature=0.3)
        return {'insight': resp.choices[0].message.content}
    except Exception as e:
        return {'error': str(e)}