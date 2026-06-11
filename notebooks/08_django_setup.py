import os
import sys
from pathlib import Path
import django
ROOT = Path(__file__).parent
print("=" * 60)
print("TASK 3: Django Dashboard Setup")
print(f"Project root: {ROOT}")
print("=" * 60)
# ── Check Django is installed ─────────────────────────────
try:
    print(f"✓ Django {django.__version__} found")
except ImportError:
    print("✗ Django not installed. Run: pip install django")
    sys.exit(1)
# ── Create all folders ────────────────────────────────────
folders = [
    "dashboard",
    "analytics",
    "templates/analytics",
    "templates/registration",
    "static/css",
    "static/js",
]
for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {folder}/")
# ══════════════════════════════════════════════════════════
# WRITE ALL FILES
# ══════════════════════════════════════════════════════════
def write(path: str, content: str):
    """Write a file, creating parent dirs if needed."""
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"  ✓ Created: {path}")
# ── manage.py ─────────────────────────────────────────────
write("manage.py", '''#!/usr/bin/env python
import os, sys
def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Couldn't import Django.") from exc
    execute_from_command_line(sys.argv)
if __name__ == "__main__":
    main()
''')
# ── dashboard/__init__.py ─────────────────────────────────
write("dashboard/__init__.py", "")
# ── dashboard/settings.py ─────────────────────────────────
write("dashboard/settings.py", '''
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = "django-insecure-superstore-dev-key-change-in-production-2024"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "analytics",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "dashboard.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {
        "context_processors": [
            "django.template.context_processors.debug",
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
        ],
    },
}]
WSGI_APPLICATION = "dashboard.wsgi.application"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
''')
# ── dashboard/urls.py ─────────────────────────────────────
write("dashboard/urls.py", '''
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("login/", auth_views.LoginView.as_view(
        template_name="registration/login.html"
    ), name="login"),
    
    # FIXED: next_page passed correctly to as_view()
    path("logout/", auth_views.LogoutView.as_view(
        template_name="registration/logged_out.html",
        next_page="/login/"
    ), name="logout"),
    
    path("", include("analytics.urls")),
]
''')
# ── dashboard/wsgi.py ─────────────────────────────────────
write("dashboard/wsgi.py", '''
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
application = get_wsgi_application()
''')
# ── analytics/__init__.py ─────────────────────────────────
write("analytics/__init__.py", "")
# ── analytics/apps.py ─────────────────────────────────────
write("analytics/apps.py", '''
from django.apps import AppConfig
class AnalyticsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "analytics"
''')
# ── analytics/models.py ───────────────────────────────────
# Role-based access uses Django Groups: Admin, Analyst, Viewer
write("analytics/models.py", '''
# Role-based access control uses Django\'s built-in Group system.
# Groups are created automatically in analytics/migrations/0001_create_groups.py
#
# Roles:
#   Admin   — can see everything + run pipeline
#   Analyst — can see all charts + trigger models
#   Viewer  — read-only: KPIs and charts only, no pipeline/SHAP
''')
# ── analytics/migrations/__init__.py ──────────────────────
write("analytics/migrations/__init__.py", "")
# ── analytics/migrations/0001_create_groups.py ────────────
# FIX: this migration creates auth.Group rows, so it MUST depend on the
# auth app's initial migration. With dependencies = [], Django cannot
# guarantee the auth_group table exists yet when this runs on a fresh
# database, which can raise "no such table: auth_group".
write("analytics/migrations/0001_create_groups.py", '''
from django.db import migrations
def create_groups(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    for name in ["Admin", "Analyst", "Viewer"]:
        Group.objects.get_or_create(name=name)
def delete_groups(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    Group.objects.filter(name__in=["Admin", "Analyst", "Viewer"]).delete()
class Migration(migrations.Migration):
    dependencies = [
        ("auth", "__first__"),
    ]
    operations = [migrations.RunPython(create_groups, delete_groups)]
''')
# ── analytics/decorators.py ───────────────────────────────
write("analytics/decorators.py", '''
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from functools import wraps
def role_required(*roles):
    """
    Decorator: user must be logged in AND belong to one of the given groups.
    Usage: @role_required("Admin", "Analyst")
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped(request, *args, **kwargs):
            user_groups = set(request.user.groups.values_list("name", flat=True))
            # Superusers always pass
            if request.user.is_superuser or user_groups.intersection(set(roles)):
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden(
                "<h2>Access Denied</h2>"
                f"<p>This page requires one of these roles: {', '.join(roles)}.</p>"
                "<p><a href=\'/\'>Back to dashboard</a></p>"
            )
        return _wrapped
    return decorator
''')
# ── analytics/urls.py ─────────────────────────────────────
write("analytics/urls.py", '''
from django.urls import path
from . import views
urlpatterns = [
    # Pages
    path("",          views.dashboard,  name="dashboard"),
    path("forecast/", views.forecast,   name="forecast"),
    path("ml/",       views.ml_model,   name="ml"),
    path("market/",   views.market,     name="market"),
    path("pipeline/", views.pipeline,   name="pipeline"),
    path("shap/",     views.shap_view,  name="shap"),
    # JSON API endpoints
    path("api/kpis/",         views.api_kpis,         name="api_kpis"),
    path("api/sales-trend/",  views.api_sales_trend,  name="api_sales_trend"),
    path("api/category/",     views.api_category,     name="api_category"),
    path("api/market/",       views.api_market_data,  name="api_market"),
    path("api/discount/",     views.api_discount,     name="api_discount"),
    path("api/forecast/",     views.api_forecast_data,name="api_forecast"),
    path("api/shap/",         views.api_shap,         name="api_shap"),
    path("api/llm-insight/",  views.api_llm_insight,  name="api_llm"),
    path("api/run-pipeline/", views.api_run_pipeline, name="api_pipeline"),
]
''')
# ── analytics/views.py ────────────────────────────────────
write("analytics/views.py", '''
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.contrib.auth.decorators import login_required
from .decorators import role_required
from . import data_service
import json
# ── PAGE VIEWS ────────────────────────────────────────────
@login_required
def dashboard(request):
    try:
        kpis = data_service.get_kpis()
    except Exception as e:
        kpis = {"error": str(e)}
    role = _get_role(request)
    return render(request, "analytics/dashboard.html", {"kpis": kpis, "role": role})
@login_required
def forecast(request):
    return render(request, "analytics/forecast.html", {"role": _get_role(request)})
@login_required
def ml_model(request):
    return render(request, "analytics/ml.html", {"role": _get_role(request)})
@login_required
def market(request):
    return render(request, "analytics/market.html", {"role": _get_role(request)})
@role_required("Admin", "Analyst")
def pipeline(request):
    """Only Admin and Analyst roles can access the pipeline page."""
    return render(request, "analytics/pipeline.html", {"role": _get_role(request)})
@role_required("Admin", "Analyst")
def shap_view(request):
    """Only Admin and Analyst roles can run SHAP analysis."""
    return render(request, "analytics/shap.html", {"role": _get_role(request)})
# ── API VIEWS ─────────────────────────────────────────────
@login_required
def api_kpis(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_kpis()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@login_required
def api_sales_trend(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_sales_trend()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@login_required
def api_category(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_category_data()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@login_required
def api_market_data(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_market_data()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@login_required
def api_discount(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_discount_impact()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@login_required
def api_forecast_data(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_forecast_data()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@role_required("Admin", "Analyst")
def api_shap(request):
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_shap_data()})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@login_required
def api_llm_insight(request):
    api_key = request.GET.get("key", "") or getattr(settings, "OPENAI_API_KEY", "")
    try:
        return JsonResponse({"status": "ok", "data": data_service.get_llm_insight(api_key)})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
@role_required("Admin")
def api_run_pipeline(request):
    """Only Admin can trigger the data pipeline."""
    if request.method == "POST":
        try:
            return JsonResponse({"status": "ok", "data": data_service.run_pipeline_job()})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "POST required"}, status=405)
# ── HELPER ────────────────────────────────────────────────
def _get_role(request) -> str:
    if request.user.is_superuser:
        return "Admin"
    groups = list(request.user.groups.values_list("name", flat=True))
    if "Admin" in groups:
        return "Admin"
    if "Analyst" in groups:
        return "Analyst"
    return "Viewer"
''')
# ── analytics/data_service.py ─────────────────────────────
write("analytics/data_service.py", '''
import pandas as pd
import numpy as np
from pathlib import Path
import json, os, gc, warnings
warnings.filterwarnings("ignore")
_cache = {}
def get_data() -> pd.DataFrame:
    if "df" not in _cache:
        # Try clean_data.csv first, then fall back to raw CSV
        candidates = [
            Path("outputs/clean_data.csv"),
            Path("../outputs/clean_data.csv"),
            Path("../data/SampleSuperStore.csv"),
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path, parse_dates=["Order_Date"], low_memory=False)
                if "Profit_Margin" not in df.columns and "Profit" in df.columns:
                    df["Profit_Margin"] = df["Profit"] / (df["Sales"] + 1e-9) * 100
                if "Year" not in df.columns and "Order_Date" in df.columns:
                    df["Year"]      = df["Order_Date"].dt.year
                    df["Month_Num"] = df["Order_Date"].dt.month
                    df["Quarter"]   = df["Order_Date"].dt.quarter
                _cache["df"] = df
                print(f"Loaded {len(df):,} rows from {path}")
                break
        else:
            raise FileNotFoundError("Run notebooks/01_setup_and_eda.py first.")
    return _cache["df"]
def get_kpis() -> dict:
    df = get_data()
    yoy = df.groupby("Year")["Sales"].sum()
    growth = float((yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2] * 100) if len(yoy) >= 2 else 0
    return {
        "total_revenue": int(df["Sales"].sum()),
        "total_profit":  float(round(df["Profit"].sum(), 0)),
        "avg_margin":    float(round(df["Profit_Margin"].mean(), 1)),
        "total_orders":  int(len(df)),
        "yoy_growth":    round(growth, 1),
        "loss_orders":   int((df["Profit"] < 0).sum()),
        "loss_pct":      round((df["Profit"] < 0).mean() * 100, 1),
        "avg_shipping_days": round(df["Shipping_Days"].mean(), 1) if "Shipping_Days" in df.columns else 0,
    }
def get_sales_trend() -> dict:
    df = get_data()
    monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly.columns = ["Period", "Sales"]
    monthly["Date"]    = monthly["Period"].dt.to_timestamp()
    monthly            = monthly.sort_values("Date")
    monthly["Rolling"] = monthly["Sales"].rolling(3).mean()
    return {
        "labels":  [d.strftime("%b %Y") for d in monthly["Date"]],
        "sales":   monthly["Sales"].tolist(),
        "rolling": monthly["Rolling"].fillna(0).tolist(),
    }
def get_category_data() -> dict:
    df = get_data()
    cat = df.groupby("Category").agg(Revenue=("Sales","sum"), Profit=("Profit","sum")).reset_index()
    sub_col = "Sub_Category" if "Sub_Category" in df.columns else "Sub-Category"
    sub = df.groupby(sub_col)["Profit"].sum().sort_values().reset_index()
    sub.columns = ["name", "profit"]
    return {
        "categories": cat["Category"].tolist(),
        "revenues":   cat["Revenue"].tolist(),
        "profits":    cat["Profit"].tolist(),
        "sub_names":  sub["name"].tolist(),
        "sub_profits":[round(p, 0) for p in sub["profit"].tolist()],
    }
def get_market_data() -> dict:
    df = get_data()
    mkt = df.groupby("Market").agg(
        Revenue=("Sales","sum"), Profit=("Profit","sum"), Orders=("Sales","count")
    ).reset_index()
    mkt["Margin"] = (mkt["Profit"] / mkt["Revenue"] * 100).round(1)
    mkt = mkt.sort_values("Revenue", ascending=False)
    return {
        "markets":  mkt["Market"].tolist(),
        "revenues": [int(v) for v in mkt["Revenue"].tolist()],
        "profits":  [round(float(v), 0) for v in mkt["Profit"].tolist()],
        "margins":  mkt["Margin"].tolist(),
        "orders":   mkt["Orders"].tolist(),
    }
def get_discount_impact() -> dict:
    df = get_data()
    bands  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    labels = ["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40%+"]
    margins, counts = [], []
    for i in range(len(bands) - 1):
        mask = (df["Discount"] >= bands[i]) & (df["Discount"] < bands[i+1])
        margins.append(round(float(df.loc[mask, "Profit_Margin"].mean()), 1))
        counts.append(int(mask.sum()))
    return {"labels": labels, "margins": margins, "counts": counts}
def get_forecast_data() -> dict:
    if "forecast" in _cache:
        return _cache["forecast"]
    df = get_data()
    monthly = df.groupby(df["Order_Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly.columns = ["Period", "Sales"]
    monthly["Date"] = monthly["Period"].dt.to_timestamp()
    monthly = monthly.sort_values("Date").set_index("Date")
    ts = monthly["Sales"]
    result = {"labels": [d.strftime("%b %Y") for d in ts.index],
                "actual": ts.tolist(), "forecast": [], "upper": [], "lower": [],
                "future_labels": [], "future_vals": [], "rmse": 0, "mape": 0}
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False,
                        freq='MS')  
        fitted = model.fit(disp=False)
        
        fc = fitted.get_forecast(steps=len(test))
        pred, ci = fc.predicted_mean, fc.conf_int()
        
        rmse = float(np.sqrt(np.mean((test.values - pred.values)**2)))
        mape = float(np.mean(np.abs((test.values - pred.values) / (test.values + 1e-9))) * 100)
        
        final = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False,
                        freq='MS') 
        final_fit = final.fit(disp=False)
        future = final_fit.get_forecast(steps=3)
        fp, fci = future.predicted_mean, future.conf_int()
        
        result.update({
            "forecast": [None]*train_size + pred.tolist(),
            "upper":    [None]*train_size + ci.iloc[:, 1].tolist(),
            "lower":    [None]*train_size + ci.iloc[:, 0].tolist(),
            "future_labels": [d.strftime("%b %Y") for d in fp.index],
            "future_vals":   [int(v) for v in fp.tolist()],
            "future_upper":  [int(v) for v in fci.iloc[:, 1].tolist()],
            "future_lower":  [int(v) for v in fci.iloc[:, 0].tolist()],
            "rmse": int(rmse), "mape": round(mape, 1), "train_size": train_size,
        })
        _cache["forecast"] = result
    except Exception as e:
        result["error"] = str(e)
    return result
def get_shap_data() -> dict:
    if "shap" in _cache:
        return _cache["shap"]
    df = get_data()
    model = None
    explainer = None
    try:
        import shap
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb
        if "Profit_Margin" not in df.columns:
            df["Profit_Margin"] = df["Profit"] / (df["Sales"] + 1e-9) * 100
        df["Margin_Class"] = pd.cut(df["Profit_Margin"],
            bins=[-999, 0, 10, 999], labels=["Low/Loss", "Medium", "High"])
        cat_cols = [c for c in ["Category","Sub_Category","Market","Region",
                                "Segment","Ship_Mode","Order_Priority"] if c in df.columns]
        num_cols = [c for c in ["Discount","Quantity","Shipping_Cost",
                                "Sales","Year","Month_Num","Quarter"] if c in df.columns]
        all_cols = cat_cols + num_cols
        df_ml = df[all_cols + ["Margin_Class"]].dropna().copy()
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        le_y = LabelEncoder()
        y = le_y.fit_transform(df_ml["Margin_Class"].astype(str))
        X = df_ml[all_cols]
        idx = np.random.choice(len(X), min(3000, len(X)), replace=False)
        X_s, y_s = X.iloc[idx], y[idx]
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4,
                                   random_state=42, verbosity=0, eval_metric="mlogloss")
        model.fit(X_s, y_s)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_s[:500])
        if isinstance(shap_vals, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_shap = np.abs(shap_vals).mean(axis=0)
        feat_imp = pd.Series(mean_shap, index=all_cols).sort_values(ascending=False)
        result = {
            "features": feat_imp.index.tolist(),
            "importance": [round(float(v), 4) for v in feat_imp.values.tolist()],
        }
        _cache["shap"] = result
        return result
    except ImportError:
        return {"error": "shap not installed. Run: pip install shap"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # FIX: explicitly tear down the XGBoost booster / SHAP explainer here,
        # while the xgboost module is still fully loaded. Without this, these
        # objects can be garbage-collected during interpreter shutdown, which
        # raises a harmless but noisy:
        #   AttributeError: 'NoneType' object has no attribute 'XGBoosterFree'
        try:
            if explainer is not None:
                del explainer
            if model is not None:
                booster = model.get_booster()
                del model
                del booster
        except Exception:
            pass
        gc.collect()
def run_pipeline_job() -> dict:
    import time
    df = get_data()
    log, start = [], time.time()
    def step(n, msg, status="ok"):
        log.append({"step": n, "msg": msg, "status": status,
                    "time": round(time.time() - start, 2)})
    missing_pct = df.isnull().mean().max() * 100
    step(1, f"Data validation: {len(df):,} rows, {missing_pct:.1f}% max missing",
         "warn" if missing_pct > 20 else "ok")
    step(2, f"KPIs: Revenue=${df[\'Sales\'].sum():,.0f}, Profit=${df[\'Profit\'].sum():,.0f}")
    loss_orders = (df["Profit"] < 0).sum()
    step(3, f"Anomalies: {loss_orders:,} loss-making orders",
         "warn" if loss_orders > 1000 else "ok")
    step(4, f"Feature engineering: {len(df.columns)} columns ready")
    _cache.pop("forecast", None)
    step(5, "Forecast cache cleared")
    step(6, f"Pipeline complete in {time.time()-start:.1f}s")
    return {"log": log, "timestamp": pd.Timestamp.now().isoformat()}
def get_llm_insight(api_key: str = "") -> dict:
    if not api_key:
        return {"error": "No API key",
                "demo": "Demo: $12.6M revenue, 9,571 high-discount loss orders. Cap discounts at 20%."}
    try:
        from openai import OpenAI
        kpis = get_kpis()
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Senior Data Scientist: give 5 sharp quantified insights. "
                f"Revenue=${kpis[\'total_revenue\']:,}, Profit=${kpis[\'total_profit\']:,.0f}, "
                f"Margin={kpis[\'avg_margin\']}%, Loss orders={kpis[\'loss_orders\']:,}. "
                f"Max 150 words, use numbers."}],
            max_tokens=200, temperature=0.2)
        return {"insight": resp.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
''')
# ── BASE TEMPLATE ─────────────────────────────────────────
write("templates/base.html", '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}Global Superstore Analytics{% endblock %}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#F3F4F6;color:#111827}
nav{background:#1E3A5F;padding:0 24px;display:flex;align-items:center;height:56px;gap:24px}
nav a{color:#CBD5E1;text-decoration:none;font-size:14px;padding:6px 10px;border-radius:6px}
nav a:hover,nav a.active{background:rgba(255,255,255,0.12);color:#fff}
nav .brand{font-weight:700;font-size:16px;color:#fff;margin-right:16px}
nav .spacer{flex:1}
nav .user{font-size:13px;color:#94A3B8}
nav .role-badge{font-size:11px;padding:2px 8px;border-radius:10px;margin-left:6px;
    font-weight:600;background:rgba(255,255,255,0.15);color:#E2E8F0}
.main{max-width:1400px;margin:0 auto;padding:28px 24px}
.page-title{font-size:24px;font-weight:700;margin-bottom:4px}
.page-sub{font-size:14px;color:#6B7280;margin-bottom:24px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:28px}
.kpi-card{background:#fff;border-radius:10px;padding:16px 18px;border-left:4px solid #2563EB;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.kpi-label{font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.kpi-val{font-size:26px;font-weight:700;color:#111827}
.kpi-delta{font-size:12px;margin-top:2px}
.kpi-up{color:#16A34A}.kpi-down{color:#DC2626}
.card{background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:20px}
.card-title{font-size:14px;font-weight:600;color:#374151;margin-bottom:16px;text-transform:uppercase;letter-spacing:.04em}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
.btn{display:inline-block;padding:9px 18px;border-radius:7px;border:none;cursor:pointer;font-size:14px;font-weight:500}
.btn-primary{background:#2563EB;color:#fff}.btn-primary:hover{background:#1D4ED8}
.btn-danger{background:#DC2626;color:#fff}.btn-danger:hover{background:#B91C1C}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}
.badge-ok{background:#D1FAE5;color:#065F46}
.badge-warn{background:#FEF3C7;color:#92400E}
.badge-err{background:#FEE2E2;color:#991B1B}
.alert-info{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:12px 16px;color:#1E40AF;font-size:13px;margin-bottom:16px}
canvas{max-height:320px}
</style>
</head>
<body>
<nav>
  <span class="brand">📊 Superstore Analytics</span>
  {% if user.is_authenticated %}
  <a href="/" {% if request.path == "/" %}class="active"{% endif %}>Dashboard</a>
  <a href="/forecast/" {% if "/forecast/" in request.path %}class="active"{% endif %}>Forecast</a>
  <a href="/ml/" {% if "/ml/" in request.path %}class="active"{% endif %}>ML Model</a>
  <a href="/market/" {% if "/market/" in request.path %}class="active"{% endif %}>Markets</a>
  {% if role in "Admin,Analyst" %}
  <a href="/shap/" {% if "/shap/" in request.path %}class="active"{% endif %}>SHAP</a>
  {% endif %}
  {% if role == "Admin" %}
  <a href="/pipeline/" {% if "/pipeline/" in request.path %}class="active"{% endif %}>Pipeline</a>
  {% endif %}
  <span class="spacer"></span>
  <span class="user">{{ user.username }}<span class="role-badge">{{ role }}</span></span>
    <form method="post" action="/logout/" style="display:inline;margin-left:8px">
    {% csrf_token %}
    <button type="submit" style="background:none;border:none;color:#94A3B8;font-size:12px;cursor:pointer">Logout</button>
    </form>
  {% endif %}
</nav>
<div class="main">
{% block content %}{% endblock %}
</div>
</body>
</html>
''')
# ── LOGIN TEMPLATE ────────────────────────────────────────
write("templates/registration/login.html", '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Login — Superstore Analytics</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#F3F4F6;display:flex;align-items:center;justify-content:center;min-height:100vh}
.login-box{background:#fff;border-radius:12px;padding:40px 36px;width:360px;box-shadow:0 4px 24px rgba(0,0,0,.1)}
h1{font-size:22px;font-weight:700;color:#111827;margin-bottom:6px}
.sub{font-size:13px;color:#6B7280;margin-bottom:28px}
label{display:block;font-size:12px;font-weight:600;color:#374151;margin-bottom:5px;margin-top:14px;text-transform:uppercase;letter-spacing:.04em}
input{width:100%;padding:10px 12px;border:1px solid #D1D5DB;border-radius:7px;font-size:14px;outline:none}
input:focus{border-color:#2563EB;box-shadow:0 0 0 3px rgba(37,99,235,.1)}
.btn{display:block;width:100%;margin-top:22px;padding:11px;background:#2563EB;color:#fff;border:none;border-radius:7px;font-size:15px;font-weight:600;cursor:pointer}
.btn:hover{background:#1D4ED8}
.error{background:#FEE2E2;border:1px solid #FECACA;border-radius:7px;padding:10px 14px;color:#991B1B;font-size:13px;margin-bottom:14px}
.roles{background:#F0F9FF;border-radius:8px;padding:12px 14px;margin-top:20px;font-size:12px;color:#0369A1}
.roles b{display:block;margin-bottom:5px;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
</style>
</head>
<body>
<div class="login-box">
  <h1>📊 Superstore Analytics</h1>
  <div class="sub">Sign in to access the dashboard</div>
  {% if form.errors %}
  <div class="error">Invalid username or password. Please try again.</div>
  {% endif %}
  <form method="post">
    {% csrf_token %}
    <input type="hidden" name="next" value="{{ next }}">
    <label>Username</label>
    <input type="text" name="username" autofocus autocomplete="username">
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password">
    <button type="submit" class="btn">Sign In</button>
  </form>
  <div class="roles">
    <b>Role-Based Access</b>
    <b>Admin</b> — Full access: all pages + pipeline trigger<br>
    <b>Analyst</b> — Charts, SHAP, forecasting (no pipeline)<br>
    <b>Viewer</b> — Read-only: KPIs and charts only
  </div>
</div>
</body>
</html>
''')

# ── LOGGED OUT TEMPLATE ───────────────────────────────────
write("templates/registration/logged_out.html", '''{% extends "base.html" %}
{% block title %}Logged Out{% endblock %}
{% block content %}
<div style="max-width:500px;margin:100px auto;text-align:center">
  <h2>✅ You have been logged out</h2>
  <p><a href="/login/" class="btn btn-primary">Sign in again</a></p>
</div>
{% endblock %}
''')

# ── DASHBOARD TEMPLATE ────────────────────────────────────
write("templates/analytics/dashboard.html", '''{% extends "base.html" %}
{% block title %}Dashboard — Superstore{% endblock %}
{% block content %}
<div class="page-title">Executive Dashboard</div>
<div class="page-sub">Global Superstore — 51,290 orders across 7 markets (2011-2014)</div>
<div class="kpi-grid" id="kpi-grid">
  <div class="kpi-card" style="border-left-color:#2563EB">
    <div class="kpi-label">Total Revenue</div>
    <div class="kpi-val" id="kpi-rev">Loading…</div>
    <div class="kpi-delta kpi-up" id="kpi-growth"></div>
  </div>
  <div class="kpi-card" style="border-left-color:#16A34A">
    <div class="kpi-label">Total Profit</div>
    <div class="kpi-val" id="kpi-profit">Loading…</div>
  </div>
  <div class="kpi-card" style="border-left-color:#7C3AED">
    <div class="kpi-label">Avg Margin</div>
    <div class="kpi-val" id="kpi-margin">Loading…</div>
  </div>
  <div class="kpi-card" style="border-left-color:#DC2626">
    <div class="kpi-label">Loss Orders</div>
    <div class="kpi-val" id="kpi-loss">Loading…</div>
  </div>
  <div class="kpi-card" style="border-left-color:#0D9488">
    <div class="kpi-label">Avg Shipping</div>
    <div class="kpi-val" id="kpi-ship">Loading…</div>
  </div>
</div>
<div class="grid-2">
  <div class="card">
    <div class="card-title">Monthly Sales Trend</div>
    <canvas id="trendChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Revenue by Category</div>
    <canvas id="catChart"></canvas>
  </div>
</div>
<div class="grid-2">
  <div class="card">
    <div class="card-title">Discount Impact on Margin</div>
    <canvas id="discChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Revenue by Market</div>
    <canvas id="mktChart"></canvas>
  </div>
</div>
<script>
async function fetchJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (j.status !== "ok") throw new Error(j.message);
  return j.data;
}
function fmt(n) { return "$" + (n/1e6).toFixed(2) + "M"; }
function fmtK(n) { return "$" + (n/1000).toFixed(0) + "K"; }
(async () => {
  try {
    const kpis = await fetchJSON("/api/kpis/");
    document.getElementById("kpi-rev").textContent    = fmt(kpis.total_revenue);
    document.getElementById("kpi-profit").textContent = fmt(kpis.total_profit);
    document.getElementById("kpi-margin").textContent = kpis.avg_margin.toFixed(1) + "%";
    document.getElementById("kpi-loss").textContent   = kpis.loss_orders.toLocaleString() + " (" + kpis.loss_pct + "%)";
    document.getElementById("kpi-ship").textContent   = kpis.avg_shipping_days + "d";
    document.getElementById("kpi-growth").textContent = (kpis.yoy_growth >= 0 ? "▲ " : "▼ ") + Math.abs(kpis.yoy_growth) + "% YoY";
  } catch(e) { console.error("KPI error:", e); }
  try {
    const trend = await fetchJSON("/api/sales-trend/");
    new Chart(document.getElementById("trendChart"), {
      type: "line",
      data: {
        labels: trend.labels,
        datasets: [
          { label: "Sales", data: trend.sales, borderColor: "#2563EB", backgroundColor: "rgba(37,99,235,.07)", fill: true, tension: 0.3, pointRadius: 2 },
          { label: "3-Mo Avg", data: trend.rolling, borderColor: "#EA580C", borderDash: [5,4], tension: 0.3, pointRadius: 0 }
        ]
      },
      options: { responsive: true, plugins: { legend: { position: "top" } },
        scales: { y: { ticks: { callback: v => "$" + (v/1000).toFixed(0) + "K" } },
                  x: { ticks: { maxTicksLimit: 8 } } } }
    });
  } catch(e) { console.error("Trend error:", e); }
  try {
    const cat = await fetchJSON("/api/category/");
    new Chart(document.getElementById("catChart"), {
      type: "doughnut",
      data: { labels: cat.categories, datasets: [{ data: cat.revenues,
        backgroundColor: ["#2563EB","#16A34A","#EA580C"], borderWidth: 2 }] },
      options: { responsive: true, cutout: "55%",
        plugins: { legend: { position: "bottom" } } }
    });
  } catch(e) { console.error("Cat error:", e); }
  try {
    const disc = await fetchJSON("/api/discount/");
    new Chart(document.getElementById("discChart"), {
      type: "bar",
      data: { labels: disc.labels, datasets: [{ label: "Avg Margin %", data: disc.margins,
        backgroundColor: disc.margins.map(m => m < 0 ? "#DC2626" : "#16A34A"), borderRadius: 4 }] },
      options: { responsive: true,
        scales: { y: { ticks: { callback: v => v + "%" } } } }
    });
  } catch(e) { console.error("Disc error:", e); }
  try {
    const mkt = await fetchJSON("/api/market/");
    new Chart(document.getElementById("mktChart"), {
      type: "bar",
      data: { labels: mkt.markets, datasets: [{ label: "Revenue", data: mkt.revenues,
        backgroundColor: "#2563EB", borderRadius: 4 }] },
      options: { indexAxis: "y", responsive: true,
        scales: { x: { ticks: { callback: v => "$" + (v/1e6).toFixed(1) + "M" } } } }
    });
  } catch(e) { console.error("Mkt error:", e); }
})();
</script>
{% endblock %}
''')
# ── PIPELINE TEMPLATE ─────────────────────────────────────
write("templates/analytics/pipeline.html", '''{% extends "base.html" %}
{% block title %}Pipeline — Superstore{% endblock %}
{% block content %}
<div class="page-title">AI Data Pipeline</div>
<div class="page-sub">Automated pipeline — Admin only</div>
<div class="alert-info">
  Role-based access: Only the <strong>Admin</strong> role can trigger the pipeline.
  Analysts and Viewers can see results but cannot run new jobs.
</div>
{% if role == "Admin" %}
<button class="btn btn-primary" onclick="runPipeline()">▶ Run Pipeline Now</button>
{% else %}
<div class="card"><p>You need Admin role to trigger the pipeline.</p></div>
{% endif %}
<div class="card" style="margin-top:20px" id="log-card" style="display:none">
  <div class="card-title">Pipeline Log</div>
  <div id="pipeline-log" style="font-family:monospace;font-size:13px;line-height:1.8"></div>
</div>
<script>
async function runPipeline() {
  document.getElementById("pipeline-log").innerHTML = "Running…";
  document.getElementById("log-card").style.display = "block";
  try {
    const r = await fetch("/api/run-pipeline/", {
      method: "POST",
      headers: { "X-CSRFToken": getCookie("csrftoken") }
    });
    const j = await r.json();
    if (j.status !== "ok") throw new Error(j.message);
    const html = j.data.log.map(s => {
      const icon = s.status === "ok" ? "✓" : s.status === "warn" ? "⚠" : "✗";
      const color = s.status === "ok" ? "#16A34A" : s.status === "warn" ? "#D97706" : "#DC2626";
      return `<div style="color:${color}">[${s.time}s] ${icon} Step ${s.step}: ${s.msg}</div>`;
    }).join("");
    document.getElementById("pipeline-log").innerHTML = html;
  } catch(e) {
    document.getElementById("pipeline-log").innerHTML = "<span style='color:red'>Error: " + e.message + "</span>";
  }
}
function getCookie(name) {
  const v = document.cookie.match("(^|;) ?" + name + "=([^;]*)(;|$)");
  return v ? v[2] : null;
}
</script>
{% endblock %}
''')
# ── MINIMAL REMAINING TEMPLATES ───────────────────────────
for tpl_name, tpl_title, tpl_api, tpl_chart_id in [
    ("forecast", "Forecast", "api/forecast/", "forecastChart"),
    ("ml",       "ML Model", "api/shap/",     "shapChart"),
    ("market",   "Markets",  "api/market/",   "mktDetailChart"),
    ("shap",     "SHAP",     "api/shap/",     "shapChart2"),
]:
    write(f"templates/analytics/{tpl_name}.html", f'''{{% extends "base.html" %}}
{{% block title %}}{tpl_title} — Superstore{{% endblock %}}
{{% block content %}}
<div class="page-title">{tpl_title}</div>
<div class="page-sub">Data loaded from outputs/clean_data.csv</div>
<div class="card">
  <div class="card-title">{tpl_title} Chart</div>
  <canvas id="{tpl_chart_id}"></canvas>
  <div id="chart-msg" style="font-size:13px;color:#6B7280;margin-top:10px">Loading...</div>
</div>
<script>
fetch("/{tpl_api}")
  .then(r => r.json())
  .then(j => {{
    document.getElementById("chart-msg").textContent = "Data loaded: " + JSON.stringify(j).slice(0,100) + "...";
  }})
  .catch(e => {{ document.getElementById("chart-msg").textContent = "Error: " + e.message; }});
</script>
{{% endblock %}}
''')
# ── FINAL INSTRUCTIONS ────────────────────────────────────
print("\\n" + "=" * 60)
print("✓ Django setup complete!")
print()
print("NEXT STEPS:")
print()
print("1. Install Django (if not done):")
print("   pip install django")
print()
print("2. Run migrations:")
print("   python manage.py migrate")
print()
print("3. Create your superuser (Admin):")
print("   python manage.py createsuperuser")
print()
print("4. Assign roles via Django admin:")
print("   python manage.py runserver")
print("   → http://127.0.0.1:8000/admin/")
print("   → Auth > Users > Select user > Groups")
print("   → Assign: Admin / Analyst / Viewer")
print()
print("5. Visit the dashboard:")
print("   http://127.0.0.1:8000")
print()
print("ROLE ACCESS SUMMARY:")
print("   Admin   → All pages + pipeline trigger + SHAP")
print("   Analyst → All pages + SHAP, no pipeline")
print("   Viewer  → Dashboard, Forecast, ML, Market only")
print("=" * 60)