
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
