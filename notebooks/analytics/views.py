
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
