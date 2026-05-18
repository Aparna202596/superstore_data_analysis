from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from . import data_service
import json


# ══════════════════════════════════════════════════════════
# PAGE VIEWS — return HTML pages
# ══════════════════════════════════════════════════════════

def dashboard(request):
    """Main dashboard page — KPI overview."""
    try:
        kpis = data_service.get_kpis()
    except Exception as e:
        kpis = {'error': str(e)}
    return render(request, 'analytics/dashboard.html', {'kpis': kpis})


def forecast(request):
    """Time-series forecasting page."""
    return render(request, 'analytics/forecast.html', {})


def ml_model(request):
    """ML model results page."""
    return render(request, 'analytics/ml.html', {})


def market(request):
    """Market intelligence page."""
    return render(request, 'analytics/market.html', {})


def pipeline(request):
    """AI pipeline control page."""
    return render(request, 'analytics/pipeline.html', {})


def shap_view(request):
    """SHAP feature importance page."""
    return render(request, 'analytics/shap.html', {})


# ══════════════════════════════════════════════════════════
# API VIEWS — return JSON data (called by JavaScript charts)
# ══════════════════════════════════════════════════════════

def api_kpis(request):
    """GET /api/kpis/ → returns KPI numbers as JSON"""
    try:
        data = data_service.get_kpis()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_sales_trend(request):
    """GET /api/sales-trend/ → monthly sales data for line chart"""
    try:
        data = data_service.get_sales_trend()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_category(request):
    """GET /api/category/ → category breakdown data"""
    try:
        data = data_service.get_category_data()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_market_data(request):
    """GET /api/market/ → market performance data"""
    try:
        data = data_service.get_market_data()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_discount(request):
    """GET /api/discount/ → discount vs margin data"""
    try:
        data = data_service.get_discount_impact()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_forecast_data(request):
    """GET /api/forecast/ → SARIMA forecast data (slow first time)"""
    try:
        data = data_service.get_forecast_data()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_shap(request):
    """GET /api/shap/ → SHAP feature importance values"""
    try:
        data = data_service.get_shap_data()
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_llm_insight(request):
    """GET /api/llm-insight/ → LLM-generated insight text"""
    api_key = request.GET.get('key', '') or getattr(settings, 'OPENAI_API_KEY', '')
    try:
        data = data_service.get_llm_insight(api_key)
        return JsonResponse({'status': 'ok', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_run_pipeline(request):
    """POST /api/run-pipeline/ → runs the data pipeline and returns log"""
    if request.method == 'POST':
        try:
            result = data_service.run_pipeline_job()
            return JsonResponse({'status': 'ok', 'data': result})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)