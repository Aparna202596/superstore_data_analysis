"""
============================================================
setup_django.py
============================================================
Run this script ONCE to set up the entire Django project.
It creates all necessary folders and config files.

Usage:
    cd "C:\\visual studio\\data_analysis"
    python setup_django.py
    python manage.py migrate
    python manage.py runserver

Then open: http://127.0.0.1:8000
============================================================
"""

import os, sys, shutil
from pathlib import Path

ROOT = Path(__file__).parent
print("=" * 60)
print("Django Dashboard Setup")
print(f"Project root: {ROOT}")
print("=" * 60)


# ── STEP 1: Create folder structure ──────────────────────
folders = [
    "dashboard",
    "analytics",
    "templates/analytics",
    "static/css",
    "static/js",
    "../outputs/reports",
    "../outputs/charts",
    "../outputs/models",
    "../outputs/synthetic",
]
for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created: {folder}/")


# ── STEP 2: Create dashboard/__init__.py ─────────────────
(ROOT / "dashboard" / "__init__.py").write_text("")
(ROOT / "analytics" / "__init__.py").write_text("")
print("  ✓ Created __init__.py files")


# ── STEP 3: Create dashboard/wsgi.py ─────────────────────
wsgi = '''import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dashboard.settings')
application = get_wsgi_application()
'''
(ROOT / "dashboard" / "wsgi.py").write_text(wsgi)
print("  ✓ Created wsgi.py")


# ── STEP 4: Create manage.py ─────────────────────────────
manage = '''#!/usr/bin/env python
import os, sys
def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dashboard.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Couldn't import Django.") from exc
    execute_from_command_line(sys.argv)
if __name__ == '__main__':
    main()
'''
(ROOT / "manage.py").write_text(manage)
print("  ✓ Created manage.py")


# ── STEP 5: Update views.py with pipeline context ────────
views_addition = '''
# Add this import at top of views.py
from .pipeline_context import PIPELINE_STEPS, EXAMPLE_QUESTIONS

# Replace your pipeline() function with:
def pipeline(request):
    """AI pipeline control page."""
    return render(request, 'analytics/pipeline.html', {
        'pipeline_steps': PIPELINE_STEPS,
        'example_questions': EXAMPLE_QUESTIONS,
    })
'''
print("\n  NOTE: Update your views.py pipeline() function:")
print(views_addition)


# ── STEP 6: Create analytics/apps.py ─────────────────────
apps = '''from django.apps import AppConfig
class AnalyticsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "analytics"
'''
(ROOT / "analytics" / "apps.py").write_text(apps)
print("  ✓ Created analytics/apps.py")


# ── STEP 7: Create analytics/models.py ───────────────────
# We don't use a database for this project (read-only CSV)
# but Django requires this file
(ROOT / "analytics" / "models.py").write_text(
    "# No database models needed — data comes from CSV files\n"
)
print("  ✓ Created analytics/models.py")


# ── STEP 8: Verify key files exist ───────────────────────
print("\n[Checking required files...]")
required = [
    "dashboard/settings.py",
    "dashboard/urls.py",
    "analytics/views.py",
    "analytics/urls.py",
    "analytics/data_service.py",
    "templates/base.html",
    "templates/analytics/dashboard.html",
    "templates/analytics/forecast.html",
    "templates/analytics/ml.html",
    "templates/analytics/shap.html",
    "templates/analytics/market.html",
    "templates/analytics/pipeline.html",
]
all_good = True
for f in required:
    exists = (ROOT / f).exists()
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status} {f}")
    if not exists:
        all_good = False

print()
if all_good:
    print("=" * 60)
    print("✓ All files present!")
    print()
    print("NEXT STEPS:")
    print()
    print("1. Install Django (if not done):")
    print("   pip install django")
    print()
    print("2. Run database migration:")
    print("   python manage.py migrate")
    print()
    print("3. Start the development server:")
    print("   python manage.py runserver")
    print()
    print("4. Open in browser:")
    print("   http://127.0.0.1:8000")
    print()
    print("5. (Optional) Create admin user:")
    print("   python manage.py createsuperuser")
    print("   → Then visit http://127.0.0.1:8000/admin/")
    print("=" * 60)
else:
    print("=" * 60)
    print("✗ Some files are missing.")
    print("  Make sure you downloaded ALL files from the project.")
    print("=" * 60)