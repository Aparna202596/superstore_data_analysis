
# Role-based access control uses Django's built-in Group system.
# Groups are created automatically in analytics/migrations/0001_initial_groups.py
# 
# Roles:
#   Admin   — can see everything + run pipeline
#   Analyst — can see all charts + trigger models
#   Viewer  — read-only: KPIs and charts only, no pipeline/SHAP
