
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
                "<p><a href='/'>Back to dashboard</a></p>"
            )
        return _wrapped
    return decorator
