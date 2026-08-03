"""
Minimal API-key authentication for write endpoints (POST/DELETE).

Day 19's writeup flagged "no authentication on the API" as a limitation of
the series so far; this is the day-20 step: a lightweight shared-secret
check via the X-API-Key header, enforced by a Flask decorator. This is
intentionally simple (a single static key from config/env, no user
accounts, no rate limiting) — appropriate for a demo, explicitly NOT a
production auth system (see README limitations).
"""
import os
import functools
from flask import request, jsonify


def get_configured_api_key(settings):
    # environment variable takes precedence over the config file, so a real
    # deployment can override the checked-in demo key without editing YAML
    return os.environ.get("PRICING_API_KEY") or settings.get("api", {}).get("api_key", "")


def require_api_key(settings):
    """Decorator factory: wraps a Flask view to require a valid X-API-Key header."""
    def decorator(view_func):
        @functools.wraps(view_func)
        def wrapped(*args, **kwargs):
            expected = get_configured_api_key(settings)
            provided = request.headers.get("X-API-Key", "")
            if not expected:
                return jsonify({"error": "server misconfiguration: no API key configured"}), 500
            if provided != expected:
                return jsonify({"error": "unauthorized: missing or invalid X-API-Key header"}), 401
            return view_func(*args, **kwargs)
        return wrapped
    return decorator
