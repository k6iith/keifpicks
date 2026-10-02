"""The app imports cleanly and every page/API the site uses is registered."""


def test_app_imports_and_routes_registered():
    from backend.main import app

    # Newer FastAPI wraps included routers, so read the API paths from the
    # OpenAPI schema; /admin is hidden from the schema, so check it directly.
    paths = set(app.openapi()["paths"]) | {getattr(r, "path", None) for r in app.routes}
    for expected in (
        "/", "/admin", "/api/health",
        "/api/props/today", "/api/props/passing", "/api/props/rushing", "/api/props/receiving",
        "/api/props/play-of-the-week", "/api/odds/credits",
        "/api/admin/status", "/api/admin/refresh-week",
    ):
        assert expected in paths, f"route {expected} is missing"


def test_scheduler_jobs_import():
    import backend.scheduler as scheduler

    for name in ("start_scheduler", "start_manual_refresh", "current_week_prediction_count"):
        assert callable(getattr(scheduler, name)), name
