"""
PROPCAST – /api/health endpoint
Simple liveness check; no DB dependency.
"""
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


# Also answers HEAD: uptime monitors such as UptimeRobot (used to keep the
# free Render instance awake) send HEAD by default, and a GET-only route
# answers those with 405, which the monitor reports as "down".
@router.api_route("/health", methods=["GET", "HEAD"], summary="Liveness check")
def health_check() -> dict:
    """Returns service status and current UTC timestamp."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "propcast-api",
    }
