"""
PROPCAST – /api/health endpoint
Simple liveness check; no DB dependency.
"""
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness check")
def health_check() -> dict:
    """Returns service status and current UTC timestamp."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "propcast-api",
    }
