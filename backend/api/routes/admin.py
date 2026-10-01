"""
PROPCAST – /api/admin/* endpoints (require the X-Admin-Key header).

- GET  /api/admin/status        current week, its prediction count, recent job runs
- POST /api/admin/refresh-week  run the new-week refresh now, in the background

Both are disabled unless the ADMIN_KEY setting is configured. The /admin page
(backend/static/admin.html) is a small form that calls these.
"""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.database import get_db
from backend.db.models import PipelineRun

router = APIRouter(tags=["Admin"])


def require_admin(x_admin_key: Optional[str] = Header(None)) -> None:
    if not settings.admin_key:
        raise HTTPException(status_code=403, detail="Admin is disabled: set ADMIN_KEY on the server.")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(status_code=403, detail="Wrong admin key.")


@router.get("/admin/status", dependencies=[Depends(require_admin)])
def admin_status(db: Session = Depends(get_db)):
    from backend.scheduler import _heavy_job_lock, current_week_prediction_count

    season, week, n_predictions, has_games = current_week_prediction_count()
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(30).all()
    return {
        "season": season,
        "current_week": week,
        "week_has_games": has_games,
        "predictions_this_week": n_predictions,
        "refresh_running": _heavy_job_lock.locked(),
        "runs": [
            {
                "task": r.task_name,
                "status": getattr(r.status, "value", r.status),
                "started_at": r.started_at.isoformat() + "Z" if r.started_at else None,
                "completed_at": r.completed_at.isoformat() + "Z" if r.completed_at else None,
                "error": r.error_message,
            }
            for r in runs
        ],
    }


@router.post("/admin/refresh-week", status_code=202, dependencies=[Depends(require_admin)])
def admin_refresh_week():
    from backend.scheduler import start_manual_refresh

    if not start_manual_refresh():
        raise HTTPException(status_code=409, detail="A refresh is already running. Check status in a few minutes.")
    return {
        "status": "started",
        "detail": "Refreshing schedule, stats, injuries, odds (~96 credits) and this week's predictions. "
                  "Takes about 5-10 minutes.",
    }
