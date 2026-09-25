"""
PROPCAST – /api/injuries endpoint
Current NFL injury report with player status, description, and source.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.api.schemas import InjuryList, InjuryWithPlayer
from backend.db.database import get_db
from backend.db.models import Injury, Player

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Injuries"])


# ---------------------------------------------------------------------------
# GET /api/injuries
# ---------------------------------------------------------------------------
@router.get(
    "/injuries",
    summary="Current injury report",
    response_model=InjuryList,
)
def get_injuries(
    game_status: Optional[str] = Query(
        None,
        description="Filter by game status: Out | Doubtful | Questionable | IR",
    ),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    position: Optional[str] = Query(None, description="Filter by player position"),
    days: int = Query(7, description="Only return injuries reported within this many days", ge=1, le=90),
    db: Session = Depends(get_db),
):
    """
    Return the current injury report.
    Includes player name, team, position, practice status, game status,
    injury description, data source, and report timestamp.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = (
        select(Injury)
        .options(joinedload(Injury.player).joinedload(Player.team))
        .where(Injury.report_date >= cutoff)
        .order_by(Injury.report_date.desc())
    )

    if game_status:
        stmt = stmt.where(Injury.game_status == game_status)

    injuries = db.execute(stmt).scalars().unique().all()

    # Apply team / position filters in Python (avoids multiple joins)
    if team or position:
        filtered = []
        for inj in injuries:
            if inj.player is None:
                continue
            if team and (inj.player.team is None or inj.player.team.abbreviation != team.upper()):
                continue
            if position and inj.player.position != position.upper():
                continue
            filtered.append(inj)
        injuries = filtered

    last_updated = injuries[0].report_date if injuries else None

    return InjuryList(
        injuries=injuries,  # type: ignore[arg-type]
        count=len(injuries),
        last_updated=last_updated,
    )
