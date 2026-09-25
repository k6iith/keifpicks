"""
PROPCAST – /api/players/* endpoints
Player lookup with optional filtering, season stats, and projections.

IMPORTANT: Never fabricate statistics or projections.
           If data is unavailable, DATA_UNAVAILABLE is returned.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.api.schemas import (
    DataUnavailable,
    InjurySchema,
    MarketLineSchema,
    PlayerDetail,
    PlayerGameStatSchema,
    PlayerWithTeam,
    PredictionSchema,
)
from backend.db.database import get_db
from backend.db.models import Injury, MarketLine, Player, PlayerGameStat, Prediction, Team

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Players"])

DISCLAIMER = (
    "Model probabilities are statistical estimates, not guarantees. "
    "Sports outcomes contain substantial randomness."
)

SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB", "K"}


# ---------------------------------------------------------------------------
# GET /api/players
# ---------------------------------------------------------------------------
@router.get(
    "/players",
    summary="List players",
    response_model=List[PlayerWithTeam],
    responses={200: {"description": "Filtered list of players with team info"}},
)
def list_players(
    team: Optional[str] = Query(None, description="Team abbreviation e.g. DAL"),
    position: Optional[str] = Query(None, description="Position e.g. WR"),
    db: Session = Depends(get_db),
):
    """
    Return players filtered by optional ?team= (abbreviation) and ?position= parameters.
    Only skill-position players (QB/RB/WR/TE/FB/K) are returned by default.
    """
    stmt = (
        select(Player)
        .options(joinedload(Player.team))
        .where(Player.position.in_(SKILL_POSITIONS))
        .order_by(Player.full_name)
    )

    if position:
        stmt = stmt.where(Player.position == position.upper())

    if team:
        # Join to Team table to filter by abbreviation
        stmt = stmt.join(Team, Player.team_id == Team.id).where(
            Team.abbreviation == team.upper()
        )

    players = db.execute(stmt).scalars().unique().all()

    if not players:
        logger.info("No players found for team=%s position=%s", team, position)
        # Return empty list – callers should handle empty gracefully
        return []

    return players  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# GET /api/player/{player_id}
# ---------------------------------------------------------------------------
@router.get(
    "/player/{player_id}",
    summary="Player detail",
    response_model=PlayerDetail,
    responses={
        404: {"description": "Player not found"},
        200: {"description": "Player with stats, projections, and injury report"},
    },
)
def get_player_detail(player_id: int, db: Session = Depends(get_db)):
    """
    Return full player profile including recent game stats (last 8 games),
    current model projections, market lines, and injury status.
    """
    player = (
        db.execute(
            select(Player).options(joinedload(Player.team)).where(Player.id == player_id)
        )
        .scalars()
        .first()
    )

    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found.")

    # --- Recent game stats (last 8 games, most recent first) ---
    recent_stats = (
        db.execute(
            select(PlayerGameStat)
            .where(PlayerGameStat.player_id == player_id)
            .order_by(PlayerGameStat.season.desc(), PlayerGameStat.week.desc())
            .limit(8)
        )
        .scalars()
        .all()
    )

    # --- Current projections (is_current=True) ---
    projections = (
        db.execute(
            select(Prediction)
            .where(Prediction.player_id == player_id, Prediction.is_current.is_(True))
            .order_by(Prediction.prediction_created_at.desc())
        )
        .scalars()
        .all()
    )

    # --- Current market lines ---
    market_lines = (
        db.execute(
            select(MarketLine)
            .where(MarketLine.player_id == player_id)
            .order_by(MarketLine.fetched_at.desc())
            .limit(20)
        )
        .scalars()
        .all()
    )

    # --- Active injuries ---
    injuries = (
        db.execute(
            select(Injury)
            .where(Injury.player_id == player_id)
            .order_by(Injury.report_date.desc())
            .limit(5)
        )
        .scalars()
        .all()
    )

    return PlayerDetail(
        player=player,  # type: ignore[arg-type]
        recent_stats=recent_stats,  # type: ignore[arg-type]
        current_projections=projections,  # type: ignore[arg-type]
        current_market_lines=market_lines,  # type: ignore[arg-type]
        injuries=injuries,  # type: ignore[arg-type]
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# GET /api/player/{player_id}/projections
# ---------------------------------------------------------------------------
@router.get(
    "/player/{player_id}/projections",
    summary="All current projections for a player",
    response_model=List[PredictionSchema],
    responses={
        404: {"description": "Player not found"},
        200: {"description": "All active model projections across all prop types"},
    },
)
def get_player_projections(player_id: int, db: Session = Depends(get_db)):
    """Return all is_current=True predictions for the given player across all prop types."""
    # Verify player exists
    player = db.get(Player, player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found.")

    projections = (
        db.execute(
            select(Prediction)
            .where(Prediction.player_id == player_id, Prediction.is_current.is_(True))
            .order_by(Prediction.prop_type, Prediction.prediction_created_at.desc())
        )
        .scalars()
        .all()
    )

    if not projections:
        logger.info("No current projections for player_id=%d", player_id)
        return []

    return projections  # type: ignore[return-value]
