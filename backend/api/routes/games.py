"""
PROPCAST – /api/games/* endpoints
Returns game schedules, matchup context, weather, and per-game projections.

IMPORTANT: No live/post-kickoff data is used in features.
           If data is unavailable, DATA_UNAVAILABLE is returned.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from backend.api.schemas import (
    DataUnavailable,
    GameDetail,
    GameList,
    GameWithTeams,
    InjuryWithPlayer,
    PropCard,
)
from backend.db.database import get_db
from backend.db.models import Game, Injury, Player, PlayerGameStat, Prediction, Team

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Games"])

DISCLAIMER = (
    "Model probabilities are statistical estimates, not guarantees. "
    "Sports outcomes contain substantial randomness."
)


def _load_game_options():
    """Eager-load options for games query."""
    return [
        joinedload(Game.home_team),
        joinedload(Game.away_team),
        joinedload(Game.weather),
    ]


# ---------------------------------------------------------------------------
# GET /api/games/today
# ---------------------------------------------------------------------------
@router.get(
    "/games/today",
    summary="Today's games",
    response_model=GameList,
    responses={200: {"description": "List of today's games with matchup context"}},
)
def get_todays_games(db: Session = Depends(get_db)):
    """Return all games scheduled for today (UTC date)."""
    today = date.today()
    start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    games = (
        db.execute(
            select(Game)
            .options(*_load_game_options())
            .where(Game.kickoff_time >= start, Game.kickoff_time < end)
            .order_by(Game.kickoff_time)
        )
        .scalars()
        .all()
    )

    if not games:
        # Return empty list with informative count rather than 404
        logger.info("No games found for today (%s)", today.isoformat())
        return GameList(games=[], count=0)

    return GameList(games=games, count=len(games))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GET /api/games/week
# ---------------------------------------------------------------------------
@router.get(
    "/games/week",
    summary="Current week's games",
    response_model=GameList,
    responses={200: {"description": "All games in the current NFL week"}},
)
def get_week_games(db: Session = Depends(get_db)):
    """Return all games for the most recent season/week in the database."""
    # Find the max season first, then max week for that season
    season_row = db.execute(select(func.max(Game.season))).scalar()
    if season_row is None:
        return GameList(games=[], count=0)

    week_row = db.execute(
        select(func.max(Game.week)).where(Game.season == season_row)
    ).scalar()
    if week_row is None:
        return GameList(games=[], count=0)

    games = (
        db.execute(
            select(Game)
            .options(*_load_game_options())
            .where(Game.season == season_row, Game.week == week_row)
            .order_by(Game.kickoff_time)
        )
        .scalars()
        .all()
    )

    return GameList(games=games, count=len(games))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GET /api/game/{game_id}
# ---------------------------------------------------------------------------
@router.get(
    "/game/{game_id}",
    summary="Single game detail",
    response_model=GameDetail,
    responses={
        404: {"description": "Game not found"},
        200: {"description": "Full game detail with rosters, injuries, and projections"},
    },
)
def get_game_detail(game_id: int, db: Session = Depends(get_db)):
    """Return a single game with team rosters, injuries, and all player projections."""
    game = (
        db.execute(
            select(Game)
            .options(*_load_game_options())
            .where(Game.id == game_id)
        )
        .scalars()
        .first()
    )

    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found.")

    # --- Players for each team ---
    home_players: List[Player] = []
    away_players: List[Player] = []

    if game.home_team_id:
        home_players = (
            db.execute(
                select(Player)
                .where(Player.team_id == game.home_team_id)
                .where(Player.position.in_(["QB", "RB", "WR", "TE"]))
                .order_by(Player.position, Player.full_name)
            )
            .scalars()
            .all()
        )
    if game.away_team_id:
        away_players = (
            db.execute(
                select(Player)
                .where(Player.team_id == game.away_team_id)
                .where(Player.position.in_(["QB", "RB", "WR", "TE"]))
                .order_by(Player.position, Player.full_name)
            )
            .scalars()
            .all()
        )

    # --- Injuries for this game ---
    injuries = (
        db.execute(
            select(Injury)
            .options(joinedload(Injury.player))
            .where(Injury.game_id == game_id)
        )
        .scalars()
        .all()
    )

    # --- Current predictions for this game ---
    raw_preds = (
        db.execute(
            select(Prediction)
            .options(
                joinedload(Prediction.player).joinedload(Player.team),
            )
            .where(Prediction.game_id == game_id, Prediction.is_current.is_(True))
        )
        .scalars()
        .all()
    )

    prop_cards: List[PropCard] = []
    for pred in raw_preds:
        player = pred.player
        team = player.team if player else None

        # Determine opponent team id
        if team and game.home_team_id and game.away_team_id:
            opp_id = (
                game.away_team_id
                if team.id == game.home_team_id
                else game.home_team_id
            )
            opp = db.get(Team, opp_id)
        else:
            opp = None

        card = PropCard(
            player=player,
            team=team,
            opponent=opp,
            game=game,
            prop_type=pred.prop_type,
            projection=pred.projection,
            std_dev=pred.std_dev,
            percentile_25=pred.percentile_25,
            percentile_50=pred.percentile_50,
            percentile_75=pred.percentile_75,
            market_line=None,
            over_odds=None,
            under_odds=None,
            book=None,
            over_probability=None,
            under_probability=None,
            model_edge=None,
            data_quality_score=None,
            last_updated=pred.prediction_created_at,
            model_version=pred.model_version,
        )
        prop_cards.append(card)

    return GameDetail(
        game=game,  # type: ignore[arg-type]
        home_players=home_players,  # type: ignore[arg-type]
        away_players=away_players,  # type: ignore[arg-type]
        predictions=prop_cards,
        injuries=injuries,  # type: ignore[arg-type]
        disclaimer=DISCLAIMER,
    )
