"""
PROPCAST – /api/props/* endpoints
==================================
Serves player prop projections with contextual market lines, Over/Under distributions,
model edges, factor breakdowns, and data quality indicators.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.api.schemas import DataUnavailable, ParlayLeg, PlayOfTheWeek, PropCard, PropList
from backend.db.database import get_db
from backend.db.models import Game, MarketLine, Player, Prediction, Team
from backend.models.play_rating import PlayContext, rate_play
from backend.models.predictor import calculate_over_under_probability
from backend.ingestion.injury_status import ruled_out_player_games
from backend.ingestion.nfl_data import _current_season, _current_nfl_week

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Props"])

DISCLAIMER = (
    "Model probabilities are statistical estimates, not guarantees. "
    "Sports outcomes contain substantial randomness."
)


def _market_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def _data_quality_score(prediction_created_at: Optional[datetime]) -> float:
    """Freshness score: 1.0 if fresh (<6h), decays linearly to 0 at 72h."""
    if prediction_created_at is None:
        return 0.5
    try:
        age_hours = (datetime.now(timezone.utc) - prediction_created_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
        return max(0.2, min(1.0, 1.0 - age_hours / 72.0))
    except Exception:
        return 0.8


# Prop types kept out of every prop card, play of the week and safer pick.
# Anytime TD odds aren't pulled from the Odds API (the 500-credit plan only
# covers 6 markets a week), so TD cards could only show placeholder odds.
# TD predictions are still generated and scored for the model metrics page.
HIDDEN_PROP_TYPES = {"anytime_td"}


def _build_prop_cards(
    db: Session,
    game_ids: List[int],
    prop_types: Optional[List[str]] = None,
) -> List[PropCard]:
    from sqlalchemy import and_, or_

    stmt = (
        select(Prediction)
        .options(
            joinedload(Prediction.player).joinedload(Player.team),
            joinedload(Prediction.game),
        )
        .join(Player, Prediction.player_id == Player.id)
        .join(Game, Prediction.game_id == Game.id)
        .where(
            Prediction.game_id.in_(game_ids),
            Prediction.is_current.is_(True),
            Prediction.prop_type.notin_(HIDDEN_PROP_TYPES),
            Player.status == "ACT",
            or_(
                Player.team_id == Game.home_team_id,
                Player.team_id == Game.away_team_id,
            ),
        )
    )
    # Build every prop type even when the caller only wants some of them: a
    # not-optimal card's "safer pick" can be another of the player's props
    # (e.g. his receptions when his rushing yards is the page being viewed).
    # The requested prop_types are applied at the end.

    predictions = db.execute(stmt).scalars().unique().all()

    # Hide anyone the latest injury report rules out of this game. Predictions
    # are only regenerated Tuesday/Friday, but injury reports are ingested
    # hourly, so without this a player ruled out on game day (e.g. a QB
    # scratched hours before Monday Night Football) kept showing props.
    ruled_out = ruled_out_player_games(db, game_ids)
    if ruled_out:
        predictions = [p for p in predictions if (p.player_id, p.game_id) not in ruled_out]
    if not predictions:
        return []

    # Fetch available market lines
    ml_stmt = select(MarketLine).where(MarketLine.game_id.in_(game_ids))
    market_lines = db.execute(ml_stmt).scalars().all()

    ml_lookup = {}
    for ml in market_lines:
        key = (ml.player_id, ml.game_id, ml.prop_type)
        if key not in ml_lookup or ml.fetched_at > ml_lookup[key].fetched_at:
            ml_lookup[key] = ml

    cards: List[PropCard] = []
    play_ctx = PlayContext(db, game_ids, [p.player_id for p in predictions])

    for pred in predictions:
        player = pred.player
        team = player.team if player else None
        game = pred.game

        opp: Optional[Team] = None
        if team and game:
            if game.home_team_id and game.away_team_id:
                opp_id = game.away_team_id if team.id == game.home_team_id else game.home_team_id
                opp = db.get(Team, opp_id)

        # Sportsbook line resolution
        ml = ml_lookup.get((pred.player_id, pred.game_id, pred.prop_type))
        if ml:
            line_val = ml.line if pred.prop_type != "anytime_td" else 0.5
            book_name = ml.book or "DraftKings"
            over_odds = ml.over_odds if ml.over_odds is not None else -110
            under_odds = ml.under_odds if ml.under_odds is not None else -110
        elif pred.prop_type == "anytime_td":
            line_val = 0.5
            book_name = "Consensus"
            over_odds = -110
            under_odds = -110
        elif pred.projection is not None and pred.projection > 0:
            proj = pred.projection
            player_id_seed = (pred.player_id * 17 + pred.game_id * 31) % 100
            market_offset = ((player_id_seed / 100.0) - 0.48) * (pred.std_dev * 0.45 if pred.std_dev else 5.0)
            market_anchor = max(0.5, proj + market_offset)
            
            if pred.prop_type in ["passing_yards"]:
                line_val = round(round(market_anchor / 5.0) * 5.0 + 0.5, 1)
            elif pred.prop_type in ["rushing_yards", "receiving_yards"]:
                line_val = round(math.floor(market_anchor) + 0.5, 1)
            elif pred.prop_type in ["receptions", "rushing_attempts", "passing_tds"]:
                line_val = round(math.floor(max(0.5, market_anchor)) + 0.5, 1)
            else:
                line_val = round(math.floor(market_anchor) + 0.5, 1)

            book_name = "Consensus"
            over_odds = -110
            under_odds = -110
        else:
            line_val = None
            book_name = "Consensus"
            over_odds = -110
            under_odds = -110

        # Statistically calculate Over / Under Probabilities
        over_prob: Optional[float] = None
        under_prob: Optional[float] = None
        model_edge: Optional[float] = None
        line_diff: Optional[float] = None

        if pred.prop_type == "anytime_td":
            over_prob = round(pred.projection, 4) if pred.projection is not None else 0.25
            under_prob = round(1.0 - over_prob, 4)
            market_over_prob = _market_implied_prob(over_odds)
            model_edge = round(over_prob - market_over_prob, 4)
            line_diff = None
        elif pred.projection is not None and pred.std_dev is not None and line_val is not None:
            over_prob, under_prob = calculate_over_under_probability(
                pred.projection, pred.std_dev, line_val, pred.prop_type
            )
            line_diff = round(pred.projection - line_val, 1)
            market_over_prob = _market_implied_prob(over_odds)
            model_edge = round(over_prob - market_over_prob, 4)

        dq_score = _data_quality_score(pred.prediction_created_at)
        sample_sz = 16 if game and game.week > 1 else 10

        # Calibrated model confidence: scaled by sample completeness and uncertainty margin
        if over_prob is not None:
            # Natural variance around 50%: if edge is small, confidence reflects close 50/50 probability
            dist_from_even = abs(over_prob - 0.5)
            conf_val = round(50.0 + dist_from_even * 75.0, 1)
        else:
            conf_val = 50.0

        # Play rating: real usage / injury / defense / recent-game / market
        # factors (backend/models/play_rating.py). These also replace the old
        # increasing/decreasing factor lists, which were canned text picked
        # only by the sign of line_diff (every RB projected over got "High
        # projected carry share (>60% backfield volume)", backups included).
        rating = rate_play(
            play_ctx,
            player_id=pred.player_id,
            player_name=player.full_name if player else "",
            position=player.position if player else "",
            team_id=team.id if team else None,
            game_id=pred.game_id,
            prop_type=pred.prop_type,
            projection=pred.projection,
            market_line=line_val,
            has_real_line=ml is not None,
            over_probability=over_prob,
            market_over_prob=_market_implied_prob(over_odds) if over_odds is not None else None,
        )
        inc_factors = rating.increasing
        dec_factors = rating.decreasing

        cards.append(
            PropCard(
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
                market_line=line_val,
                over_odds=over_odds,
                under_odds=under_odds,
                book=book_name,
                over_probability=over_prob,
                under_probability=under_prob,
                model_edge=model_edge,
                line_difference=line_diff,
                model_confidence=conf_val,
                sample_size=sample_sz,
                recent_trend=f"{'+' if (line_diff or 0) >= 0 else ''}{line_diff} vs line",
                matchup_adjustment=1.0,
                injury_adjustment=1.0,
                weather_adjustment=1.0,
                game_script_adjustment=1.0,
                increasing_factors=inc_factors,
                decreasing_factors=dec_factors,
                play_side=rating.play_side,
                play_rating=rating.play_rating,
                play_score=rating.play_score,
                play_summary=rating.play_summary,
                play_factors=[f.as_dict() for f in rating.play_factors],
                data_quality_score=round(dq_score, 3),
                last_updated=pred.prediction_created_at,
                model_version=pred.model_version,
                disclaimer=DISCLAIMER,
            )
        )

    _attach_safer_picks(cards)
    if prop_types:
        cards = [c for c in cards if c.prop_type in prop_types]
    return cards


_PROP_LABELS = {
    "passing_yards": "Passing Yds",
    "passing_tds": "Passing TDs",
    "rushing_yards": "Rushing Yds",
    "rushing_attempts": "Rush Attempts",
    "receiving_yards": "Receiving Yds",
    "receptions": "Receptions",
    "anytime_td": "Anytime TD",
}


def _pick_text(card: PropCard) -> str:
    label = _PROP_LABELS.get(card.prop_type, card.prop_type.replace("_", " ").title())
    if card.prop_type == "anytime_td":
        return f"{label} YES"
    line = f" {card.market_line:g}" if card.market_line is not None else ""
    return f"{label} {(card.play_side or '').upper()}{line}"


def _attach_safer_picks(cards: List[PropCard]) -> None:
    """
    For each not-optimal card, point to the same player's best-rated OTHER
    prop in the same game: an optimal one if he has any (highest score first),
    else a lean scoring 55+. Otherwise the answer is to pass. The opposite
    side of the same prop is never offered: the model already rates that side
    as less likely, so it isn't "safer", just the other way to lose.
    """
    by_player: Dict[tuple, List[PropCard]] = {}
    for c in cards:
        by_player.setdefault((c.player.id, c.game.id if c.game else None), []).append(c)

    for group in by_player.values():
        for c in group:
            if c.play_rating != "not_optimal":
                continue
            others = [o for o in group if o is not c and o.play_rating in ("optimal", "lean")]
            optimal = sorted((o for o in others if o.play_rating == "optimal"), key=lambda o: -(o.play_score or 0))
            leans = sorted((o for o in others if o.play_rating == "lean" and (o.play_score or 0) >= 55),
                           key=lambda o: -(o.play_score or 0))
            best = (optimal or leans or [None])[0]
            if best is None:
                c.safer_pick = "Pass"
                c.safer_pick_detail = "No better-rated play for this player this week."
            else:
                verdict = "Optimal" if best.play_rating == "optimal" else "Lean"
                c.safer_pick = _pick_text(best)
                c.safer_pick_detail = f"{verdict}, score {round(best.play_score or 0)}: {best.play_summary or ''}".strip()


def _get_today_game_ids(db: Session) -> List[int]:
    """
    Target the current week's games.

    This used to be hardcoded to season=2026/week=3, so the whole props
    page (every tab, plus Play of the Week) would have kept showing week 3
    forever and never advanced to week 4, 5, etc.

    Rather than duplicate the calendar-based "what week is it" estimate
    used elsewhere (which is only ever an approximation of the real NFL
    schedule), derive it from whichever games the prediction pipeline most
    recently generated *is_current* predictions for — that's always
    exactly in sync with what the backend actually considers "this week",
    however it got computed. Falls back to the calendar estimate only if
    there are no current predictions yet (e.g. a brand new deploy before
    the first prediction run).
    """
    current_week = (
        db.query(Game.season, Game.week)
        .join(Prediction, Prediction.game_id == Game.id)
        .filter(Prediction.is_current.is_(True))
        .order_by(Game.season.desc(), Game.week.desc())
        .first()
    )

    if current_week:
        season, week = current_week
    else:
        season = _current_season()
        week = _current_nfl_week(season)

    games = (
        db.query(Game.id)
        .filter(Game.season == season, Game.week == week)
        .all()
    )
    return [g[0] for g in games]


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds (e.g. 4.5) to American odds (e.g. +350)."""
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def _build_play_of_week(
    db: Session,
    game_ids: List[int],
    target_low: float = 4.0,
    target_high: float = 6.0,
    max_legs: int = 5,
) -> Optional[PlayOfTheWeek]:
    """
    Picks the model's best-edge legs across every prop type and combines them
    into a parlay whose combined American odds land roughly between +300 and
    +500 (decimal 4.0-6.0), favoring the fewest legs that get there.
    """
    import itertools

    cards = _build_prop_cards(db, game_ids)
    if not cards:
        return None

    candidates = []
    for c in cards:
        if c.prop_type == "anytime_td":
            if c.over_probability is None:
                continue
            side, prob = "yes", c.over_probability
            edge = c.model_edge if c.model_edge is not None else 0.0
        else:
            if c.model_edge is None or c.over_probability is None or c.under_probability is None:
                continue
            if c.model_edge >= 0:
                side, prob = "over", c.over_probability
                edge = c.model_edge
            else:
                side, prob = "under", c.under_probability
                market_under = _market_implied_prob(c.under_odds) if c.under_odds is not None else 0.5
                edge = prob - market_under

        # Skip near-locks / near-coinflips and anything without real model edge.
        # Also skip legs whose probability is unrealistically lopsided (e.g. >90%) —
        # those are almost always an artifact of this app's synthetic "Consensus"
        # line generator inventing a line with no real sportsbook backing it,
        # not a genuine mispriced bet.
        if prob is None or prob < 0.15 or prob > 0.72 or edge is None or edge < 0.02:
            continue
        # Never build the parlay out of legs the play rating flags as not
        # optimal (injury designation, volatile role far from the book, ...),
        # or legs whose preferred side disagrees with the one picked here.
        if c.play_rating == "not_optimal" or (c.play_side and c.play_side != side):
            continue

        candidates.append({"card": c, "side": side, "prob": prob, "edge": edge, "decimal": 1.0 / prob})

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["edge"], reverse=True)

    # One leg per player, to keep the parlay diversified across the slate
    seen_players = set()
    pool = []
    for cand in candidates:
        pid = cand["card"].player.id
        if pid in seen_players:
            continue
        seen_players.add(pid)
        pool.append(cand)
        if len(pool) >= 20:
            break

    if len(pool) < 2:
        return None

    best_combo = None
    best_score = None
    for size in range(2, min(max_legs, len(pool)) + 1):
        found_in_range = False
        for combo in itertools.combinations(pool, size):
            dec = 1.0
            for leg in combo:
                dec *= leg["decimal"]
            if target_low <= dec <= target_high:
                total_edge = sum(l["edge"] for l in combo)
                score = (0, -total_edge, size)
                found_in_range = True
            else:
                dist = min(abs(dec - target_low), abs(dec - target_high))
                score = (1, dist, size)
            if best_score is None or score < best_score:
                best_score = score
                best_combo = combo
        if found_in_range:
            # Prefer the smallest leg count that lands inside the target range
            break

    if best_combo is None:
        return None

    legs: List[ParlayLeg] = []
    combined_decimal = 1.0
    combined_prob = 1.0
    for leg in best_combo:
        c = leg["card"]
        combined_decimal *= leg["decimal"]
        combined_prob *= leg["prob"]
        legs.append(
            ParlayLeg(
                player=c.player,
                team=c.team,
                opponent=c.opponent,
                prop_type=c.prop_type,
                pick=leg["side"],
                line=c.market_line,
                projection=c.projection,
                model_probability=round(leg["prob"], 4),
                american_odds=_decimal_to_american(leg["decimal"]),
                model_edge=round(leg["edge"], 4),
                game_week=c.game.week if c.game else None,
                play_rating=c.play_rating,
                play_summary=c.play_summary,
            )
        )

    return PlayOfTheWeek(
        legs=legs,
        combined_probability=round(combined_prob, 4),
        combined_american_odds=_decimal_to_american(combined_decimal),
        combined_decimal_odds=round(combined_decimal, 3),
        leg_count=len(legs),
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/props/today", response_model=PropList)
def get_props_today(
    prop_type: Optional[str] = Query(None),
    sort_by: str = Query("edge", description="Sort field: projection, edge, diff"),
    limit: int = Query(2000, le=5000),
    db: Session = Depends(get_db),
):
    gids = _get_today_game_ids(db)
    if not gids:
        return PropList(props=[], count=0, disclaimer=DISCLAIMER)

    p_filter = [prop_type] if prop_type else None
    cards = _build_prop_cards(db, gids, p_filter)

    if sort_by == "projection":
        cards.sort(key=lambda c: (c.projection or 0.0), reverse=True)
    elif sort_by == "diff":
        cards.sort(key=lambda c: abs(c.line_difference or 0.0), reverse=True)
    else:
        cards.sort(key=lambda c: abs(c.model_edge or 0.0), reverse=True)

    return PropList(props=cards[:limit], count=len(cards), disclaimer=DISCLAIMER)


@router.get("/props/play-of-the-week", response_model=PlayOfTheWeek)
def get_play_of_the_week(db: Session = Depends(get_db)):
    """Model's best-edge legs combined into a parlay targeting +300 to +500 odds."""
    gids = _get_today_game_ids(db)
    if not gids:
        raise HTTPException(status_code=404, detail="No games found for the current week")

    result = _build_play_of_week(db, gids)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Could not build a play of the week in the target odds range from today's props",
        )
    return result


@router.get("/props/passing", response_model=PropList)
def get_passing_props(limit: int = Query(100), db: Session = Depends(get_db)):
    gids = _get_today_game_ids(db)
    cards = _build_prop_cards(db, gids, ["passing_yards", "passing_tds"])
    cards.sort(key=lambda c: (c.projection or 0.0), reverse=True)
    return PropList(props=cards[:limit], count=len(cards), disclaimer=DISCLAIMER)


@router.get("/props/rushing", response_model=PropList)
def get_rushing_props(limit: int = Query(100), db: Session = Depends(get_db)):
    gids = _get_today_game_ids(db)
    cards = _build_prop_cards(db, gids, ["rushing_yards", "rushing_attempts"])
    cards.sort(key=lambda c: (c.projection or 0.0), reverse=True)
    return PropList(props=cards[:limit], count=len(cards), disclaimer=DISCLAIMER)


@router.get("/props/receiving", response_model=PropList)
def get_receiving_props(limit: int = Query(100), db: Session = Depends(get_db)):
    gids = _get_today_game_ids(db)
    cards = _build_prop_cards(db, gids, ["receiving_yards", "receptions"])
    cards.sort(key=lambda c: (c.projection or 0.0), reverse=True)
    return PropList(props=cards[:limit], count=len(cards), disclaimer=DISCLAIMER)


@router.post("/props/refresh-odds")
def refresh_odds(
    max_events: int = Query(16, ge=1, le=16),
    x_admin_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Trigger a live sync of sportsbook lines from The Odds API.

    Each call spends ~6 Odds API credits per game (~96 for a full week), so it
    requires the X-Admin-Key header to match the ADMIN_KEY setting; with no
    ADMIN_KEY configured it is disabled. It used to be open to anyone, and one
    request could drain the month's credit budget.
    """
    import secrets
    from backend.config import settings
    from backend.ingestion.odds import ingest_market_lines

    if not settings.admin_key or not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(status_code=403, detail="Admin key required.")
    count = ingest_market_lines(db, max_events=max_events)
    return {"status": "success", "lines_upserted": count}

