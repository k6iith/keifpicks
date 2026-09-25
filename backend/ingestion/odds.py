"""
PROPCAST – The Odds API Market Line Ingestion
Fetches NFL player prop lines from The Odds API.
Supports real lines from DraftKings, FanDuel, BetMGM, Caesars, etc.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.models import Game, MarketLine, Player, Team
from backend.ingestion.name_utils import find_player_by_name

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_HTTP_TIMEOUT = 25.0

# Map The Odds API market keys to PropType values
_MARKET_TO_PROP_TYPE: dict[str, str] = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
    "player_pass_tds": "passing_tds",
    "player_rush_attempts": "rushing_attempts",
}

DEFAULT_MARKETS = list(_MARKET_TO_PROP_TYPE.keys())


def _has_api_key() -> bool:
    return bool(settings.odds_api_key and settings.odds_api_key.strip())


def _safe_headers() -> dict:
    return {"Accept": "application/json"}


def fetch_nfl_events() -> list[dict]:
    """Fetch current NFL events from The Odds API."""
    if not _has_api_key():
        return []

    url = f"{ODDS_API_BASE}/sports/americanfootball_nfl/events"
    params = {
        "apiKey": settings.odds_api_key,
        "sport": "americanfootball_nfl",
    }

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(url, params=params, headers=_safe_headers())
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("Error fetching NFL events from Odds API: %s", exc)
        return []


def fetch_nfl_props(event_id: str, markets: Optional[list[str]] = None) -> list[dict]:
    """Fetch all player prop lines for a single NFL event."""
    if not _has_api_key():
        return []

    markets_to_fetch = markets or DEFAULT_MARKETS
    url = f"{ODDS_API_BASE}/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": settings.odds_api_key,
        "regions": "us",
        "markets": ",".join(markets_to_fetch),
        "oddsFormat": "american",
    }

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(url, params=params, headers=_safe_headers())
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Odds API error for event %s: %s", event_id, exc)
        return []

    records: list[dict] = []
    fetched_at = datetime.now(timezone.utc)

    try:
        bookmakers = data.get("bookmakers", [])
        for book in bookmakers:
            book_name: str = book.get("title", "Unknown")
            for market in book.get("markets", []):
                market_key: str = market.get("key", "")
                prop_type = _MARKET_TO_PROP_TYPE.get(market_key)
                if not prop_type:
                    continue

                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", outcome.get("name", ""))
                    name = outcome.get("name", "").lower()
                    price = outcome.get("price")
                    point = outcome.get("point")

                    existing_rec = next(
                        (
                            r for r in records
                            if r["event_id"] == event_id
                            and r["player_name"] == player_name
                            and r["prop_type"] == prop_type
                            and r["book"] == book_name
                        ),
                        None,
                    )

                    if existing_rec is None:
                        existing_rec = {
                            "event_id": event_id,
                            "player_name": player_name,
                            "prop_type": prop_type,
                            "line": point,
                            "over_odds": None,
                            "under_odds": None,
                            "book": book_name,
                            "fetched_at": fetched_at,
                        }
                        records.append(existing_rec)

                    if "over" in name:
                        existing_rec["over_odds"] = int(price) if price is not None else None
                        if point is not None:
                            existing_rec["line"] = point
                    elif "under" in name:
                        existing_rec["under_odds"] = int(price) if price is not None else None
                    elif "yes" in name:
                        existing_rec["over_odds"] = int(price) if price is not None else None
                        existing_rec["line"] = 0.5
                    elif "no" in name:
                        existing_rec["under_odds"] = int(price) if price is not None else None
                        existing_rec["line"] = 0.5

    except Exception as exc:
        logger.warning("Error parsing Odds API response for event %s: %s", event_id, exc)
        return []

    return records


def _match_game(db: Session, home_team_name: str, away_team_name: str) -> Optional[Game]:
    """Match an Odds API event to a DB Game row by team names."""
    home_mascot = home_team_name.split()[-1]
    away_mascot = away_team_name.split()[-1]

    # Search in upcoming scheduled games for 2026 week 3 or current active games
    games = (
        db.query(Game)
        .join(Game.home_team.of_type(Team))
        .filter(Game.season == 2026, Game.week == 3)
        .all()
    )

    for g in games:
        h_name = g.home_team.full_name if g.home_team else ""
        a_name = g.away_team.full_name if g.away_team else ""
        if home_mascot.lower() in h_name.lower() or away_mascot.lower() in a_name.lower():
            return g

    # Fallback to any recent/scheduled game
    return (
        db.query(Game)
        .join(Game.home_team.of_type(Team))
        .filter(Team.full_name.ilike(f"%{home_mascot}%"))
        .order_by(Game.kickoff_time.desc())
        .first()
    )


def _find_player(db: Session, player_name: str, game: Optional[Game] = None) -> Optional[Player]:
    """
    Look up player, prioritizing active roster player in the game's teams.

    Matches by exact name first, then falls back to a suffix-insensitive
    match ("James Cook" ~ "James Cook III") so that a scraped name spelled
    slightly differently than our roster source doesn't silently attach
    real sportsbook lines to the wrong duplicate Player row.
    """
    clean_name = player_name.strip()
    team_ids = [game.home_team_id, game.away_team_id] if game else None

    p = find_player_by_name(db, clean_name, team_ids=team_ids)
    if p:
        return p

    # Last resort: any active offensive player with this exact name.
    return (
        db.query(Player)
        .filter(
            Player.full_name == clean_name,
            Player.status == "ACT",
            Player.position.in_(["QB", "RB", "WR", "TE", "K"]),
        )
        .first()
    )


def ingest_market_lines(db: Session, max_events: int = 16) -> int:
    """
    Fetch and upsert real sportsbook MarketLine records from The Odds API.
    """
    if not _has_api_key():
        logger.info("Odds API key not configured, skipping market line ingestion.")
        return 0

    events = fetch_nfl_events()
    if not events:
        logger.info("No NFL events found from Odds API.")
        return 0

    count = 0
    events_processed = 0

    for ev in events:
        if events_processed >= max_events:
            break

        event_id = ev.get("id")
        home_team = ev.get("home_team", "")
        away_team = ev.get("away_team", "")

        game = _match_game(db, home_team, away_team)
        if not game:
            continue

        props = fetch_nfl_props(event_id)
        if not props:
            continue

        events_processed += 1

        for prop in props:
            player = _find_player(db, prop["player_name"], game)
            if not player:
                continue

            try:
                existing = (
                    db.query(MarketLine)
                    .filter(
                        MarketLine.player_id == player.id,
                        MarketLine.game_id == game.id,
                        MarketLine.prop_type == prop["prop_type"],
                        MarketLine.book == prop["book"],
                    )
                    .first()
                )

                if existing:
                    existing.line = prop["line"]
                    existing.over_odds = prop["over_odds"]
                    existing.under_odds = prop["under_odds"]
                    existing.fetched_at = prop["fetched_at"]
                else:
                    ml = MarketLine(
                        player_id=player.id,
                        game_id=game.id,
                        prop_type=prop["prop_type"],
                        line=prop["line"],
                        over_odds=prop["over_odds"],
                        under_odds=prop["under_odds"],
                        book=prop["book"],
                        fetched_at=prop["fetched_at"],
                    )
                    db.add(ml)

                count += 1
            except Exception as exc:
                logger.warning("Error upserting market line: %s", exc)
                continue

    try:
        db.commit()
        logger.info("Market line ingestion complete: %d records upserted across %d games.", count, events_processed)
    except Exception as exc:
        db.rollback()
        logger.error("Market line commit failed: %s", exc)
        return 0

    return count
