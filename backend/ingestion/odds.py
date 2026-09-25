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
                     
