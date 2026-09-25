"""
PROPCAST – ESPN Injury Ingestion
Fetches current NFL injury reports from ESPN's unofficial public API.
No API key required. Handles failures gracefully — never raises.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from backend.db.models import Injury, Player

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

# Request timeout in seconds
_HTTP_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_espn_injuries() -> list[dict]:
    """
    GET the ESPN unofficial injuries endpoint and parse the response.

    Returns
    -------
    list[dict]
        Each dict contains:
        {
          'player_id_espn': str | None,
          'player_name': str,
          'team_abbr': str,
          'status': str,          # e.g. 'Questionable', 'Out', 'IR'
          'injury_type': str,     # body part / description
          'source': 'ESPN',
          'fetched_at': datetime,
        }
        Returns an empty list if the API is unavailable or returns an error.
    """
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(ESPN_INJURIES_URL)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "ESPN injuries API returned HTTP %s — skipping injury ingestion.",
            exc.response.status_code,
        )
        return []
    except httpx.RequestError as exc:
        logger.warning(
            "ESPN injuries API request failed (%s) — skipping injury ingestion.",
            type(exc).__name__,
        )
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unexpected error fetching ESPN injuries (%s: %s) — skipping.",
            type(exc).__name__,
            exc,
        )
        return []

    records: list[dict] = []
    fetched_at = datetime.utcnow()

    try:
        # ESPN response structure: {"injuries": [{"team": {...}, "injuries": [...]}]}
        team_groups = data.get("injuries", [])
        for team_group in team_groups:
            team_info = team_group.get("team", {})
            # ESPN uses short abbreviation like "KC", "SF" etc.
            team_abbr: str = team_info.get("abbreviation", "")

            for injury_entry in team_group.get("injuries", []):
                athlete = injury_entry.get("athlete", {})
                player_name: str = athlete.get("displayName", athlete.get("fullName", ""))
                player_id_espn: Optional[str] = athlete.get("id") or None

                # Status object
                status_obj = injury_entry.get("status", "")
                if isinstance(status_obj, dict):
                    status_str = status_obj.get("type", {}).get("description", "")
                else:
                    status_str = str(status_obj)

                # Injury details
                injury_type = injury_entry.get("details", {}).get("type", "") or ""
                detail_str = injury_entry.get("details", {}).get("detail", "") or ""
                injury_description = f"{injury_type} – {detail_str}".strip(" –") or injury_type

                if not player_name:
                    continue

                records.append(
                    {
                        "player_id_espn": str(player_id_espn) if player_id_espn else None,
                        "player_name": player_name,
                        "team_abbr": team_abbr,
                        "status": status_str,
                        "injury_type": injury_description,
                        "source": "ESPN",
                        "fetched_at": fetched_at,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error parsing ESPN injury response: %s", exc)
        return []

    logger.info("ESPN injuries fetched: %d player records.", len(records))
    return records


def _find_player(db: Session, player_name: str, team_abbr: str) -> Optional[Player]:
    """
    Attempt to match a player by full_name + team abbreviation.
    Falls back to name-only match if team match fails.
    """
    from backend.db.models import Team

    # Strict: name + team
    player = (
        db.query(Player)
        .join(Player.team)
        .filter(
            Player.full_name == player_name,
            Team.abbreviation == team_abbr,
        )
        .first()
    )
    if player:
        return player

    # Fallback: name only (handles team changes mid-season)
    player = db.query(Player).filter(Player.full_name == player_name).first()
    return player


def ingest_injuries(db: Session) -> int:
    """
    Fetch ESPN injury data and upsert Injury records into the database.

    Matching strategy:
      1. Try player_name + team_abbr exact match.
      2. Fallback to player_name only.
      3. If no match found, skip (we never create phantom players).

    Parameters
    ----------
    db : Session
        Active SQLAlchemy session.

    Returns
    -------
    int
        Number of records successfully upserted (0 on any failure).
    """
    raw_records = fetch_espn_injuries()
    if not raw_records:
        return 0

    count = 0
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        for rec in raw_records:
            player = _find_player(db, rec["player_name"], rec["team_abbr"])
            if not player:
                logger.debug(
                    "No player match for '%s' (%s) — skipping injury record.",
                    rec["player_name"],
                    rec["team_abbr"],
                )
                continue

            # Upsert: one injury record per player per day (report_date)
            existing: Optional[Injury] = (
                db.query(Injury)
                .filter(
                    Injury.player_id == player.id,
                    Injury.report_date == today,
                )
                .first()
            )

            if existing:
                existing.game_status = rec["status"]
                existing.injury_description = rec["injury_type"]
                existing.source = rec["source"]
            else:
                injury = Injury(
                    player_id=player.id,
                    report_date=today,
                    game_status=rec["status"],
                    injury_description=rec["injury_type"],
                    source=rec["source"],
                )
                db.add(injury)

            count += 1

        db.commit()
        logger.info("Injury ingestion complete: %d records upserted.", count)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Error during injury ingestion commit: %s", exc)
        return 0

    return count
