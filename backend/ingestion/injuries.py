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

from sqlalchemy import or_

from backend.db.models import Injury, Player, Game
from backend.ingestion.name_utils import find_player_by_name

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
          'team_full_name': str,
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

    # ESPN's response used to nest each team's info under a "team": {...}
    # sub-object with an "abbreviation" field. That's gone — each entry in
    # the top-level "injuries" array now just has "id"/"displayName"
    # directly on it (e.g. displayName="Arizona Cardinals"), with no
    # abbreviation anywhere in the payload at all. team_group.get("team",
    # {}) was silently returning {} for every team, so team_abbr was always
    # "" — which meant every injury record fed team-scoped player matching
    # nothing to scope by. Use the team's full display name instead (it
    # matches Team.full_name in our own DB) and resolve to an abbreviation
    # from there.
    try:
        team_groups = data.get("injuries", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error reading ESPN injury response: %s", exc)
        return []

    for team_group in team_groups:
        team_full_name: str = team_group.get("displayName", "")

        for injury_entry in team_group.get("injuries", []):
            # Per-entry try/except so one malformed record doesn't blank out
            # every other team's injuries along with it.
            try:
                athlete = injury_entry.get("athlete", {})
                player_name: str = athlete.get("displayName", athlete.get("fullName", ""))
                player_id_espn: Optional[str] = athlete.get("id") or None

                # Status object
                status_obj = injury_entry.get("status", "")
                if isinstance(status_obj, dict):
                    status_str = status_obj.get("type", {}).get("description", "")
                else:
                    status_str = str(status_obj) if status_obj else ""

                # Injury details
                details = injury_entry.get("details") or {}
                injury_type = details.get("type", "") or ""
                detail_str = details.get("detail", "") or ""
                injury_description = f"{injury_type} – {detail_str}".strip(" –") or injury_type

                if not player_name:
                    continue

                records.append(
                    {
                        "player_id_espn": str(player_id_espn) if player_id_espn else None,
                        "player_name": player_name,
                        "team_full_name": team_full_name,
                        "status": status_str,
                        "injury_type": injury_description,
                        "source": "ESPN",
                        "fetched_at": fetched_at,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping one malformed ESPN injury entry: %s", exc)
                continue

    logger.info("ESPN injuries fetched: %d player records.", len(records))
    return records


def _find_player(db: Session, player_name: str, team_full_name: str) -> Optional[Player]:
    """
    Attempt to match a player by full_name + team, falling back to a
    suffix-insensitive match ("Kenneth Walker" ~ "Kenneth Walker III") and
    then to a name-only match (handles team changes mid-season).

    ESPN's injury feed only gives us the team's full display name (e.g.
    "Arizona Cardinals"), not an abbreviation, so we resolve against
    Team.full_name here rather than Team.abbreviation.
    """
    from backend.db.models import Team

    team_ids = None
    if team_full_name:
        team = db.query(Team).filter(Team.full_name == team_full_name).first()
        if team:
            team_ids = [team.id]

    player = find_player_by_name(db, player_name, team_ids=team_ids)
    if player:
        return player

    # Fallback: name only, no team scoping at all.
    return db.query(Player).filter(Player.full_name == player_name).first()


def _find_upcoming_game(db: Session, player: Player) -> Optional[Game]:
    """
    Find the next not-yet-played game for a player's current team, so an
    injury report can be tied to the game it actually applies to.

    Without this, Injury rows never got a game_id at all, which meant
    add_injury_features()'s query (`WHERE i.game_id IS NOT NULL`) never
    matched a single row — every injury tag in the model was hardcoded to
    False regardless of who was actually hurt.
    """
    if not player.team_id:
        return None

    return (
        db.query(Game)
        .filter(
            or_(Game.home_team_id == player.team_id, Game.away_team_id == player.team_id),
            Game.status != "final",
        )
        .order_by(Game.kickoff_time.asc())
        .first()
    )


def ingest_injuries(db: Session) -> int:
    """
    Fetch ESPN injury data and upsert Injury records into the database.

    Matching strategy:
      1. Try player_name + team full name exact match.
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
            player = _find_player(db, rec["player_name"], rec["team_full_name"])
            if not player:
                logger.debug(
                    "No player match for '%s' (%s) — skipping injury record.",
                    rec["player_name"],
                    rec["team_full_name"],
                )
                continue

            upcoming_game = _find_upcoming_game(db, player)
            game_id = upcoming_game.id if upcoming_game else None

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
                existing.game_id = game_id
            else:
                injury = Injury(
                    player_id=player.id,
                    game_id=game_id,
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
