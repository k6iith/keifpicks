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
    except Exception as exc:  
