"""
PROPCAST – Shared "is this player ruled out?" logic.

Used both by the feature builder (to drop ruled-out players from the week's
predictions and set the *_is_out features) and by the props API (to hide a
ruled-out player's props immediately, without waiting for the next
prediction regeneration).
"""
from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

# ESPN spells statuses out ("Injured Reserve", not "IR") and wording/case has
# shifted over time, so match on a normalized form rather than exact strings.
_OUT_STATUSES = (
    "out", "doubtful", "ir", "injured reserve", "physically unable to perform",
    "pup", "suspended", "suspension",
)


def is_out_status(status: Optional[str]) -> bool:
    # Whole-word match so e.g. "IR - Designated to Return" or "Out (Season)"
    # count, but an unrelated word that merely starts with "out" doesn't.
    s = " ".join((status or "").strip().lower().replace("-", " ").replace("(", " ").split())
    return any(s == p or s.startswith(p + " ") for p in _OUT_STATUSES)


def ruled_out_player_games(
    db: Session, game_ids: Optional[Iterable[int]] = None
) -> Set[Tuple[int, int]]:
    """
    (player_id, game_id) pairs whose MOST RECENT injury report for that game
    rules the player out. Only the latest report counts, so a player listed
    Out on Wednesday but cleared by Friday is not included.
    """
    params = {}
    where = "game_id IS NOT NULL"
    if game_ids is not None:
        game_ids = list(game_ids)
        if not game_ids:
            return set()
        placeholders = ", ".join(f":g{i}" for i in range(len(game_ids)))
        params = {f"g{i}": g for i, g in enumerate(game_ids)}
        where += f" AND game_id IN ({placeholders})"

    rows = db.execute(
        text(f"""
            SELECT player_id, game_id, game_status
            FROM injuries
            WHERE {where}
            ORDER BY player_id, game_id, report_date, id
        """),
        params,
    ).fetchall()

    latest = {}
    for player_id, game_id, status in rows:
        latest[(player_id, game_id)] = status  # later rows overwrite earlier
    return {key for key, status in latest.items() if is_out_status(status)}
