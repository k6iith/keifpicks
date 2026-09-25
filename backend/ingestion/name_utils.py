"""
PROPCAST – Shared player name matching utilities.

Different ingestion sources spell player names differently — most notably,
ESPN's `displayName` includes generational suffixes ("James Cook III") while
our primary roster/odds sources often omit them ("James Cook"). A naive
`Player.full_name == name` match treats these as two different people,
which silently creates duplicate Player rows: one holding the real
sportsbook market lines, the other holding the current model predictions
(or vice versa). The prop card built from the "predictions" player then
never finds its real market line and falls back to a synthetic one.

These helpers give every ingestion source a single, consistent way to
resolve a scraped name to the correct existing Player row before ever
creating a new one. Critically, when a name matches more than one existing
Player row (i.e. we already have a duplicate on file), we prefer whichever
row actually holds the current model predictions — that's the row the
rest of the app is built around — rather than just the first string match,
so that ingestion doesn't keep re-attaching new data to a stale duplicate.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from sqlalchemy.orm import Session

from backend.db.models import Player, Prediction

# Generational / suffix tokens that commonly differ between sources.
_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", re.IGNORECASE)


def normalize_player_name(name: str) -> str:
    """Lowercase, strip whitespace, and drop a trailing generational suffix."""
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = _SUFFIX_RE.sub("", cleaned)
    # Collapse punctuation differences (periods, extra spaces) too.
    cleaned = re.sub(r"[.\s]+", " ", cleaned).strip()
    return cleaned.lower()


def _pick_best(db: Session, candidates: List[Player]) -> Optional[Player]:
    """
    Given multiple Player rows that all match the same (normalized) name,
    pick the one the rest of the app is actually using: the one with a
    current model prediction. Falls back to an active-roster player, then
    just the first candidate, so we always return something deterministic.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ids = [c.id for c in candidates]
    current_ids = {
        row[0]
        for row in db.query(Prediction.player_id)
        .filter(Prediction.player_id.in_(ids), Prediction.is_current.is_(True))
        .distinct()
        .all()
    }
    with_current = [c for c in candidates if c.id in current_ids]
    if len(with_current) == 1:
        return with_current[0]
    if len(with_current) > 1:
        candidates = with_current  # narrow further below

    active = [c for c in candidates if c.status == "ACT"]
    if len(active) == 1:
        return active[0]
    if active:
        candidates = active

    return candidates[0]


def find_player_by_name(
    db: Session,
    name: str,
    *,
    team_ids: Optional[Iterable[int]] = None,
) -> Optional[Player]:
    """
    Resolve a scraped player name to an existing Player row.

    Matches on a suffix-normalized name ("James Cook III" ~ "James Cook"),
    optionally scoped to team_ids first, then unscoped as a fallback. When
    a name resolves to more than one existing row (an existing duplicate),
    prefers the row that currently holds live model predictions — see
    `_pick_best`.

    Returns None if nothing matches — callers decide whether to create
    a new player in that case.
    """
    if not name:
        return None

    target = normalize_player_name(name)
    if not target:
        return None

    team_ids = list(team_ids) if team_ids else None

    def _candidates(scope_teams: bool) -> List[Player]:
        q = db.query(Player)
        if scope_teams and team_ids:
            q = q.filter(Player.team_id.in_(team_ids))
        return [c for c in q.all() if normalize_player_name(c.full_name) == target]

    if team_ids:
        scoped = _candidates(scope
