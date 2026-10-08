"""The current NFL week comes from the schedule, not the calendar."""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

import backend.ingestion.nfl_data as nfl_data


def _freeze(monkeypatch, local_now):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return local_now.replace(tzinfo=tz) if tz else local_now

    monkeypatch.setattr(nfl_data, "datetime", Frozen)


def _weeks(db, season):
    rows = db.execute(text("""
        SELECT week, MIN(kickoff_time), MAX(kickoff_time) FROM games
        WHERE season = :s AND kickoff_time IS NOT NULL AND game_type = 'REG'
        GROUP BY week ORDER BY week
    """), {"s": season}).fetchall()
    return [(w, datetime.fromisoformat(str(a)), datetime.fromisoformat(str(b))) for w, a, b in rows]


@pytest.mark.parametrize("week_index", [0, 1, 3, 8])
def test_week_before_first_kickoff(db, monkeypatch, week_index):
    weeks = _weeks(db, 2026)
    week, first_kickoff, _ = weeks[week_index]
    _freeze(monkeypatch, first_kickoff - timedelta(hours=1))
    assert nfl_data._current_nfl_week(2026, db) == week


def test_rolls_over_after_monday_night(db, monkeypatch):
    weeks = _weeks(db, 2026)
    (w3, _, last_kickoff), (w4, _, _) = weeks[2], weeks[3]
    _freeze(monkeypatch, last_kickoff + timedelta(hours=1))
    assert nfl_data._current_nfl_week(2026, db) == w3  # still playing
    _freeze(monkeypatch, last_kickoff + timedelta(hours=6))
    assert nfl_data._current_nfl_week(2026, db) == w4


def test_calendar_fallback_without_schedule(db):
    # A season with no games falls back to the calendar estimate without crashing.
    assert 1 <= nfl_data._current_nfl_week(1990, db) <= 22


def test_week_endpoint_matches_props_page(client, db):
    from backend.api.routes.props import _get_today_game_ids
    from backend.db.models import Game

    w = client.get("/api/week").json()
    ids = _get_today_game_ids(db)
    if ids:
        g = db.get(Game, ids[0])
        assert (g.season, g.week) == (w["season"], w["week"])
    games = client.get("/api/games/week").json()["games"]
    assert all(g["season"] == w["season"] and g["week"] == w["week"] for g in games)


def test_header_badge_is_not_hardcoded():
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "backend/static/index.html").read_text()
    assert "WEEK 3 LIVE" not in html
    assert "/api/week" in html
