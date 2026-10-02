"""Sportsbook events attach to the right game (the week-4 'no real lines' bug)."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from backend.db.models import Game
from backend.ingestion.odds import _match_game


def _event_time(kickoff_local: datetime) -> str:
    local = kickoff_local.replace(tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")


def _games(db, weeks):
    return db.query(Game).filter(Game.season == 2026, Game.week.in_(weeks), Game.kickoff_time.isnot(None)).all()


def test_every_game_matches_itself(db):
    games = _games(db, [4, 5, 6, 7, 8])
    assert len(games) > 50
    for g in games:
        found = _match_game(db, g.home_team.full_name, g.away_team.full_name, _event_time(g.kickoff_time))
        assert found is not None and found.id == g.id, f"week {g.week} {g.away_team.full_name} @ {g.home_team.full_name}"


def test_nickname_and_swapped_sides(db):
    g = _games(db, [4])[0]
    nick_home, nick_away = g.home_team.full_name.split()[-1], g.away_team.full_name.split()[-1]
    assert _match_game(db, nick_home, nick_away, _event_time(g.kickoff_time)).id == g.id
    assert _match_game(db, g.away_team.full_name, g.home_team.full_name, _event_time(g.kickoff_time)).id == g.id


def test_no_match_far_from_kickoff(db):
    g = _games(db, [4])[0]
    far = g.kickoff_time + timedelta(days=60)
    found = _match_game(db, g.home_team.full_name, g.away_team.full_name, _event_time(far))
    assert found is None or found.id != g.id
