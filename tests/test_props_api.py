"""Prop endpoints respond, never show anytime TD, and rate every card."""
import pytest
from sqlalchemy import text


@pytest.mark.parametrize("path", ["/api/props/today", "/api/props/passing", "/api/props/rushing", "/api/props/receiving"])
def test_prop_endpoints(client, path):
    r = client.get(path)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= len(body["props"])
    assert all(p["prop_type"] != "anytime_td" for p in body["props"])


def test_play_of_the_week_responds(client):
    # 404 is the documented answer when too few plays are Optimal.
    assert client.get("/api/props/play-of-the-week").status_code in (200, 404)


def test_odds_credits_without_key(client):
    assert client.get("/api/odds/credits").json() == {"configured": False}


def test_cards_for_a_week_with_predictions(db):
    """Builds cards for the latest week that has predictions in the bundled database."""
    from backend.api.routes.props import _build_prop_cards

    season, week = db.execute(text("""
        SELECT g.season, g.week FROM predictions p JOIN games g ON g.id = p.game_id
        GROUP BY g.season, g.week ORDER BY g.season DESC, g.week DESC LIMIT 1
    """)).one()
    gids = [r[0] for r in db.execute(text("SELECT id FROM games WHERE season = :s AND week = :w"), {"s": season, "w": week})]
    cards = _build_prop_cards(db, gids, None)
    assert cards, f"no prop cards built for {season} week {week}"
    assert all(c.prop_type != "anytime_td" for c in cards)
    assert {c.play_rating for c in cards} <= {"optimal", "lean", "not_optimal"}
