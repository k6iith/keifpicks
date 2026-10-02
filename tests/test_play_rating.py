"""Optimal / lean / not optimal rules."""
from datetime import datetime

from backend.models.play_rating import PlayContext, rate_play


def _ctx():
    ctx = PlayContext(None, [], [])  # nothing to load
    ctx.games[1] = {"id": 1, "season": 2026, "week": 4, "home_team_id": 10, "away_team_id": 20,
                    "home_spread": None, "game_total": None, "kickoff": datetime(2026, 10, 4, 13)}
    ctx._by_player = {7: [
        {"kickoff": datetime(2026, 9, d, 13), "rushing_yards": y, "carries": 15}
        for d, y in zip(range(1, 29, 3), [60, 72, 55, 68, 80, 64, 70, 66, 75])
    ]}
    return ctx


def _rate(ctx=None, **kw):
    args = dict(player_id=7, player_name="Test Back", position="RB", team_id=10, game_id=1,
                prop_type="rushing_yards", projection=78.0, market_line=66.5, has_real_line=True,
                over_probability=0.62)
    args.update(kw)
    return rate_play(ctx or _ctx(), **args)


def test_strong_play_with_real_line_is_optimal():
    r = _rate()
    assert r.play_rating == "optimal" and r.play_side == "over"


def test_under_side():
    r = _rate(projection=55.0, over_probability=0.35)
    assert r.play_side == "under" and r.play_rating == "optimal"


def test_no_real_line_is_never_optimal():
    assert _rate(has_real_line=False).play_rating != "optimal"


def test_coin_flip_is_not_optimal():
    assert _rate(over_probability=0.51).play_rating == "not_optimal"


def test_questionable_player_is_not_optimal():
    ctx = _ctx()
    ctx.questionable[(7, 1)] = "Questionable"
    assert _rate(ctx).play_rating == "not_optimal"


def test_far_from_market_is_not_optimal():
    assert _rate(projection=140.0).play_rating == "not_optimal"


def test_td_without_odds_is_not_optimal():
    r = _rate(prop_type="anytime_td", has_real_line=False, over_probability=0.7, market_over_prob=0.5)
    assert r.play_rating == "not_optimal"


def test_missing_probability_is_lean():
    assert _rate(over_probability=None).play_rating == "lean"
