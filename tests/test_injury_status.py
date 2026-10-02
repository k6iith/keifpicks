"""Which injury designations hide a player."""
import pytest

from backend.ingestion.injury_status import is_out_status, ruled_out_player_games


@pytest.mark.parametrize("status", [
    "Out", "OUT", "Doubtful", "IR", "Injured Reserve", "IR - Designated to Return",
    "Out (Season)", "Physically Unable to Perform", "PUP", "Suspended",
])
def test_out(status):
    assert is_out_status(status)


@pytest.mark.parametrize("status", [None, "", "Questionable", "Active", "Probable", "Outside linebacker", "Day-To-Day"])
def test_not_out(status):
    assert not is_out_status(status)


def test_empty_game_list(db):
    assert ruled_out_player_games(db, []) == set()


def test_runs_against_database(db):
    result = ruled_out_player_games(db)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in result)
