"""The ESPN depth-chart sync never inserts the same ESPN player twice."""
import pandas as pd

import backend.features.builder as builder
import backend.ingestion.espn_sync as espn_sync
from backend.db.models import Player


class _Resp:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


def _fake_espn(athlete):
    """BAL only, with one athlete on the depth chart at TE."""
    slot = {"position": {"abbreviation": "TE"}, "athletes": [athlete]}

    def get(url, timeout=None):
        if url.endswith("/teams"):
            teams = [{"team": {"abbreviation": "BAL", "id": "33"}}]
            return _Resp({"sports": [{"leagues": [{"teams": teams}]}]})
        if url.endswith("/depthcharts"):
            return _Resp({"depthchart": [{"positions": {"te": slot}}]})
        return _Resp({}, status_code=404)

    return get


def test_renamed_espn_player_is_not_inserted_twice(db, monkeypatch):
    # An earlier sync created this player under the name ESPN used then.
    db.add(Player(gsis_id="ESPN_990000001", full_name="Zzyzxander Testplayer", position="TE", status="ACT"))
    db.commit()
    athlete = {"id": "990000001", "displayName": "Zzyz Testplayer", "jersey": "88"}
    monkeypatch.setattr(espn_sync.httpx, "get", _fake_espn(athlete))
    # Only the roster step is under test here, not prediction generation.
    monkeypatch.setattr(builder, "build_current_week_features", lambda *a, **k: pd.DataFrame())

    espn_sync.sync_espn_rosters_and_depth_charts()

    rows = db.query(Player).filter(Player.gsis_id == "ESPN_990000001").all()
    assert len(rows) == 1
