"""Weekly retraining: learned calibration, and retrained models surviving a restart."""
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from backend.models import retrain, trainer


def test_scipy_is_a_declared_requirement():
    # predictor.py imports scipy directly; relying on scikit-learn to pull it
    # in transitively is what the explicit requirement prevents.
    from pathlib import Path

    reqs = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().lower()
    assert any(line.startswith("scipy") for line in reqs.splitlines())


def test_calibration_ratio_learns_bias_and_shrinks_small_samples():
    preds = np.full(5000, 10.0)
    assert trainer._calibration_ratio(preds * 1.1, preds) == pytest.approx(1.1, abs=0.01)
    # 200 samples (QB-sized): only part of the measured +10% is applied.
    small = np.full(200, 10.0)
    r = trainer._calibration_ratio(small * 1.1, small)
    assert 1.0 < r < 1.05


def test_dispersion_recovers_known_spread():
    rng = np.random.default_rng(0)
    n = 20000
    preds = rng.uniform(5, 100, n)
    player_std = rng.uniform(5, 30, n)
    true_var = 50.0 + 0.09 * preds ** 2 + 0.4 * player_std ** 2
    y = preds + rng.normal(0, np.sqrt(true_var))
    d = trainer._fit_dispersion(preds, y, player_std)
    fitted = d["intercept"] + d["pred_sq"] * preds ** 2 + d["player_var"] * player_std ** 2
    assert np.mean(fitted) == pytest.approx(np.mean(true_var), rel=0.05)
    assert d["pred_sq"] == pytest.approx(0.09, rel=0.15)
    assert d["player_var"] == pytest.approx(0.4, rel=0.15)


def test_dispersion_needs_enough_samples():
    x = np.ones(10)
    assert trainer._fit_dispersion(x, x, x) is None


def test_predictor_has_no_hardcoded_calibration_table():
    import inspect

    from backend.models import predictor

    assert "prop_calibration_pcts" not in inspect.getsource(predictor)


@pytest.fixture()
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "MODELS_DIR", tmp_path)
    return tmp_path


def _write_model(models_dir, prop, trained_at, payload=b"model"):
    (models_dir / f"{prop}_v1.0.pkl").write_bytes(payload)
    (models_dir / f"{prop}_v1.0.json").write_text(json.dumps({"trained_at": trained_at.isoformat()}))


def test_retrained_models_are_restored_after_restart(db, models_dir):
    from backend.db.database import engine
    from backend.db.models import ModelArtifact

    ModelArtifact.__table__.create(bind=engine, checkfirst=True)
    now = datetime.utcnow()
    try:
        # A retrain saves the new model...
        _write_model(models_dir, "receptions", now, b"retrained")
        retrain._save_models_to_db(db, ["receptions"], "1.0")
        # ...then a restart puts the older committed file back on disk.
        _write_model(models_dir, "receptions", now - timedelta(days=10), b"committed")

        assert retrain.restore_models_from_db(db) == ["receptions"]
        assert (models_dir / "receptions_v1.0.pkl").read_bytes() == b"retrained"
        # Already current: nothing to do.
        assert retrain.restore_models_from_db(db) == []
    finally:
        db.query(ModelArtifact).delete()
        db.commit()


def test_models_from_another_sklearn_version_are_not_restored(db, models_dir):
    from backend.db.database import engine
    from backend.db.models import ModelArtifact

    ModelArtifact.__table__.create(bind=engine, checkfirst=True)
    now = datetime.utcnow()
    try:
        _write_model(models_dir, "receptions", now, b"retrained")
        retrain._save_models_to_db(db, ["receptions"], "1.0")
        for row in db.query(ModelArtifact).all():
            row.sklearn_version = "0.0.1"
        db.commit()
        _write_model(models_dir, "receptions", now - timedelta(days=10), b"committed")

        assert retrain.restore_models_from_db(db) == []
        assert (models_dir / "receptions_v1.0.pkl").read_bytes() == b"committed"
    finally:
        db.query(ModelArtifact).delete()
        db.commit()


def test_admin_retrain_requires_key(client):
    assert client.post("/api/admin/retrain").status_code == 403


def test_blend_falls_back_to_model_without_baseline():
    model = np.array([10.0, 20.0])
    baseline = np.array([30.0, np.nan])
    assert trainer._blend(model, baseline, 0.5).tolist() == [20.0, 20.0]


def test_blend_weight_is_learned_and_shrunk():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 100, 2000)
    good, bad = y + rng.normal(0, 2, y.size), y + rng.normal(0, 40, y.size)
    # Model clearly better -> leans model, but never all the way (shrunk toward 0.5).
    assert trainer._learn_blend_weight(y, good, bad) == pytest.approx(0.75, abs=0.03)
    assert trainer._learn_blend_weight(y, bad, good) == pytest.approx(0.25, abs=0.03)


def test_committed_models_beat_their_baseline():
    from backend.models.trainer import MODELS_DIR

    for prop in ["passing_yards", "rushing_yards", "rushing_attempts", "receiving_yards", "receptions"]:
        meta = json.loads((MODELS_DIR / f"{prop}_v1.0.json").read_text())
        assert meta["mae"] < meta["baseline_mae"], prop
        assert meta["blend"] and meta["dispersion"], prop


def test_starting_cb_absence_comes_from_snap_counts(db):
    from sqlalchemy import text

    from backend.features.builder import _cb1_absent_keys

    absent = _cb1_absent_keys(db)
    team_games = db.execute(text(
        "SELECT COUNT(*) FROM (SELECT DISTINCT season, week, team_abbr FROM cb_snap_counts)"
    )).scalar()
    # A starting corner sits out a few percent of games, not never and not always.
    assert 0.01 < len(absent) / team_games < 0.2
