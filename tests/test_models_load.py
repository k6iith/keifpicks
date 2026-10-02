"""Every committed model file loads with the pinned scikit-learn and can predict."""
import json
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_TYPES = sorted(p.stem.rsplit("_v", 1)[0] for p in MODELS_DIR.glob("*_v1.0.json"))


def test_found_models():
    assert {"passing_yards", "rushing_yards", "receiving_yards", "receptions"} <= set(MODEL_TYPES)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_loads(model_type):
    from backend.models.trainer import load_model

    model, meta = load_model(model_type)
    assert model is not None and meta is not None
    estimator = model["model"] if isinstance(model, dict) else model
    assert hasattr(estimator, "predict")


def test_pinned_sklearn_matches_requirements():
    import sklearn

    reqs = (MODELS_DIR.parent / "requirements.txt").read_text()
    pin = next(line.split("==")[1].strip() for line in reqs.splitlines() if line.startswith("scikit-learn=="))
    assert sklearn.__version__ == pin


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_metadata_is_valid_json(model_type):
    meta = json.loads((MODELS_DIR / f"{model_type}_v1.0.json").read_text())
    assert isinstance(meta, dict) and meta
