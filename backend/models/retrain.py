"""
PROPCAST – Weekly Model Retraining Orchestrator
================================================
Wraps trainer.py's per-prop training functions with:
  - A rolling feature matrix that grows with every newly-completed week,
    instead of the models being frozen at whatever data existed when
    someone last trained them by hand.
  - A validation guardrail: a newly-trained model only replaces the live
    one if it's not meaningfully worse (by MAE for yardage/count props, by
    Brier score for anytime_td) than the model it would replace. A bad
    retrain — a data glitch, a week with too little signal, an unlucky
    random seed — gets logged and discarded instead of silently going live.
  - Backups of every live model file before it's overwritten, so a bad
    retrain that *does* pass the guardrail (or a guardrail that itself
    turns out to be wrong) can still be rolled back by hand.

  - Persistence: every accepted model is also saved to the model_artifacts
    table. The hosted app's disk is wiped on every restart/redeploy, so
    without this each retrain would silently revert to the models committed
    in the repo. restore_models_from_db() puts them back at startup.

This intentionally does NOT change predictor.py's model-loading path: the
"live" model for each prop is always MODELS_DIR / f"{prop}_v{version}.pkl",
exactly as load_model() already expects. Only retrain.py's own backup copies
are versioned by timestamp.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sklearn
from sqlalchemy.orm import Session

from backend.db.models import ModelArtifact
from backend.features.builder import build_feature_matrix
from backend.ingestion.nfl_data import _current_season
from backend.models.trainer import MODELS_DIR, train_prop_regressor, train_td_model

logger = logging.getLogger(__name__)

BACKUPS_DIR = MODELS_DIR / "backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

# prop_type -> target column in the feature matrix (mirrors predictor.py's
# PROP_CONFIGS, minus anytime_td which is handled separately below).
_PROP_TARGET_MAP: Dict[str, str] = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "rushing_yards": "rushing_yards",
    "rushing_attempts": "carries",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
}

# How much worse (relative) a freshly-trained model is allowed to be before
# it gets rejected and the previous live model is kept instead. 5% is
# deliberately forgiving — the point is to catch a genuinely broken retrain,
# not to block every small week-to-week fluctuation.
_REJECTION_TOLERANCE = 0.05


def _backup_live_files(prop_type: str, version: str, extra_suffixes: List[str] | None = None) -> Dict[str, Path]:
    """
    Copy whatever live .pkl/.json (and any extra files, e.g. the anytime_td
    calibrator) currently exist for this prop to BACKUPS_DIR with a
    timestamp, before they're about to be overwritten. Returns a map of
    {original_path: backup_path} for whatever actually existed.
    """
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffixes = ["pkl", "json"] + (extra_suffixes or [])
    backed_up: Dict[str, Path] = {}

    for suffix, filename in [
        (s, f"{prop_type}_v{version}.{s}" if s in ("pkl", "json") else f"{prop_type}_{s}_v{version}.pkl")
        for s in suffixes
    ]:
        live_path = MODELS_DIR / filename
        if not live_path.exists():
            continue
        backup_path = BACKUPS_DIR / f"{live_path.stem}_{stamp}{live_path.suffix}"
        shutil.copy2(live_path, backup_path)
        backed_up[str(live_path)] = backup_path

    return backed_up


def _restore_backup(backed_up: Dict[str, Path]) -> None:
    """Undo a rejected retrain by copying each backup back over the live file."""
    for original, backup in backed_up.items():
        shutil.copy2(backup, original)


def _live_files(prop_type: str, version: str) -> List[Path]:
    names = [f"{prop_type}_v{version}.pkl", f"{prop_type}_v{version}.json"]
    if prop_type == "anytime_td":
        names.append(f"anytime_td_calibrator_v{version}.pkl")
    return [MODELS_DIR / n for n in names]


def _save_models_to_db(db: Session, prop_types: List[str], version: str) -> None:
    """Store the live files of each newly accepted model in model_artifacts."""
    for prop_type in prop_types:
        meta = _read_live_meta(prop_type, version) or {}
        trained_at = datetime.fromisoformat(meta["trained_at"]) if meta.get("trained_at") else None
        for path in _live_files(prop_type, version):
            if not path.exists():
                continue
            row = db.get(ModelArtifact, path.name) or ModelArtifact(filename=path.name)
            row.content = path.read_bytes()
            row.sklearn_version = sklearn.__version__
            row.trained_at = trained_at
            row.saved_at = datetime.utcnow()
            db.merge(row)
    db.commit()


def restore_models_from_db(db: Session, version: str = "1.0") -> List[str]:
    """
    Write any model saved by a past retrain back to MODELS_DIR, if it's newer
    than the file on disk (the disk copy is the one committed to the repo
    after a restart or redeploy). Models pickled by a different scikit-learn
    version are skipped: they can't be loaded. Returns the props restored.
    """
    rows = db.query(ModelArtifact).all()
    by_name = {r.filename: r for r in rows}
    restored: List[str] = []
    for prop_type in list(_PROP_TARGET_MAP) + ["anytime_td"]:
        files = _live_files(prop_type, version)
        saved = [by_name.get(p.name) for p in files]
        if any(r is None for r in saved[:2]):
            continue
        if any(r is not None and r.sklearn_version != sklearn.__version__ for r in saved):
            logger.warning(
                "Not restoring saved %s model: pickled with scikit-learn %s, running %s.",
                prop_type, saved[0].sklearn_version, sklearn.__version__,
            )
            continue
        live_meta = _read_live_meta(prop_type, version) or {}
        live_trained = datetime.fromisoformat(live_meta["trained_at"]) if live_meta.get("trained_at") else None
        saved_trained = saved[1].trained_at
        if saved_trained is None or (live_trained is not None and saved_trained <= live_trained):
            continue
        for path, row in zip(files, saved):
            if row is not None:
                path.write_bytes(row.content)
        restored.append(prop_type)
    if restored:
        logger.info("Restored retrained models from the database: %s", restored)
    return restored


def latest_model_trained_at(version: str = "1.0") -> datetime | None:
    """Most recent trained_at across the live model files, or None."""
    stamps = []
    for prop_type in list(_PROP_TARGET_MAP) + ["anytime_td"]:
        meta = _read_live_meta(prop_type, version) or {}
        if meta.get("trained_at"):
            stamps.append(datetime.fromisoformat(meta["trained_at"]))
    return max(stamps) if stamps else None


def _read_live_meta(prop_type: str, version: str) -> Dict[str, Any] | None:
    meta_path = MODELS_DIR / f"{prop_type}_v{version}.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def retrain_all_models(db: Session, version: str = "1.0") -> Dict[str, Any]:
    """
    Rebuild the feature matrix from every completed game on file (through
    the current, partially-played season) and retrain every prop model
    against it, subject to the validation guardrail described above.

    Returns a summary dict keyed by prop_type, each with at least a
    "status" of "accepted" | "rejected" | "insufficient_data" | "error",
    plus whatever MAE/Brier numbers were involved in that decision.
    """
    season = _current_season()
    seasons = list(range(2022, season + 1))

    logger.info("Retraining: building feature matrix for seasons %s...", seasons)
    try:
        df = build_feature_matrix(db, seasons)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retraining aborted: failed to build feature matrix: %s", exc)
        return {"status": "error", "error": str(exc)}

    if df.empty:
        logger.warning("Retraining aborted: feature matrix is empty.")
        return {"status": "INSUFFICIENT_DATA"}

    results: Dict[str, Any] = {}

    for prop_type, target_col in _PROP_TARGET_MAP.items():
        try:
            old_meta = _read_live_meta(prop_type, version)
            old_mae = old_meta.get("mae") if old_meta else None

            backed_up = _backup_live_files(prop_type, version)

            train_result = train_prop_regressor(df, target_col, prop_type, version=version)

            if train_result.get("status") == "INSUFFICIENT_DATA":
                results[prop_type] = {"status": "insufficient_data"}
                continue

            new_mae = train_result["mae"]

            if old_mae is not None and new_mae > old_mae * (1.0 + _REJECTION_TOLERANCE):
                _restore_backup(backed_up)
                logger.warning(
                    "Rejected retrain for %s: new MAE %.2f is worse than live MAE %.2f "
                    "(tolerance %.0f%%) — kept previous model.",
                    prop_type, new_mae, old_mae, _REJECTION_TOLERANCE * 100,
                )
                results[prop_type] = {
                    "status": "rejected", "old_mae": old_mae, "new_mae": new_mae,
                }
            else:
                logger.info(
                    "Accepted retrain for %s: new MAE %.2f (previous %s).",
                    prop_type, new_mae, f"{old_mae:.2f}" if old_mae is not None else "none",
                )
                results[prop_type] = {
                    "status": "accepted", "old_mae": old_mae, "new_mae": new_mae,
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retraining %s failed: %s", prop_type, exc)
            results[prop_type] = {"status": "error", "error": str(exc)}

    # anytime_td: separate model type (classifier + calibrator), scored by
    # Brier score instead of MAE — lower is better either way.
    try:
        old_meta = _read_live_meta("anytime_td", version)
        old_brier = old_meta.get("brier_score") if old_meta else None

        backed_up = _backup_live_files("anytime_td", version, extra_suffixes=["calibrator"])

        train_result = train_td_model(df, version=version)

        if train_result.get("status") == "INSUFFICIENT_DATA":
            results["anytime_td"] = {"status": "insufficient_data"}
        else:
            new_brier = train_result["brier_score"]
            if old_brier is not None and new_brier > old_brier * (1.0 + _REJECTION_TOLERANCE):
                _restore_backup(backed_up)
                logger.warning(
                    "Rejected retrain for anytime_td: new Brier %.4f is worse than live "
                    "Brier %.4f (tolerance %.0f%%) — kept previous model.",
                    new_brier, old_brier, _REJECTION_TOLERANCE * 100,
                )
                results["anytime_td"] = {
                    "status": "rejected", "old_brier": old_brier, "new_brier": new_brier,
                }
            else:
                logger.info(
                    "Accepted retrain for anytime_td: new Brier %.4f (previous %s).",
                    new_brier, f"{old_brier:.4f}" if old_brier is not None else "none",
                )
                results["anytime_td"] = {
                    "status": "accepted", "old_brier": old_brier, "new_brier": new_brier,
                }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retraining anytime_td failed: %s", exc)
        results["anytime_td"] = {"status": "error", "error": str(exc)}

    accepted = [p for p, r in results.items() if r.get("status") == "accepted"]
    if accepted:
        try:
            _save_models_to_db(db, accepted, version)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.exception("Could not save retrained models to the database: %s", exc)
            results["persist_error"] = f"{type(exc).__name__}: {exc}"

    logger.info("Retraining complete: %s", results)
    return results
