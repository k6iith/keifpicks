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

This intentionally does NOT change predictor.py's model-loading path: the
"live" model for each prop is always MODELS_DIR / f"{prop}_v{version}.pkl",
exactly as load_model() already expects. Only retrain.py's own backup copies
are versioned by timestamp.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from backend.features.builder import build_feature_matrix
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


def _current_season() -> int:
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


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

    logger.info("Retraining complete: %s", results)
    return results
