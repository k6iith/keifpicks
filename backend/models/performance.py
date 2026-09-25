import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
from sqlalchemy.orm import Session

from backend.db.models import Prediction, PlayerGameStat, ModelPerformance, Game

logger = logging.getLogger(__name__)


def score_completed_games(db: Session) -> Dict[str, Any]:
    """
    Compare previous predictions against actual game results once status='final'.
    Calculates MAE, RMSE for continuous props and Brier/LogLoss for TD props.
    Persists to ModelPerformance.
    """
    final_games = db.query(Game).filter(Game.status == "final").all()
    if not final_games:
        return {"status": "NO_COMPLETED_GAMES"}

    game_ids = [g.id for g in final_games]
    preds = db.query(Prediction).filter(
        Prediction.game_id.in_(game_ids),
        Prediction.is_current == True
    ).all()

    if not preds:
        return {"status": "NO_PREDICTIONS_TO_SCORE"}

    # Bulk load all player game stats for these games into a hash map
    stats_rows = db.query(PlayerGameStat).filter(PlayerGameStat.game_id.in_(game_ids)).all()
    stats_map = {(s.player_id, s.game_id): s for s in stats_rows}

    scored_by_key: Dict[Tuple[str, int, int], List[Dict[str, float]]] = {}

    for p in preds:
        stat = stats_map.get((p.player_id, p.game_id))
        if not stat:
            continue

        actual = None
        if p.prop_type == "passing_yards":
            actual = float(stat.passing_yards or 0)
        elif p.prop_type == "rushing_yards":
            actual = float(stat.rushing_yards or 0)
        elif p.prop_type == "receiving_yards":
            actual = float(stat.receiving_yards or 0)
        elif p.prop_type == "receptions":
            actual = float(stat.receptions or 0)
        elif p.prop_type == "anytime_td":
            actual = 1.0 if ((stat.rushing_tds or 0) > 0 or (stat.receiving_tds or 0) > 0) else 0.0

        if actual is not None:
            key = (p.prop_type, stat.season, stat.week)
            scored_by_key.setdefault(key, []).append({
                "pred": float(p.projection),
                "actual": actual
            })

    results = {}
    now = datetime.utcnow()

    for (prop_type, season, week), pairs in scored_by_key.items():
        if not pairs:
            continue

        y_pred = np.array([x["pred"] for x in pairs])
        y_true = np.array([x["actual"] for x in pairs])

        record = ModelPerformance(
            model_version="1.0",
            prop_type=prop_type,
            season=season,
            week=week,
            n_predictions=len(pairs),
            evaluated_at=now
        )

        if prop_type == "anytime_td":
            record.brier_score = float(brier_score_loss(y_true, y_pred))
            record.log_loss = float(log_loss(y_true, np.clip(y_pred, 1e-5, 1 - 1e-5)))
        else:
            record.mae = float(mean_absolute_error(y_true, y_pred))
            record.rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        db.add(record)
        results[f"{prop_type}_S{season}W{week}"] = {
            "n_predictions": len(pairs),
            "mae": record.mae,
            "rmse": record.rmse,
            "brier_score": record.brier_score
        }

    db.commit()
    return results


def get_calibration_data(db: Session, prop_type: str = "anytime_td") -> List[Dict[str, Any]]:
    """Compute calibration bucket data for UI visualization."""
    preds = db.query(Prediction).filter(
        Prediction.prop_type == prop_type,
        Prediction.is_current == True
    ).all()

    if not preds:
        return []

    buckets = np.linspace(0, 1.0, 11)
    results = []

    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i+1]
        results.append({
            "bucket": f"{int(low*100)}-{int(high*100)}%",
            "predicted_mid": round((low + high) / 2, 2),
            "sample_count": 0,
            "actual_rate": 0.0
        })

    return results
