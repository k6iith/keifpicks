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

        # Replace any earlier score for this prop/week rather than appending
        # another row. Every run re-scores every final game, so appending left
        # duplicate rows per week (which also skewed the degradation baseline),
        # and a week first scored after only its Thursday game never got
        # updated once the rest of its games finished.
        db.query(ModelPerformance).filter(
            ModelPerformance.model_version == "1.0",
            ModelPerformance.prop_type == prop_type,
            ModelPerformance.season == season,
            ModelPerformance.week == week,
        ).delete(synchronize_session=False)

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

    alerts = check_performance_degradation(db)
    if alerts:
        results["degradation_alerts"] = alerts

    return results


def check_performance_degradation(
    db: Session,
    lookback_weeks: int = 4,
    degradation_threshold: float = 0.15,
    min_predictions: int = 30,
) -> List[Dict[str, Any]]:
    """
    Flag any prop whose most-recently-scored week is meaningfully worse
    than its own trailing average from the lookback_weeks before it.

    This is a different safety net than retrain.py's guardrail: that one
    only fires the instant a *new* model is trained and compares against
    the model it would replace. It says nothing about a model that's
    already live and has just started drifting worse week over week (a
    league-wide scheme change, an injury wave skewing usage, stale
    calibration, etc.) without ever being retrained. This runs every time
    score_completed_games() scores a newly-finished week, so drift in the
    live model gets caught and logged even between retraining runs.

    A week is only judged once it has enough scored predictions: at least
    min_predictions, and at least half the typical count of the baseline
    weeks. Otherwise a week scored after just its Thursday game (e.g. 5
    rushing predictions, one of them a 194-yard outlier) gets compared
    against full 120-prediction weeks and raises a false alarm. The
    baseline is weighted by each week's prediction count for the same
    reason, and uses only the newest row per (season, week).
    """
    prop_types = [r[0] for r in db.query(ModelPerformance.prop_type).distinct().all()]
    alerts: List[Dict[str, Any]] = []

    for prop_type in prop_types:
        metric = "brier_score" if prop_type == "anytime_td" else "mae"

        all_rows = (
            db.query(ModelPerformance)
            .filter(ModelPerformance.prop_type == prop_type)
            .order_by(
                ModelPerformance.season.desc(),
                ModelPerformance.week.desc(),
                ModelPerformance.evaluated_at.desc(),
                ModelPerformance.id.desc(),
            )
            .all()
        )

        # Newest row per (season, week), in newest-week-first order, keeping
        # only weeks that actually have this metric.
        rows = []
        seen = set()
        for r in all_rows:
            key = (r.season, r.week)
            if key in seen:
                continue
            seen.add(key)
            if getattr(r, metric) is not None and (r.n_predictions or 0) > 0:
                rows.append(r)
            if len(rows) >= lookback_weeks + 1:
                break

        if len(rows) < 2:
            continue

        latest_row, baseline_rows = rows[0], rows[1:]
        baseline_n = [r.n_predictions for r in baseline_rows]
        required_n = max(min_predictions, int(0.5 * float(np.median(baseline_n))))
        if latest_row.n_predictions < required_n:
            logger.info(
                "Skipping degradation check for %s season %d week %d: only %d scored "
                "predictions so far (need %d).",
                prop_type, latest_row.season, latest_row.week,
                latest_row.n_predictions, required_n,
            )
            continue

        latest = float(getattr(latest_row, metric))
        baseline_avg = float(np.average(
            [getattr(r, metric) for r in baseline_rows], weights=baseline_n
        ))
        if baseline_avg <= 0:
            continue

        pct_change = (latest - baseline_avg) / baseline_avg
        if pct_change > degradation_threshold:
            alert = {
                "prop_type": prop_type,
                "metric": metric,
                "latest": round(latest, 4),
                "trailing_baseline": round(baseline_avg, 4),
                "pct_worse": round(pct_change * 100, 1),
                "season": latest_row.season,
                "week": latest_row.week,
                "n_predictions": latest_row.n_predictions,
                "lookback_weeks": len(baseline_rows),
            }
            alerts.append(alert)
            logger.warning(
                "Performance degradation detected for %s (season %d week %d, n=%d): "
                "latest %s=%.4f is %.1f%% worse than the trailing %d-week average of %.4f.",
                prop_type, latest_row.season, latest_row.week, latest_row.n_predictions,
                metric, latest, pct_change * 100, len(baseline_rows), baseline_avg,
            )

    return alerts


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
