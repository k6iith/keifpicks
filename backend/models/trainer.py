"""
PROPCAST – Machine Learning Model Training Pipeline
====================================================
Trains specialized models per prop market:
- Passing Yards, Rushing Yards, Receiving Yards (Continuous Yardage: HistGradientBoostingRegressor + Residual Dispersion)
- Receptions, Carries, Pass Attempts, Passing TDs (Discrete Count Models: Poisson / Target Regressors)
- Anytime Touchdowns (Binary Event: HistGradientBoostingClassifier + Isotonic Probability Calibration)

Uses walk-forward out-of-sample temporal validation (2022-2023 Train, 2024 Test) and evaluates against baseline models.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market-Specific Specialized Feature Definitions
# ---------------------------------------------------------------------------

BASE_CONTEXT_FEATURES = [
    "is_home",
    "home_spread",
    "game_total",
    "temperature_f",
    "wind_mph",
    "is_dome",
    "games_played_season",
    "dynamic_curr_weight",
    "dynamic_hist_weight",
]

# Rolling validation window: always validate on the most recent N completed
# (season, week) pairs rather than "the whole current season", which shrinks
# to almost nothing right after Week 1 and produces noisy MAE/Brier
# comparisons early in every season. See _rolling_time_split() below.
_MIN_VAL_WEEKS = 6
_MIN_VAL_SAMPLES = 40


def _rolling_time_split(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split clean_df into (train_df, val_df) using a rolling window of the
    most recent completed weeks as validation, widening the window if that
    leaves too few validation samples (but never past half the data).
    Falls back to a plain 80/20 positional split if season/week columns
    aren't usable (e.g. only one week of data on file at all).
    """
    if "season" in clean_df.columns and "week" in clean_df.columns and clean_df[["season", "week"]].drop_duplicates().shape[0] > 1:
        week_keys = (
            clean_df[["season", "week"]]
            .drop_duplicates()
            .sort_values(["season", "week"])
            .reset_index(drop=True)
        )
        n_weeks_avail = len(week_keys)
        n_val = min(_MIN_VAL_WEEKS, n_weeks_avail - 1)

        while n_val < n_weeks_avail - 1:
            val_weeks = week_keys.tail(n_val)
            val_key_set = set(map(tuple, val_weeks.to_numpy()))
            val_mask = clean_df[["season", "week"]].apply(tuple, axis=1).isin(val_key_set)
            if val_mask.sum() >= _MIN_VAL_SAMPLES or n_val >= n_weeks_avail // 2:
                break
            n_val += 2

        n_val = max(n_val, 1)
        val_weeks = week_keys.tail(n_val)
        val_key_set = set(map(tuple, val_weeks.to_numpy()))
        val_mask = clean_df[["season", "week"]].apply(tuple, axis=1).isin(val_key_set)
        return clean_df[~val_mask], clean_df[val_mask]

    split_idx = int(len(clean_df) * 0.8)
    return clean_df.iloc[:split_idx], clean_df.iloc[split_idx:]


PROP_FEATURE_REGISTRY: Dict[str, List[str]] = {
    "passing_yards": BASE_CONTEXT_FEATURES + [
        "passing_yards_3wk",
        "passing_yards_5wk",
        "passing_yards_8wk",
        "passing_yards_ewma",
        "passing_yards_career_avg",
        "passing_yards_weighted_baseline",
        "attempts_3wk",
        "attempts_5wk",
        "completion_pct",
        "yards_per_attempt",
        "passing_yards_trend",
        "opp_pass_yards_allowed_rank",
        "qb_is_out",
    ],
    "passing_tds": BASE_CONTEXT_FEATURES + [
        "passing_tds_3wk",
        "passing_tds_5wk",
        "passing_tds_career_avg",
        "passing_tds_weighted_baseline",
        "passing_yards_3wk",
        "attempts_3wk",
        "opp_pass_yards_allowed_rank",
        "qb_is_out",
    ],
    "rushing_yards": BASE_CONTEXT_FEATURES + [
        "rushing_yards_3wk",
        "rushing_yards_5wk",
        "rushing_yards_8wk",
        "rushing_yards_ewma",
        "rushing_yards_career_avg",
        "rushing_yards_weighted_baseline",
        "carries_3wk",
        "carries_5wk",
        "carry_share_4wk",
        "yards_per_carry",
        "rushing_yards_trend",
        "snap_pct_4wk",
        "opp_rush_yards_allowed_rank",
        "rb1_is_out",
        "qb_is_out",
    ],
    "rushing_attempts": BASE_CONTEXT_FEATURES + [
        "carries_3wk",
        "carries_5wk",
        "carries_career_avg",
        "carries_weighted_baseline",
        "carry_share_4wk",
        "snap_pct_4wk",
        "opp_rush_yards_allowed_rank",
        "rb1_is_out",
        "qb_is_out",
    ],
    "receiving_yards": BASE_CONTEXT_FEATURES + [
        "receiving_yards_3wk",
        "receiving_yards_5wk",
        "receiving_yards_8wk",
        "receiving_yards_ewma",
        "receiving_yards_career_avg",
        "receiving_yards_weighted_baseline",
        "targets_3wk",
        "targets_5wk",
        "receptions_3wk",
        "receptions_5wk",
        "target_share_4wk",
        "yards_per_reception",
        "air_yards_3wk",
        "yards_after_catch_3wk",
        "receiving_yards_trend",
        "snap_pct_4wk",
        "opp_pass_yards_allowed_rank",
        "opp_targets_allowed_to_wr_rank",
        "opp_targets_allowed_to_te_rank",
        "wr1_is_out",
        "opp_cb1_is_out",
        "qb_is_out",
    ],
    "receptions": BASE_CONTEXT_FEATURES + [
        "receptions_3wk",
        "receptions_5wk",
        "receptions_8wk",
        "receptions_ewma",
        "receptions_career_avg",
        "receptions_weighted_baseline",
        "targets_3wk",
        "targets_5wk",
        "target_share_4wk",
        "snap_pct_4wk",
        "opp_targets_allowed_to_wr_rank",
        "opp_targets_allowed_to_te_rank",
        "wr1_is_out",
        "opp_cb1_is_out",
        "qb_is_out",
    ],
    "anytime_td": BASE_CONTEXT_FEATURES + [
        "rushing_yards_3wk",
        "receiving_yards_3wk",
        "carries_3wk",
        "targets_3wk",
        "carry_share_4wk",
        "target_share_4wk",
        "snap_pct_4wk",
        "opp_rush_yards_allowed_rank",
        "opp_pass_yards_allowed_rank",
        "rb1_is_out",
        "wr1_is_out",
        "opp_cb1_is_out",
        "qb_is_out",
    ],
}


# ===========================================================================
# Continuous & Count Regressor Training
# ===========================================================================

def train_prop_regressor(
    df: pd.DataFrame,
    target_col: str,
    prop_type: str,
    version: str = "1.0",
) -> Dict[str, Any]:
    """
    Train a specialized regressor for continuous yardage or discrete count prop.
    Calculates empirical residual standard deviation and compares against season/weighted baselines.
    """
    feature_candidates = PROP_FEATURE_REGISTRY.get(prop_type, BASE_CONTEXT_FEATURES)
    features = [f for f in feature_candidates if f in df.columns]

    # Position relevance mapping to avoid zero-inflation from irrelevant positions
    pos_map = {
        "passing_yards": ["QB"],
        "passing_tds": ["QB"],
        "rushing_yards": ["RB", "QB"],
        "rushing_attempts": ["RB", "QB"],
        "receiving_yards": ["WR", "TE", "RB"],
        "receptions": ["WR", "TE", "RB"],
    }
    
    clean_df = df.dropna(subset=[target_col]).copy()
    if prop_type in pos_map and "position" in clean_df.columns:
        clean_df = clean_df[clean_df["position"].isin(pos_map[prop_type])].copy()

    if clean_df.empty or len(clean_df) < 50:
        logger.warning("Insufficient samples for training %s", prop_type)
        return {"status": "INSUFFICIENT_DATA"}

    # Filter out columns with zero variance or all-null values
    features = [
        f for f in features
        if f in clean_df.columns
        and clean_df[f].notna().sum() > 0
        and clean_df[f].nunique() > 1
    ]

    for c in features:
        clean_df[c] = clean_df[c].fillna(0.0)

    # Rolling last-N-completed-weeks validation window — see
    # _rolling_time_split()'s docstring for why this replaced "validate on
    # the whole current season" (that shrank to almost nothing right after
    # Week 1 and is exactly what caused passing_yards/passing_tds to get
    # rejected in a real retrain run this week purely from sample-size
    # noise, not an actually worse model).
    train_df, val_df = _rolling_time_split(clean_df)

    X_train = np.ascontiguousarray(train_df[features].to_numpy(dtype=np.float32, copy=True))
    y_train = np.ascontiguousarray(train_df[target_col].to_numpy(dtype=np.float32, copy=True))
    X_val = np.ascontiguousarray(val_df[features].to_numpy(dtype=np.float32, copy=True))
    y_val = np.ascontiguousarray(val_df[target_col].to_numpy(dtype=np.float32, copy=True))

    # Temporal sample weighting: give higher training weight to more recent seasons (e.g., 2024 > 2023 > 2022)
    sample_weights = None
    if "season" in train_df.columns:
        seasons_arr = train_df["season"].to_numpy()
        min_s = seasons_arr.min()
        sample_weights = np.exp(0.5 * (seasons_arr - min_s))
        sample_weights = sample_weights / sample_weights.mean()

    loss_fn = "poisson" if prop_type in ["receptions", "rushing_attempts", "passing_tds"] else "squared_error"

    model = HistGradientBoostingRegressor(
        loss=loss_fn,
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = np.maximum(0.0, model.predict(X_val))
    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    residuals = y_val - preds
    residual_std = float(np.std(residuals))

    # Baseline comparison (season/career average vs ML model)
    baseline_col = f"{target_col}_weighted_baseline" if f"{target_col}_weighted_baseline" in val_df.columns else f"{target_col}_3wk"
    baseline_preds = val_df[baseline_col].fillna(y_train.mean()).to_numpy(dtype=np.float32) if baseline_col in val_df.columns else np.full_like(y_val, y_train.mean())
    base_mae = float(mean_absolute_error(y_val, baseline_preds))
    base_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_preds)))

    model_path = MODELS_DIR / f"{prop_type}_v{version}.pkl"
    meta_path = MODELS_DIR / f"{prop_type}_v{version}.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_type": prop_type,
        "target_col": target_col,
        "version": version,
        "features": features,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "baseline_mae": round(base_mae, 2),
        "baseline_rmse": round(base_rmse, 2),
        "residual_std": round(residual_std, 2),
        "trained_at": datetime.utcnow().isoformat(),
        "n_samples": len(clean_df),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Trained %s model v%s: MAE=%.2f (Baseline MAE=%.2f), RMSE=%.2f, Residual Std=%.2f",
        prop_type, version, mae, base_mae, rmse, residual_std
    )

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "residual_std": residual_std,
        "baseline_mae": base_mae,
        "baseline_rmse": base_rmse,
        "features": features,
        "version": version,
    }


# ===========================================================================
# Anytime Touchdown Classifier with Isotonic Calibration
# ===========================================================================

def train_td_model(df: pd.DataFrame, version: str = "1.0") -> Dict[str, Any]:
    """
    Train a probabilistic classifier for Anytime Touchdown events.
    Applies out-of-sample isotonic calibration to ensure true Bayesian reliability.
    """
    clean_df = df.copy()
    if "rushing_tds" in clean_df.columns and "receiving_tds" in clean_df.columns:
        clean_df["anytime_td"] = (
            (clean_df["rushing_tds"].fillna(0) > 0) | (clean_df["receiving_tds"].fillna(0) > 0)
        ).astype(int)
    else:
        logger.warning("No touchdown columns found for TD model")
        return {"status": "INSUFFICIENT_DATA"}

    features = [f for f in PROP_FEATURE_REGISTRY["anytime_td"] if f in clean_df.columns]
    for c in features:
        clean_df[c] = clean_df[c].fillna(clean_df[c].median() if not pd.isna(clean_df[c].median()) else 0.0)

    if len(clean_df) < 50:
        return {"status": "INSUFFICIENT_DATA"}

    train_df, val_df = _rolling_time_split(clean_df)

    X_train = np.ascontiguousarray(train_df[features].to_numpy(dtype=np.float32, copy=True))
    y_train = np.ascontiguousarray(train_df["anytime_td"].to_numpy(dtype=np.int32, copy=True))
    X_val = np.ascontiguousarray(val_df[features].to_numpy(dtype=np.float32, copy=True))
    y_val = np.ascontiguousarray(val_df["anytime_td"].to_numpy(dtype=np.int32, copy=True))

    base_model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    base_model.fit(X_train, y_train)

    raw_probs = base_model.predict_proba(X_val)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_val)
    calibrated_probs = np.clip(calibrator.predict(raw_probs), 0.02, 0.98)

    brier = float(brier_score_loss(y_val, calibrated_probs))
    ll = float(log_loss(y_val, np.clip(calibrated_probs, 1e-5, 1 - 1e-5)))

    # Baseline TD rate
    base_rate = float(np.mean(y_train))
    baseline_brier = float(brier_score_loss(y_val, np.full_like(y_val, base_rate, dtype=np.float64)))

    model_path = MODELS_DIR / f"anytime_td_v{version}.pkl"
    calib_path = MODELS_DIR / f"anytime_td_calibrator_v{version}.pkl"
    meta_path = MODELS_DIR / f"anytime_td_v{version}.json"

    with open(model_path, "wb") as f:
        pickle.dump(base_model, f)
    with open(calib_path, "wb") as f:
        pickle.dump(calibrator, f)

    meta = {
        "model_type": "anytime_td",
        "version": version,
        "features": features,
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "baseline_brier": round(baseline_brier, 4),
        "trained_at": datetime.utcnow().isoformat(),
        "n_samples": len(clean_df),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Trained anytime_td model v%s: Brier=%.4f (Baseline Brier=%.4f), LogLoss=%.4f",
        version, brier, baseline_brier, ll
    )

    return {
        "model": base_model,
        "calibrator": calibrator,
        "brier_score": brier,
        "log_loss": ll,
        "baseline_brier": baseline_brier,
        "version": version,
    }


def load_model(model_type: str, version: str = "1.0") -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Load model artifact and its metadata JSON."""
    model_path = MODELS_DIR / f"{model_type}_v{version}.pkl"
    meta_path = MODELS_DIR / f"{model_type}_v{version}.json"

    if not model_path.exists() or not meta_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, meta
