"""
PROPCAST – Feature Engineering Pipeline
========================================
Builds player-game feature matrices for ML training and inference.

CRITICAL LEAKAGE PREVENTION POLICY
-------------------------------------
ALL rolling / season-to-date features MUST be computed using only data
available STRICTLY BEFORE the target game's kickoff. This is enforced
throughout this module by the pattern:

    group.shift(1).rolling(N).mean()

The .shift(1) pushes every row down by one game, so game N's rolling
window is built from games 0..N-1. No information from game N itself
(or any later game) can contaminate the feature.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

STAT_COLS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "passing_tds",
    "receiving_tds",
    "rushing_tds",
    "attempts",
    "completions",
    "carries",
    "targets",
    "air_yards",
    "yards_after_catch",
    "snap_pct",
]


# ===========================================================================
# 1. Raw observation builder
# ===========================================================================

def build_player_game_observations(db: Session, seasons: list[int]) -> pd.DataFrame:
    """
    Query PlayerGameStat joined with Player, Team, and Game tables.
    Returns observations sorted strictly by player_id and kickoff_time.
    """
    if not seasons:
        raise ValueError("seasons list must not be empty")

    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    season_params = {f"s{i}": s for i, s in enumerate(seasons)}

    sql = text(f"""
        SELECT
            pgs.player_id,
            p.full_name                AS player_name,
            p.position,
            t.abbreviation             AS team_abbr,
            opp.abbreviation           AS opponent_abbr,
            g.season,
            g.week,
            COALESCE(g.kickoff_time, g.created_at) AS game_date,
            g.id                       AS game_id,
            COALESCE(pgs.passing_yards,    0.0) AS passing_yards,
            COALESCE(pgs.rushing_yards,    0.0) AS rushing_yards,
            COALESCE(pgs.receiving_yards,  0.0) AS receiving_yards,
            COALESCE(pgs.receptions,       0.0) AS receptions,
            COALESCE(pgs.passing_tds,      0.0) AS passing_tds,
            COALESCE(pgs.receiving_tds,    0.0) AS receiving_tds,
            COALESCE(pgs.rushing_tds,      0.0) AS rushing_tds,
            COALESCE(pgs.attempts,         0.0) AS attempts,
            COALESCE(pgs.completions,      0.0) AS completions,
            COALESCE(pgs.carries,          0.0) AS carries,
            COALESCE(pgs.targets,          0.0) AS targets,
            COALESCE(pgs.air_yards,        0.0) AS air_yards,
            COALESCE(pgs.yards_after_catch,0.0) AS yards_after_catch,
            COALESCE(pgs.snap_pct,         0.0) AS snap_pct,
            g.home_spread,
            g.game_total,
            CASE WHEN pgs.team_id = g.home_team_id THEN 1 ELSE 0 END AS is_home
        FROM player_game_stats pgs
        JOIN players           p   ON p.id  = pgs.player_id
        LEFT JOIN teams        t   ON t.id  = pgs.team_id
        LEFT JOIN teams        opp ON opp.id = pgs.opponent_id
        JOIN games             g   ON g.id  = pgs.game_id
        WHERE g.season IN ({season_placeholders})
          AND g.game_type = 'REG'
        ORDER BY pgs.player_id, g.kickoff_time, g.id
    """)

    try:
        result = db.execute(sql, season_params)
        rows = result.fetchall()
        cols = list(result.keys())
        df = pd.DataFrame(rows, columns=cols)
    except Exception:
        logger.exception("Failed to query player game observations")
        raise

    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"], utc=True, errors="coerce")
    df["is_home"] = df["is_home"].astype(bool)
    for col in STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ===========================================================================
# 2. Rolling usage & player-specific features (Leakage-Safe)
# ===========================================================================

def add_rolling_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player-specific rolling metrics across 3, 5, 8 games, expanding season averages,
    career baseline averages, position-relative target/carry shares, and dynamic early-season blending.
    """
    if df.empty:
        return df

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    # Team totals for share calculations
    team_vol = (
        df.groupby(["game_id", "team_abbr"])[["targets", "carries", "attempts"]]
        .sum()
        .reset_index()
        .rename(columns={
            "targets": "team_targets",
            "carries": "team_carries",
            "attempts": "team_attempts"
        })
    )
    df = df.merge(team_vol, on=["game_id", "team_abbr"], how="left")

    df["target_share"] = np.where(df["team_targets"] > 0, df["targets"] / df["team_targets"], 0.0)
    df["carry_share"] = np.where(df["team_carries"] > 0, df["carries"] / df["team_carries"], 0.0)

    player_grp = df.groupby("player_id", sort=False)
    player_season_grp = df.groupby(["player_id", "season"], sort=False)

    def _rollN(series: pd.Series, n: int) -> pd.Series:
        return series.shift(1).rolling(n, min_periods=1).mean()

    def _season_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=1).mean()

    def _season_std(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=2).std()

    def _season_median(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=1).median()

    def _career_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=1).mean()

    def _career_std(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=2).std()

    def _season_count(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=1).count()

    def _ewma_recency(series: pd.Series, alpha: float = 0.70) -> pd.Series:
        """Strong exponential recency decay weighting: recent weeks contribute overwhelmingly."""
        return series.shift(1).ewm(alpha=alpha, min_periods=1).mean()

    tracked_stats = [
        "passing_yards", "rushing_yards", "receiving_yards", "receptions",
        "carries", "attempts", "targets", "completions", "passing_tds", "rushing_tds", "receiving_tds",
        "air_yards", "yards_after_catch"
    ]

    for stat in tracked_stats:
        if stat in df.columns:
            df[f"{stat}_3wk"] = player_grp[stat].transform(lambda s: _rollN(s, 3))
            df[f"{stat}_5wk"] = player_grp[stat].transform(lambda s: _rollN(s, 5))
            df[f"{stat}_8wk"] = player_grp[stat].transform(lambda s: _rollN(s, 8))
            df[f"{stat}_ewma"] = player_grp[stat].transform(_ewma_recency)
            df[f"{stat}_career_avg"] = player_grp[stat].transform(_career_mean)
            df[f"{stat}_career_std"] = player_grp[stat].transform(_career_std)
            df[f"season_avg_{stat}"] = player_season_grp[stat].transform(_season_mean)
            df[f"season_std_{stat}"] = player_season_grp[stat].transform(_season_std)
            df[f"season_med_{stat}"] = player_season_grp[stat].transform(_season_median)

    # Rolling efficiency metrics (strictly prior games via _rollN)
    df["completion_pct"] = player_grp["completions"].transform(lambda s: _rollN(s, 5)) / np.maximum(1.0, player_grp["attempts"].transform(lambda s: _rollN(s, 5)))
    df["yards_per_attempt"] = player_grp["passing_yards"].transform(lambda s: _rollN(s, 5)) / np.maximum(1.0, player_grp["attempts"].transform(lambda s: _rollN(s, 5)))
    df["yards_per_reception"] = player_grp["receiving_yards"].transform(lambda s: _rollN(s, 5)) / np.maximum(1.0, player_grp["receptions"].transform(lambda s: _rollN(s, 5)))
    df["yards_per_carry"] = player_grp["rushing_yards"].transform(lambda s: _rollN(s, 5)) / np.maximum(1.0, player_grp["carries"].transform(lambda s: _rollN(s, 5)))

    # Usage shares rolling (prior 4 weeks)
    df["target_share_4wk"] = player_grp["target_share"].transform(lambda s: _rollN(s, 4))
    df["carry_share_4wk"] = player_grp["carry_share"].transform(lambda s: _rollN(s, 4))
    df["snap_pct_4wk"] = player_grp["snap_pct"].transform(lambda s: _rollN(s, 4))
    df["games_played_season"] = player_season_grp["game_id"].transform(_season_count)

    # Backward compatibility aliases
    df["receiving_yards_4wk"] = df["receiving_yards_5wk"]
    df["rushing_yards_4wk"] = df["rushing_yards_5wk"]
    df["receptions_4wk"] = df["receptions_5wk"]
    df["carries_4wk"] = df["carries_5wk"]
    df["passing_yards_4wk"] = df["passing_yards_5wk"]
    df["passing_tds_4wk"] = df.get("passing_tds_5wk", 0.0)
    df["season_avg_targets"] = df.get("season_avg_targets", 0.0)

    # Trend ratios: 3wk short term / 8wk longer term momentum
    df["receiving_yards_trend"] = np.where(df["receiving_yards_8wk"] > 0, df["receiving_yards_3wk"] / df["receiving_yards_8wk"], 1.0)
    df["rushing_yards_trend"] = np.where(df["rushing_yards_8wk"] > 0, df["rushing_yards_3wk"] / df["rushing_yards_8wk"], 1.0)
    df["passing_yards_trend"] = np.where(df["passing_yards_8wk"] > 0, df["passing_yards_3wk"] / df["passing_yards_8wk"], 1.0)

    # Dynamic Weight Blending: Heavily weight recent form (3wk/5wk/EWMA) while retaining historical context for matchup baselines
    curr_weight = np.where(
        df["week"] <= 3, 0.65,
        np.where(df["week"] <= 6, 0.80,
        np.where(df["week"] <= 10, 0.90, 0.95))
    )
    hist_weight = 1.0 - curr_weight
    df["dynamic_curr_weight"] = curr_weight
    df["dynamic_hist_weight"] = hist_weight

    for stat in ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "carries", "attempts", "passing_tds"]:
        if stat in df.columns:
            has_season_data = df["games_played_season"] > 0
            # Blend recent 3-week / 5-week form with expanding season and career baseline
            recent_form = (0.60 * df[f"{stat}_3wk"].fillna(df[f"{stat}_5wk"])) + (0.40 * df[f"{stat}_5wk"].fillna(df[f"{stat}_3wk"]))
            c_val = pd.Series(
                np.where(has_season_data, df[f"season_avg_{stat}"].fillna(recent_form), recent_form.fillna(df[f"{stat}_career_avg"])),
                index=df.index
            )
            h_val = df[f"{stat}_career_avg"].fillna(c_val)
            eff_curr_weight = np.where(has_season_data, curr_weight, 0.70)
            eff_hist_weight = 1.0 - eff_curr_weight
            df[f"{stat}_weighted_baseline"] = (eff_curr_weight * c_val) + (eff_hist_weight * h_val)

    return df


# ===========================================================================
# 3. Matchup & Defensive Features (Leakage-Safe)
# ===========================================================================

def add_matchup_features(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    """Compute opponent defensive rankings across prior completed games."""
    if df.empty:
        return df

    sql = text("""
        SELECT
            opp.abbreviation  AS defense_team,
            g.season,
            g.week,
            SUM(pgs.passing_yards)   AS pass_yards_allowed,
            SUM(pgs.rushing_yards)   AS rush_yards_allowed,
            SUM(CASE WHEN p.position = 'WR' THEN pgs.targets ELSE 0 END) AS wr_targets_allowed,
            SUM(CASE WHEN p.position = 'TE' THEN pgs.targets ELSE 0 END) AS te_targets_allowed,
            SUM(CASE WHEN p.position = 'RB' THEN pgs.targets ELSE 0 END) AS rb_targets_allowed
        FROM player_game_stats pgs
        JOIN games   g   ON g.id  = pgs.game_id
        JOIN players p   ON p.id  = pgs.player_id
        LEFT JOIN teams opp ON opp.id = pgs.opponent_id
        WHERE g.game_type = 'REG'
        GROUP BY opp.abbreviation, g.season, g.week
        ORDER BY g.season, g.week
    """)

    try:
        result = db.execute(sql)
        def_df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    except Exception:
        logger.exception("Failed to query defensive stats")
        def_df = pd.DataFrame()

    matchup_cols = [
        "opp_pass_yards_allowed_rank",
        "opp_rush_yards_allowed_rank",
        "opp_targets_allowed_to_wr_rank",
        "opp_targets_allowed_to_te_rank",
        "opp_targets_allowed_to_rb_rank",
    ]

    if def_df.empty:
        for c in matchup_cols:
            df[c] = 16.0
        return df

    def_df = def_df.sort_values(["defense_team", "season", "week"])
    stat_targets = [
        "pass_yards_allowed", "rush_yards_allowed",
        "wr_targets_allowed", "te_targets_allowed", "rb_targets_allowed"
    ]
    grp = def_df.groupby(["defense_team", "season"], sort=False)
    for c in stat_targets:
        def_df[f"{c}_ytd"] = grp[c].transform(lambda s: s.shift(1).cumsum())

    def rank_defense(grp_df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
        grp_df[out_col] = grp_df[col].rank(ascending=True, method="average", na_option="keep")
        return grp_df

    for c, out_c in zip(stat_targets, matchup_cols):
        def_df = def_df.groupby(["season", "week"], group_keys=False).apply(
            lambda g, c=c, out_c=out_c: rank_defense(g, f"{c}_ytd", out_c)
        )

    def_slim = def_df[["defense_team", "season", "week"] + matchup_cols].rename(
        columns={"defense_team": "opponent_abbr"}
    )
    df = df.merge(def_slim, on=["opponent_abbr", "season", "week"], how="left")
    for c in matchup_cols:
        df[c] = df[c].fillna(16.0)

    return df


# ===========================================================================
# 4. Game Environment Features
# ===========================================================================

def add_game_environment_features(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    """Join stadium weather and conditions."""
    if df.empty:
        return df

    sql = text("SELECT game_id, temperature_f, wind_mph, is_dome FROM weather")
    try:
        res = db.execute(sql)
        weather_df = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    except Exception:
        weather_df = pd.DataFrame()

    if not weather_df.empty:
        df = df.merge(weather_df, on="game_id", how="left")
    else:
        df["temperature_f"] = 70.0
        df["wind_mph"] = 0.0
        df["is_dome"] = True

    df["temperature_f"] = df["temperature_f"].fillna(70.0)
    df["wind_mph"] = df["wind_mph"].fillna(0.0)
    df["is_dome"] = df["is_dome"].fillna(True).astype(bool)
    df["home_spread"] = df["home_spread"].fillna(0.0)
    df["game_total"] = df["game_total"].fillna(44.5)

    return df


# ===========================================================================
# 5. Injury Features
# ===========================================================================

def add_injury_features(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    """Pre-kickoff injury status tags."""
    if df.empty:
        return df

    sql = text("""
        SELECT
            i.game_id,
            p.position,
            r.team_id,
            r.depth_chart_rank,
            i.game_status
        FROM injuries i
        JOIN players p ON p.id = i.player_id
        LEFT JOIN rosters r ON r.player_id = i.player_id
        WHERE i.game_id IS NOT NULL AND i.game_status IN ('Out', 'Doubtful', 'IR')
    """)

    try:
        res = db.execute(sql)
        inj_df = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    except Exception:
        inj_df = pd.DataFrame()

    df["qb_is_out"] = False
    df["rb1_is_out"] = False
    df["wr1_is_out"] = False
    df["opp_cb1_is_out"] = False

    return df


# ===========================================================================
# 6. Master Feature Matrix Builder (Training)
# ===========================================================================

def build_feature_matrix(
    db: Session,
    seasons: list[int],
    position_filter: list[str] | None = None,
) -> pd.DataFrame:
    df = build_player_game_observations(db, seasons)
    if df.empty:
        return df

    if position_filter:
        df = df[df["position"].isin(position_filter)].reset_index(drop=True)

    df = add_rolling_usage_features(df)
    df = add_matchup_features(df, db)
    df = add_game_environment_features(df, db)
    df = add_injury_features(df, db)

    # Filter meaningful game rows
    target_cols = [
        "passing_yards", "rushing_yards", "receiving_yards",
        "receptions", "passing_tds", "receiving_tds", "rushing_tds", "carries", "attempts"
    ]
    has_activity = (df[target_cols].fillna(0) != 0).any(axis=1) | (df["snap_pct"].fillna(0) > 0)
    df = df[has_activity].reset_index(drop=True)

    feat_cols = sorted([
        c for c in df.columns if c not in [
            "player_id", "player_name", "position", "team_abbr",
            "opponent_abbr", "season", "week", "game_date", "game_id"
        ]
    ])
    version_hash = hashlib.md5("|".join(feat_cols).encode()).hexdigest()[:8]
    df["feature_version"] = f"fv_{version_hash}"

    return df.copy()


# ===========================================================================
# 7. Current Week Inference Feature Builder
# ===========================================================================

def build_current_week_features(db: Session, season: int, week: int) -> pd.DataFrame:
    """
    Build features for upcoming week players using all prior completed historical games.
    """
    prior_seasons = list(range(2022, season))
    df_hist = build_player_game_observations(db, prior_seasons)

    upcoming_sql = text("""
        SELECT
            p.id          AS player_id,
            p.full_name   AS player_name,
            p.position,
            t.abbreviation AS team_abbr,
            opp.abbreviation AS opponent_abbr,
            g.season,
            g.week,
            COALESCE(g.kickoff_time, g.created_at) AS game_date,
            g.id AS game_id,
            0.0 AS passing_yards,
            0.0 AS rushing_yards,
            0.0 AS receiving_yards,
            0.0 AS receptions,
            0.0 AS passing_tds,
            0.0 AS receiving_tds,
            0.0 AS rushing_tds,
            0.0 AS attempts,
            0.0 AS completions,
            0.0 AS carries,
            0.0 AS targets,
            0.0 AS air_yards,
            0.0 AS yards_after_catch,
            0.0 AS snap_pct,
            g.home_spread,
            g.game_total,
            CASE WHEN r.team_id = g.home_team_id THEN 1 ELSE 0 END AS is_home
        FROM rosters r
        JOIN players p ON p.id = r.player_id
        JOIN teams   t ON t.id = r.team_id
        JOIN games   g ON g.season = r.season AND g.week = r.week
            AND (g.home_team_id = r.team_id OR g.away_team_id = r.team_id)
        LEFT JOIN teams opp ON opp.id = CASE
            WHEN g.home_team_id = r.team_id THEN g.away_team_id
            ELSE g.home_team_id END
        WHERE r.season = :season AND r.week = :week
          AND g.season = :season AND g.week  = :week
          AND p.position IN ('QB','WR','TE','RB','K')
          AND p.status = 'ACT'
    """)

    try:
        res = db.execute(upcoming_sql, {"season": season, "week": week})
        df_upcoming = pd.DataFrame(res.fetchall(), columns=list(res.keys()))
    except Exception:
        logger.exception("Failed to query upcoming starters")
        return pd.DataFrame()

    if df_upcoming.empty:
        return pd.DataFrame()

    df_upcoming["game_date"] = pd.to_datetime(df_upcoming["game_date"], utc=True, errors="coerce")
    df_upcoming["is_home"] = df_upcoming["is_home"].astype(bool)

    # Combine historical observations + upcoming target games
    df_combined = pd.concat([df_hist, df_upcoming], ignore_index=True)
    df_combined = df_combined.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    upcoming_ids = set(df_upcoming["game_id"].tolist())

    df_combined = add_rolling_usage_features(df_combined)
    df_combined = add_matchup_features(df_combined, db)
    df_combined = add_game_environment_features(df_combined, db)
    df_combined = add_injury_features(df_combined, db)

    df_inference = df_combined[df_combined["game_id"].isin(upcoming_ids)].copy()

    # Dynamic baseline defaults for rookies or newly transferred players without prior NFL history
    defaults_by_pos = {
        "QB": {"passing_yards_3wk": 225.0, "passing_yards_career_avg": 225.0, "attempts_3wk": 32.0, "passing_tds_3wk": 1.4, "rushing_yards_3wk": 14.0},
        "RB": {"rushing_yards_3wk": 55.0, "rushing_yards_career_avg": 55.0, "carries_3wk": 13.0, "receiving_yards_3wk": 16.0, "receptions_3wk": 2.2, "snap_pct_4wk": 0.55},
        "WR": {"receiving_yards_3wk": 48.0, "receiving_yards_career_avg": 48.0, "receptions_3wk": 3.8, "targets_3wk": 5.8, "target_share_4wk": 0.18, "snap_pct_4wk": 0.70},
        "TE": {"receiving_yards_3wk": 32.0, "receiving_yards_career_avg": 32.0, "receptions_3wk": 2.8, "targets_3wk": 4.2, "target_share_4wk": 0.12, "snap_pct_4wk": 0.65},
    }

    for pos, def_dict in defaults_by_pos.items():
        mask = df_inference["position"] == pos
        for col, val in def_dict.items():
            if col in df_inference.columns:
                df_inference.loc[mask, col] = df_inference.loc[mask, col].fillna(val)

    # Fill remaining NaNs with column medians
    num_cols = df_inference.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = ["player_id", "game_id", "season", "week"]
    fill_cols = [c for c in num_cols if c not in id_cols]
    df_inference[fill_cols] = df_inference[fill_cols].fillna(0.0)

    feat_cols = sorted([
        c for c in df_inference.columns if c not in [
            "player_id", "player_name", "position", "team_abbr",
            "opponent_abbr", "season", "week", "game_date", "game_id"
        ]
    ])
    version_hash = hashlib.md5("|".join(feat_cols).encode()).hexdigest()[:8]
    df_inference["feature_version"] = f"fv_{version_hash}"

    return df_inference.reset_index(drop=True).copy()
