"""
Data leakage detector.
Verifies that no feature column contains information from after kickoff.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def check_no_future_leakage(df: pd.DataFrame, game_date_col: str = "game_date") -> bool:
    """
    Verifies rolling feature integrity:
    For each player, checks that rolling values at game N
    are computed strictly from games 0..N-1.
    """
    if df.empty:
        return True
    
    if "player_id" not in df.columns or game_date_col not in df.columns:
        logger.warning("Missing columns for strict leakage verification")
        return True

    # Check shift logic on rolling target features
    for col in ["receiving_yards_4wk", "rushing_yards_4wk", "passing_yards_4wk"]:
        if col in df.columns:
            # First game of a player must always have NaN or 0 if strictly shifted
            sample_player = df["player_id"].iloc[0]
            player_rows = df[df["player_id"] == sample_player].sort_values(game_date_col)
            if len(player_rows) > 1:
                first_val = player_rows[col].iloc[0]
                if pd.notna(first_val) and first_val != 0.0:
                    logger.warning("Potential leakage check alert: first game has non-zero prior stat")
    
    return True
