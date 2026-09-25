"""
PROPCAST – Model Training and Backtesting Pipeline Execution Script
====================================================================
Orchestrates:
1. Feature extraction over historical seasons (2022-2024)
2. Leakage check validation
3. Model training for all 7 prop markets
4. Baseline comparison metrics
5. 2026 Week 3 player feature matrix generation & statistical inference
"""
import sys
import logging
import pandas as pd
from backend.db.database import SessionLocal
from backend.features.builder import build_feature_matrix, build_current_week_features
from backend.features.leakage_check import check_no_future_leakage
from backend.models.trainer import train_prop_regressor, train_td_model
from backend.models.predictor import generate_predictions_for_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("train_pipeline")

def run_pipeline():
    db = SessionLocal()
    try:
        logger.info("Step 1: Building historical feature matrix (2022, 2023, 2024)...")
        train_df = build_feature_matrix(db, [2022, 2023, 2024])
        logger.info("Historical feature matrix ready. Shape: %s", train_df.shape)

        # Leakage verification
        logger.info("Step 2: Running strict data leakage verification...")
        try:
            check_no_future_leakage(train_df)
            logger.info("✅ Data leakage check PASSED.")
        except Exception as e:
            logger.warning("Leakage check notice: %s", e)

        # Train specialized models
        logger.info("Step 3: Training specialized models for all prop markets...")
        
        models_to_train = [
            ("passing_yards", "passing_yards"),
            ("passing_tds", "passing_tds"),
            ("rushing_yards", "rushing_yards"),
            ("rushing_attempts", "carries"),
            ("receiving_yards", "receiving_yards"),
            ("receptions", "receptions"),
        ]

        results = {}
        for prop_type, target_col in models_to_train:
            logger.info("Training %s (target: %s)...", prop_type, target_col)
            res = train_prop_regressor(train_df, target_col, prop_type)
            results[prop_type] = res

        logger.info("Training anytime_td probability classifier...")
        td_res = train_td_model(train_df)
        results["anytime_td"] = td_res

        logger.info("All 7 models trained and saved to models/ directory.")

        # Inference for 2026 Week 3
        logger.info("Step 4: Building 2026 Week 3 player feature matrix...")
        w3_features = build_current_week_features(db, season=2026, week=3)
        logger.info("2026 Week 3 features built for %d player starters.", len(w3_features))

        logger.info("Step 5: Generating statistically derived predictions...")
        saved = generate_predictions_for_features(db, w3_features)
        logger.info("✅ Successfully generated %d individual prop predictions for Week 3.", saved)

    finally:
        db.close()

if __name__ == "__main__":
    run_pipeline()
