"""
PROPCAST – /api/model/* endpoints
Model performance metrics and calibration curve data.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.schemas import CalibrationData, CalibrationPoint, ModelPerformanceList, ModelPerformanceSchema
from backend.db.database import get_db
from backend.db.models import ModelPerformance

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Model Performance"])

DISCLAIMER = (
    "Model probabilities are statistical estimates, not guarantees. "
    "Sports outcomes contain substantial randomness."
)


# ---------------------------------------------------------------------------
# GET /api/model/performance
# ---------------------------------------------------------------------------
@router.get(
    "/model/performance",
    summary="Model performance metrics",
    response_model=ModelPerformanceList,
)
def get_model_performance(
    prop_type: Optional[str] = Query(None, description="Filter by prop_type"),
    season: Optional[int] = Query(None, description="Filter by season year"),
    model_version: Optional[str] = Query(None, description="Filter by model version string"),
    db: Session = Depends(get_db),
):
    """
    Return MAE, RMSE, Brier Score, and Log Loss broken down by prop type,
    season, and week.  Optionally filter by prop_type, season, or model_version.
    """
    stmt = select(ModelPerformance).order_by(
        ModelPerformance.season.desc(),
        ModelPerformance.week.desc(),
        ModelPerformance.prop_type,
    )

    if prop_type:
        stmt = stmt.where(ModelPerformance.prop_type == prop_type)
    if season:
        stmt = stmt.where(ModelPerformance.season == season)
    if model_version:
        stmt = stmt.where(ModelPerformance.model_version == model_version)

    rows = db.execute(stmt).scalars().all()

    if not rows:
        logger.info(
            "No model performance records found (prop_type=%s season=%s)", prop_type, season
        )
        return ModelPerformanceList(metrics=[], count=0)

    return ModelPerformanceList(metrics=rows, count=len(rows))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GET /api/model/calibration
# ---------------------------------------------------------------------------
@router.get(
    "/model/calibration",
    summary="Calibration curve data",
    response_model=CalibrationData,
)
def get_model_calibration(
    prop_type: Optional[str] = Query(None, description="Filter by prop_type"),
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    db: Session = Depends(get_db),
):
    """
    Return calibration curve data points for reliability diagrams.

    NOTE: In Phase 1 this endpoint scaffolds the schema.
    Actual calibration data is populated by the model evaluation pipeline
    in a future phase.  Returns empty list if no calibration data is stored.
    """
    # Calibration data is derived from ModelPerformance records in later phases.
    # For now, return the schema shape with an empty list so the frontend
    # can render the "awaiting data" state correctly.
    calibration_points: List[CalibrationPoint] = []

    logger.info(
        "Calibration endpoint called (prop_type=%s, version=%s) — "
        "calibration pipeline not yet implemented.",
        prop_type,
        model_version,
    )

    return CalibrationData(
        calibration_points=calibration_points,
        disclaimer=DISCLAIMER,
    )
