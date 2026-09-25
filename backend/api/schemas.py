"""
PROPCAST – Comprehensive Pydantic v2 response schemas.
All schemas use model_config = ConfigDict(from_attributes=True)
so they can be constructed directly from SQLAlchemy ORM objects.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "Model probabilities are statistical estimates, not guarantees. "
    "Sports outcomes contain substantial randomness."
)


class DataUnavailable(BaseModel):
    status: str = "DATA_UNAVAILABLE"
    message: str


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
class TeamBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    abbreviation: str
    full_name: str
    city: Optional[str] = None
    conference: Optional[str] = None
    division: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    logo_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class PlayerBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    gsis_id: Optional[str] = None
    full_name: str
    position: Optional[str] = None
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None
    status: Optional[str] = None
    headshot_url: Optional[str] = None
    birth_date: Optional[datetime] = None
    height_inches: Optional[int] = None
    weight_lbs: Optional[int] = None
    college: Optional[str] = None


class PlayerWithTeam(PlayerBase):
    team: Optional[TeamBase] = None


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
class WeatherSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    temperature_f: Optional[float] = None
    wind_mph: Optional[float] = None
    wind_direction: Optional[str] = None
    humidity_pct: Optional[float] = None
    precipitation: Optional[float] = None
    condition: Optional[str] = None
    is_dome: bool = False
    fetched_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class GameBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    game_id_str: str
    season: int
    week: int
    game_type: Optional[str] = None
    kickoff_time: Optional[datetime] = None
    stadium: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    surface: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_spread: Optional[float] = None
    game_total: Optional[float] = None
    status: Optional[str] = None


class GameWithTeams(GameBase):
    home_team: Optional[TeamBase] = None
    away_team: Optional[TeamBase] = None
    weather: Optional[WeatherSchema] = None


class GameList(BaseModel):
    games: List[GameWithTeams]
    count: int


# ---------------------------------------------------------------------------
# Injury
# ---------------------------------------------------------------------------
class InjurySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: int
    game_id: Optional[int] = None
    report_date: Optional[datetime] = None
    practice_status: Optional[str] = None
    game_status: Optional[str] = None
    injury_description: Optional[str] = None
    source: Optional[str] = None
    created_at: datetime


class InjuryWithPlayer(InjurySchema):
    player: Optional[PlayerBase] = None


class InjuryList(BaseModel):
    injuries: List[InjuryWithPlayer]
    count: int
    last_updated: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Projection / Prediction
# ---------------------------------------------------------------------------
class PredictionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: int
    game_id: int
    model_version: str
    prop_type: str
    projection: Optional[float] = None
    std_dev: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_75: Optional[float] = None
    prediction_created_at: datetime
    data_updated_at: Optional[datetime] = None
    feature_version: Optional[str] = None
    is_current: bool


# ---------------------------------------------------------------------------
# MarketLine
# ---------------------------------------------------------------------------
class MarketLineSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: int
    game_id: int
    prop_type: str
    line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    book: Optional[str] = None
    fetched_at: datetime


# ---------------------------------------------------------------------------
# Prop card (combines prediction + market line + context)
# ---------------------------------------------------------------------------
class PropCard(BaseModel):
    """
    Composite view shown on prop pages.
    All probability fields are model estimates – see disclaimer.
    """
    model_config = ConfigDict(from_attributes=True)

    player: PlayerBase
    team: Optional[TeamBase] = None
    opponent: Optional[TeamBase] = None
    game: GameBase
    prop_type: str
    projection: Optional[float] = Field(None, description="Model mean projection")
    std_dev: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_75: Optional[float] = None
    market_line: Optional[float] = Field(None, description="Sportsbook consensus line")
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    book: Optional[str] = None
    over_probability: Optional[float] = Field(
        None, description="Model-estimated probability of going over the line"
    )
    under_probability: Optional[float] = Field(
        None, description="Model-estimated probability of going under the line"
    )
    model_edge: Optional[float] = Field(
        None,
        description="Difference between model implied probability and market implied probability",
    )
    line_difference: Optional[float] = Field(None, description="Difference between projection and market line")
    model_confidence: Optional[float] = Field(None, description="Confidence percentage (0-100%) based on sample size and data quality")
    sample_size: Optional[int] = Field(None, description="Number of prior games used in projection")
    recent_trend: Optional[str] = Field(None, description="Recent performance trend (e.g. +14% vs 8-wk avg)")
    matchup_adjustment: Optional[float] = Field(None, description="Opponent defense adjustment factor")
    injury_adjustment: Optional[float] = Field(None, description="Teammate/opponent injury impact factor")
    weather_adjustment: Optional[float] = Field(None, description="Wind/temperature weather impact factor")
    game_script_adjustment: Optional[float] = Field(None, description="Spread/game total script impact factor")
    increasing_factors: List[str] = Field(default_factory=list, description="Key factors driving projection up")
    decreasing_factors: List[str] = Field(default_factory=list, description="Key factors driving projection down")
    data_quality_score: Optional[float] = Field(
        None, description="0-1 score reflecting data freshness and completeness"
    )
    last_updated: Optional[datetime] = None
    model_version: Optional[str] = None
    disclaimer: str = DISCLAIMER


class PropList(BaseModel):
    props: List[PropCard]
    count: int
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Player detail
# ---------------------------------------------------------------------------
class PlayerGameStatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    game_id: int
    season: int
    week: int
    completions: Optional[int] = None
    attempts: Optional[int] = None
    passing_yards: Optional[float] = None
    passing_tds: Optional[int] = None
    interceptions: Optional[int] = None
    carries: Optional[int] = None
    rushing_yards: Optional[float] = None
    rushing_tds: Optional[int] = None
    targets: Optional[int] = None
    receptions: Optional[int] = None
    receiving_yards: Optional[float] = None
    receiving_tds: Optional[int] = None
    air_yards: Optional[float] = None
    yards_after_catch: Optional[float] = None
    snap_count: Optional[int] = None
    snap_pct: Optional[float] = None
    fantasy_points: Optional[float] = None


class PlayerDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player: PlayerWithTeam
    recent_stats: List[PlayerGameStatSchema] = []
    current_projections: List[PredictionSchema] = []
    current_market_lines: List[MarketLineSchema] = []
    injuries: List[InjurySchema] = []
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Game detail
# ---------------------------------------------------------------------------
class GameDetail(BaseModel):
    game: GameWithTeams
    home_players: List[PlayerBase] = []
    away_players: List[PlayerBase] = []
    predictions: List[PropCard] = []
    injuries: List[InjuryWithPlayer] = []
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
class ModelPerformanceSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_version: str
    prop_type: str
    season: int
    week: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    n_predictions: Optional[int] = None
    evaluated_at: datetime


class ModelPerformanceList(BaseModel):
    metrics: List[ModelPerformanceSchema]
    count: int


class CalibrationPoint(BaseModel):
    """One point on a reliability diagram."""
    mean_predicted_probability: float
    fraction_of_positives: float
    n_samples: int
    prop_type: str
    model_version: str


class CalibrationData(BaseModel):
    calibration_points: List[CalibrationPoint]
    disclaimer: str = DISCLAIMER
