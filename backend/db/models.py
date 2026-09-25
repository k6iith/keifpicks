"""
PROPCAST – SQLAlchemy ORM Models
Designed for SQLite (dev) and PostgreSQL (prod).  All tables include
created_at / updated_at timestamps and appropriate composite indexes.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PropType(str, enum.Enum):
    passing_yards = "passing_yards"
    rushing_yards = "rushing_yards"
    receiving_yards = "receiving_yards"
    receptions = "receptions"
    anytime_td = "anytime_td"
    passing_td = "passing_td"
    passing_tds = "passing_tds"
    rushing_attempts = "rushing_attempts"


class PipelineStatus(str, enum.Enum):
    success = "success"
    failure = "failure"
    running = "running"


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    abbreviation: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    conference: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)   # AFC / NFC
    division: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    primary_color: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    secondary_color: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    players: Mapped[List["Player"]] = relationship("Player", back_populates="team", lazy="select")
    home_games: Mapped[List["Game"]] = relationship(
        "Game", foreign_keys="Game.home_team_id", back_populates="home_team", lazy="select"
    )
    away_games: Mapped[List["Game"]] = relationship(
        "Game", foreign_keys="Game.away_team_id", back_populates="away_team", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Team {self.abbreviation}>"


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # NFL's official player identifier (from nfl-data-py)
    gsis_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True, nullable=True, index=True)
    full_name: Mapped[str] = mapped_column(String(150), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    team_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True, index=True
    )
    jersey_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)   # Active / IR / PUP …
    headshot_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    birth_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    height_inches: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight_lbs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    college: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    team: Mapped[Optional["Team"]] = relationship("Team", back_populates="players")
    rosters: Mapped[List["Roster"]] = relationship("Roster", back_populates="player", lazy="select")
    injuries: Mapped[List["Injury"]] = relationship("Injury", back_populates="player", lazy="select")
    game_stats: Mapped[List["PlayerGameStat"]] = relationship(
        "PlayerGameStat", back_populates="player", lazy="select"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="player", lazy="select"
    )
    market_lines: Mapped[List["MarketLine"]] = relationship(
        "MarketLine", back_populates="player", lazy="select"
    )

    __table_args__ = (
        Index("ix_players_name_pos", "full_name", "position"),
    )

    def __repr__(self) -> str:
        return f"<Player {self.full_name} ({self.position})>"


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id_str: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # REG / POST / PRE
    home_team_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True, index=True
    )
    away_team_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kickoff_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    stadium: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    surface: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)   # grass / turf
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    game_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)   # scheduled / live / final
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    home_team: Mapped[Optional["Team"]] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_games"
    )
    away_team: Mapped[Optional["Team"]] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_games"
    )
    weather: Mapped[Optional["Weather"]] = relationship(
        "Weather", back_populates="game", uselist=False, lazy="select"
    )
    injuries: Mapped[List["Injury"]] = relationship("Injury", back_populates="game", lazy="select")
    player_stats: Mapped[List["PlayerGameStat"]] = relationship(
        "PlayerGameStat", back_populates="game", lazy="select"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction", back_populates="game", lazy="select"
    )
    market_lines: Mapped[List["MarketLine"]] = relationship(
        "MarketLine", back_populates="game", lazy="select"
    )

    __table_args__ = (
        Index("ix_games_season_week", "season", "week"),
        Index("ix_games_kickoff", "kickoff_time"),
    )

    def __repr__(self) -> str:
        return f"<Game {self.game_id_str} Season {self.season} W{self.week}>"


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

class Roster(Base):
    __tablename__ = "rosters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    depth_chart_position: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    depth_chart_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="rosters")
    team: Mapped["Team"] = relationship("Team")

    __table_args__ = (
        UniqueConstraint("player_id", "team_id", "season", "week", name="uq_roster_player_week"),
        Index("ix_rosters_season_week", "season", "week"),
    )

    def __repr__(self) -> str:
        return f"<Roster player_id={self.player_id} S{self.season}W{self.week}>"


# ---------------------------------------------------------------------------
# Injury
# ---------------------------------------------------------------------------

class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    game_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="SET NULL"), nullable=True, index=True
    )
    report_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    practice_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # FP / LP / DNP
    game_status: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Questionable / Doubtful / Out / IR
    injury_description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="injuries")
    game: Mapped[Optional["Game"]] = relationship("Game", back_populates="injuries")

    __table_args__ = (
        Index("ix_injuries_player_date", "player_id", "report_date"),
    )

    def __repr__(self) -> str:
        return f"<Injury player_id={self.player_id} status={self.game_status}>"


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

class Weather(Base):
    __tablename__ = "weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    temperature_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipitation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # inches
    condition: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Clear / Rain / Snow…
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="weather")

    def __repr__(self) -> str:
        return f"<Weather game_id={self.game_id} temp={self.temperature_f}F wind={self.wind_mph}mph>"


# ---------------------------------------------------------------------------
# PlayerGameStat
# ---------------------------------------------------------------------------

class PlayerGameStat(Base):
    __tablename__ = "player_game_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    team_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True
    )
    opponent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True
    )

    # Passing
    completions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attempts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    passing_yards: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    passing_tds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    interceptions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Rushing
    carries: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rushing_yards: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rushing_tds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Receiving
    targets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    receptions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    receiving_yards: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    receiving_tds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    air_yards: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    yards_after_catch: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Usage
    snap_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    snap_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Fantasy
    fantasy_points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="game_stats")
    game: Mapped["Game"] = relationship("Game", back_populates="player_stats")
    team: Mapped[Optional["Team"]] = relationship("Team", foreign_keys=[team_id])
    opponent: Mapped[Optional["Team"]] = relationship("Team", foreign_keys=[opponent_id])

    __table_args__ = (
        UniqueConstraint("player_id", "game_id", name="uq_player_game_stat"),
        Index("ix_pgs_season_week", "season", "week"),
        Index("ix_pgs_player_season", "player_id", "season"),
    )

    def __repr__(self) -> str:
        return f"<PlayerGameStat player_id={self.player_id} game_id={self.game_id}>"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prop_type: Mapped[str] = mapped_column(
        Enum(PropType, name="prop_type_enum"), nullable=False, index=True
    )
    projection: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    std_dev: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    percentile_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    percentile_50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    percentile_75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    data_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    feature_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="predictions")
    game: Mapped["Game"] = relationship("Game", back_populates="predictions")

    __table_args__ = (
        Index("ix_pred_player_game_prop", "player_id", "game_id", "prop_type"),
        Index("ix_pred_current", "is_current", "prop_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction player_id={self.player_id} prop={self.prop_type} "
            f"proj={self.projection} v={self.model_version}>"
        )


# ---------------------------------------------------------------------------
# MarketLine
# ---------------------------------------------------------------------------

class MarketLine(Base):
    __tablename__ = "market_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True
    )
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    prop_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    over_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # American odds
    under_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    book: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="market_lines")
    game: Mapped["Game"] = relationship("Game", back_populates="market_lines")

    __table_args__ = (
        Index("ix_ml_player_game_prop", "player_id", "game_id", "prop_type"),
    )

    def __repr__(self) -> str:
        return f"<MarketLine player_id={self.player_id} prop={self.prop_type} line={self.line}>"


# ---------------------------------------------------------------------------
# ModelPerformance
# ---------------------------------------------------------------------------

class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prop_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_predictions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_mp_version_prop", "model_version", "prop_type"),
        Index("ix_mp_season_week", "season", "week"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelPerformance v={self.model_version} prop={self.prop_type} "
            f"S{self.season}W{self.week} mae={self.mae}>"
        )


# ---------------------------------------------------------------------------
# PipelineRun
# ---------------------------------------------------------------------------

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    task_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(PipelineStatus, name="pipeline_status_enum"),
        default=PipelineStatus.running,
        nullable=False,
        index=True,
    )
    records_processed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<PipelineRun task={self.task_name} status={self.status}>"
