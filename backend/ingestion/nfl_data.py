"""
PROPCAST – Core NFL Data Ingestion
Uses nfl_data_py to pull historical and current season data into the database.

All functions are safe to call repeatedly (upsert semantics).
All exceptions are caught and logged — functions always return int (count or 0).
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import (
    Game,
    PipelineRun,
    PipelineStatus,
    Player,
    PlayerGameStat,
    Roster,
    Team,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_int(val) -> Optional[int]:
    """Convert a value to int, returning None on failure."""
    try:
        if pd.isna(val):
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_str(val) -> Optional[str]:
    """Convert a value to stripped string, returning None if empty."""
    try:
        if pd.isna(val):
            return None
        s = str(val).strip()
        return s if s else None
    except (TypeError, ValueError):
        return None


def _parse_kickoff(gameday, gametime) -> Optional[datetime]:
    """
    Combine a gameday date string and gametime string into a datetime.

    Parameters
    ----------
    gameday : any
        e.g. '2023-09-07' or a date object.
    gametime : any
        e.g. '8:20PM' or '13:00' or similar.

    Returns
    -------
    datetime | None
    """
    try:
        if pd.isna(gameday):
            return None
        if isinstance(gameday, (date, datetime)):
            day = gameday if isinstance(gameday, date) else gameday.date()
        else:
            day = datetime.strptime(str(gameday).strip(), "%Y-%m-%d").date()

        if pd.isna(gametime):
            return datetime(day.year, day.month, day.day)

        time_str = str(gametime).strip().upper().replace(" ", "")
        # Try multiple time formats
        for fmt in ("%I:%M%p", "%H:%M", "%I%p"):
            try:
                t = datetime.strptime(time_str, fmt).time()
                return datetime(day.year, day.month, day.day, t.hour, t.minute)
            except ValueError:
                continue
        # If parsing fails, return midnight of gameday
        return datetime(day.year, day.month, day.day)
    except Exception:  # noqa: BLE001
        return None


def _get_team_id_map(db: Session) -> dict[str, int]:
    """Return {abbreviation: id} mapping for all teams in the DB."""
    teams = db.query(Team.abbreviation, Team.id).all()
    return {abbr: tid for abbr, tid in teams}


def _current_season() -> int:
    """Return the current NFL season year."""
    today = date.today()
    # NFL seasons start in September; if before March, we're in the prior season
    return today.year if today.month >= 3 else today.year - 1


def _current_nfl_week(season: int) -> int:
    """
    Estimate the current NFL week based on the calendar.
    Week 1 starts the first Thursday of September.
    Returns 1 if before the season starts, 18 if after the regular season.
    """
    today = date.today()
    # Approximate season start: first Thursday of September
    sept_1 = date(season, 9, 1)
    # Find first Thursday (weekday 3)
    days_to_thursday = (3 - sept_1.weekday()) % 7
    season_start = sept_1 + pd.Timedelta(days=days_to_thursday)
    season_start = season_start.to_pydatetime().date()

    if today < season_start:
        return 1
    delta_days = (today - season_start).days
    week = delta_days // 7 + 1
    return min(max(week, 1), 22)  # cap at Week 22 (postseason)


def _record_pipeline_run(
    db: Session,
    task_name: str,
    status: PipelineStatus,
    records_processed: Optional[int] = None,
    error_message: Optional[str] = None,
    started_at: Optional[datetime] = None,
) -> None:
    """Insert a PipelineRun log entry."""
    try:
        run = PipelineRun(
            task_name=task_name,
            started_at=started_at or datetime.utcnow(),
            completed_at=datetime.utcnow(),
            status=status,
            records_processed=records_processed,
            error_message=error_message,
        )
        db.add(run)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record pipeline run for '%s': %s", task_name, exc)
        try:
            db.rollback()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. ingest_teams
# ---------------------------------------------------------------------------

def ingest_teams(db: Session) -> int:
    """
    Fetch NFL team descriptors via nfl_data_py and upsert into the teams table.

    Returns
    -------
    int
        Number of records processed (0 on failure).
    """
    import nfl_data_py as nfl

    logger.info("Ingesting teams...")
    started_at = datetime.utcnow()

    try:
        df: pd.DataFrame = nfl.import_team_desc()
    except Exception as exc:  # noqa: BLE001
        logger.error("nfl_data_py.import_team_desc() failed: %s", exc)
        _record_pipeline_run(
            db, "ingest_teams", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    if df is None or df.empty:
        logger.warning("import_team_desc() returned empty DataFrame.")
        return 0

    count = 0
    try:
        for _, row in df.iterrows():
            abbr = _safe_str(row.get("team_abbr"))
            if not abbr:
                continue

            full_name = _safe_str(row.get("team_name")) or abbr
            primary_color = _safe_str(row.get("team_color"))
            secondary_color = _safe_str(row.get("team_color2"))
            logo_url = _safe_str(row.get("team_logo_espn"))
            conference = _safe_str(row.get("team_conf"))
            division = _safe_str(row.get("team_division"))

            existing: Optional[Team] = (
                db.query(Team).filter(Team.abbreviation == abbr).first()
            )

            if existing:
                existing.full_name = full_name
                existing.primary_color = primary_color
                existing.secondary_color = secondary_color
                existing.logo_url = logo_url
                existing.conference = conference
                existing.division = division
            else:
                team = Team(
                    abbreviation=abbr,
                    full_name=full_name,
                    primary_color=primary_color,
                    secondary_color=secondary_color,
                    logo_url=logo_url,
                    conference=conference,
                    division=division,
                )
                db.add(team)

            count += 1

        db.commit()
        logger.info("Teams ingested: %d records.", count)
        _record_pipeline_run(
            db, "ingest_teams", PipelineStatus.success,
            records_processed=count, started_at=started_at
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Error committing teams: %s", exc)
        _record_pipeline_run(
            db, "ingest_teams", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    return count


# ---------------------------------------------------------------------------
# 2. ingest_players
# ---------------------------------------------------------------------------

def ingest_players(db: Session, season: int) -> int:
    """
    Fetch NFL rosters for a given season and upsert Player records.

    Parameters
    ----------
    season : int
        NFL season year (e.g. 2024).

    Returns
    -------
    int
        Number of records processed.
    """
    import nfl_data_py as nfl

    logger.info("Ingesting players for season %d...", season)
    started_at = datetime.utcnow()

    try:
        df: pd.DataFrame = nfl.import_seasonal_rosters([season])
    except Exception as exc:  # noqa: BLE001
        logger.error("nfl_data_py.import_seasonal_rosters([%d]) failed: %s", season, exc)
        _record_pipeline_run(
            db, f"ingest_players_{season}", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    if df is None or df.empty:
        logger.warning("import_rosters([%d]) returned empty DataFrame.", season)
        return 0

    team_id_map = _get_team_id_map(db)
    count = 0

    try:
        for _, row in df.iterrows():
            gsis_id = _safe_str(row.get("player_id"))
            if not gsis_id:
                continue  # skip players without official NFL ID

            full_name = _safe_str(row.get("player_name")) or "Unknown"
            position = _safe_str(row.get("position"))
            headshot_url = _safe_str(row.get("headshot_url"))
            team_abbr = _safe_str(row.get("team"))
            team_id = team_id_map.get(team_abbr) if team_abbr else None
            jersey = _safe_int(row.get("jersey_number"))
            status_val = _safe_str(row.get("status"))

            existing: Optional[Player] = (
                db.query(Player).filter(Player.gsis_id == gsis_id).first()
            )

            if existing:
                existing.full_name = full_name
                existing.position = position
                existing.headshot_url = headshot_url
                existing.team_id = team_id
                existing.jersey_number = jersey
                existing.status = status_val
            else:
                player = Player(
                    gsis_id=gsis_id,
                    full_name=full_name,
                    position=position,
                    headshot_url=headshot_url,
                    team_id=team_id,
                    jersey_number=jersey,
                    status=status_val,
                )
                db.add(player)

            count += 1

            # Batch commit every 500 rows to avoid large transactions
            if count % 500 == 0:
                db.commit()
                logger.debug("Players: committed %d rows so far...", count)

        db.commit()
        logger.info("Players ingested for season %d: %d records.", season, count)
        _record_pipeline_run(
            db, f"ingest_players_{season}", PipelineStatus.success,
            records_processed=count, started_at=started_at
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Error committing players for season %d: %s", season, exc)
        _record_pipeline_run(
            db, f"ingest_players_{season}", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    return count


# ---------------------------------------------------------------------------
# 3. ingest_schedule
# ---------------------------------------------------------------------------

def ingest_schedule(db: Session, seasons: list[int]) -> int:
    """
    Fetch and upsert Game records for the specified seasons.

    Parameters
    ----------
    seasons : list[int]
        List of NFL season years.

    Returns
    -------
    int
        Number of records processed.
    """
    import nfl_data_py as nfl

    logger.info("Ingesting schedule for seasons: %s", seasons)
    started_at = datetime.utcnow()

    try:
        df: pd.DataFrame = nfl.import_schedules(seasons)
    except Exception as exc:  # noqa: BLE001
        logger.error("nfl_data_py.import_schedules(%s) failed: %s", seasons, exc)
        _record_pipeline_run(
            db, "ingest_schedule", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    if df is None or df.empty:
        logger.warning("import_schedules(%s) returned empty DataFrame.", seasons)
        return 0

    team_id_map = _get_team_id_map(db)
    count = 0

    try:
        for _, row in df.iterrows():
            game_id_str = _safe_str(row.get("game_id"))
            if not game_id_str:
                continue

            season = _safe_int(row.get("season"))
            week = _safe_int(row.get("week"))
            if season is None or week is None:
                continue

            game_type = _safe_str(row.get("game_type"))
            home_abbr = _safe_str(row.get("home_team"))
            away_abbr = _safe_str(row.get("away_team"))
            home_team_id = team_id_map.get(home_abbr) if home_abbr else None
            away_team_id = team_id_map.get(away_abbr) if away_abbr else None

            kickoff_time = _parse_kickoff(row.get("gameday"), row.get("gametime"))
            stadium = _safe_str(row.get("stadium"))
            home_score = _safe_int(row.get("home_score"))
            away_score = _safe_int(row.get("away_score"))
            home_spread = _safe_float(row.get("spread_line"))
            game_total = _safe_float(row.get("total_line"))
            surface = _safe_str(row.get("surface"))

            # Determine game status
            status = "final" if home_score is not None else "scheduled"

            existing: Optional[Game] = (
                db.query(Game).filter(Game.game_id_str == game_id_str).first()
            )

            if existing:
                existing.season = season
                existing.week = week
                existing.game_type = game_type
                existing.home_team_id = home_team_id
                existing.away_team_id = away_team_id
                existing.kickoff_time = kickoff_time
                existing.stadium = stadium
                existing.home_score = home_score
                existing.away_score = away_score
                existing.home_spread = home_spread
                existing.game_total = game_total
                existing.surface = surface
                existing.status = status
            else:
                game = Game(
                    game_id_str=game_id_str,
                    season=season,
                    week=week,
                    game_type=game_type,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    kickoff_time=kickoff_time,
                    stadium=stadium,
                    home_score=home_score,
                    away_score=away_score,
                    home_spread=home_spread,
                    game_total=game_total,
                    surface=surface,
                    status=status,
                )
                db.add(game)

            count += 1

            if count % 200 == 0:
                db.commit()
                logger.debug("Schedule: committed %d rows so far...", count)

        db.commit()
        logger.info("Schedule ingested for %s: %d records.", seasons, count)
        _record_pipeline_run(
            db, "ingest_schedule", PipelineStatus.success,
            records_processed=count, started_at=started_at
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Error committing schedule for %s: %s", seasons, exc)
        _record_pipeline_run(
            db, "ingest_schedule", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    return count


# ---------------------------------------------------------------------------
# 4. ingest_player_stats
# ---------------------------------------------------------------------------

# Columns to request from nfl_data_py weekly data
_WEEKLY_COLUMNS = [
    "player_id", "player_name", "position", "recent_team", "opponent_team",
    "season", "week",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "air_yards", "yards_after_catch",
    "snap_pct", "fantasy_points_ppr",
]


def ingest_player_stats(db: Session, seasons: list[int]) -> int:
    """
    Fetch weekly player stats and upsert PlayerGameStat records.

    Only processes stats for players and games already in the DB.
    Post-kickoff data ONLY (historical records) — never used for live prediction.

    Parameters
    ----------
    seasons : list[int]
        List of NFL season years.

    Returns
    -------
    int
        Number of records processed.
    """
    import nfl_data_py as nfl

    logger.info("Ingesting player stats for seasons: %s", seasons)
    started_at = datetime.utcnow()

    try:
        df: pd.DataFrame = nfl.import_weekly_data(years=seasons)
    except Exception as exc:  # noqa: BLE001
        logger.error("nfl_data_py.import_weekly_data(%s) failed: %s", seasons, exc)
        _record_pipeline_run(
            db, "ingest_player_stats", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    if df is None or df.empty:
        logger.warning("import_weekly_data(%s) returned empty DataFrame.", seasons)
        return 0

    # Build lookup caches for performance
    team_id_map = _get_team_id_map(db)
    player_id_map: dict[str, int] = {
        gsis: pid
        for gsis, pid in db.query(Player.gsis_id, Player.id)
        .filter(Player.gsis_id.isnot(None))
        .all()
    }

    count = 0
    skipped = 0

    try:
        for _, row in df.iterrows():
            gsis_id = _safe_str(row.get("player_id"))
            if not gsis_id:
                skipped += 1
                continue

            player_db_id = player_id_map.get(gsis_id)
            if not player_db_id:
                skipped += 1
                continue  # player not yet ingested

            season = _safe_int(row.get("season"))
            week = _safe_int(row.get("week"))
            if season is None or week is None:
                skipped += 1
                continue

            team_abbr = _safe_str(row.get("recent_team"))
            opp_abbr = _safe_str(row.get("opponent_team"))

            # Locate game by (season, week) and player's team
            game: Optional[Game] = (
                db.query(Game)
                .filter(
                    Game.season == season,
                    Game.week == week,
                )
                .filter(
                    (Game.home_team_id == team_id_map.get(team_abbr, -1))
                    | (Game.away_team_id == team_id_map.get(team_abbr, -1))
                )
                .first()
            )

            if not game:
                # Try without team filter as fallback
                skipped += 1
                continue

            team_db_id = team_id_map.get(team_abbr) if team_abbr else None
            opp_db_id = team_id_map.get(opp_abbr) if opp_abbr else None

            existing: Optional[PlayerGameStat] = (
                db.query(PlayerGameStat)
                .filter(
                    PlayerGameStat.player_id == player_db_id,
                    PlayerGameStat.game_id == game.id,
                )
                .first()
            )

            stat_kwargs = dict(
                season=season,
                week=week,
                team_id=team_db_id,
                opponent_id=opp_db_id,
                completions=_safe_int(row.get("completions")),
                attempts=_safe_int(row.get("attempts")),
                passing_yards=_safe_float(row.get("passing_yards")),
                passing_tds=_safe_int(row.get("passing_tds")),
                interceptions=_safe_int(row.get("interceptions")),
                carries=_safe_int(row.get("carries")),
                rushing_yards=_safe_float(row.get("rushing_yards")),
                rushing_tds=_safe_int(row.get("rushing_tds")),
                targets=_safe_int(row.get("targets")),
                receptions=_safe_int(row.get("receptions")),
                receiving_yards=_safe_float(row.get("receiving_yards")),
                receiving_tds=_safe_int(row.get("receiving_tds")),
                air_yards=_safe_float(row.get("air_yards")),
                yards_after_catch=_safe_float(row.get("yards_after_catch")),
                snap_pct=_safe_float(row.get("snap_pct")),
                fantasy_points=_safe_float(row.get("fantasy_points_ppr")),
            )

            if existing:
                for k, v in stat_kwargs.items():
                    setattr(existing, k, v)
            else:
                stat = PlayerGameStat(
                    player_id=player_db_id,
                    game_id=game.id,
                    **stat_kwargs,
                )
                db.add(stat)

            count += 1

            if count % 1000 == 0:
                db.commit()
                logger.debug("Player stats: committed %d rows (skipped %d)...", count, skipped)

        db.commit()
        logger.info(
            "Player stats ingested for %s: %d records upserted, %d skipped.",
            seasons, count, skipped
        )
        _record_pipeline_run(
            db, "ingest_player_stats", PipelineStatus.success,
            records_processed=count, started_at=started_at
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Error committing player stats for %s: %s", seasons, exc)
        _record_pipeline_run(
            db, "ingest_player_stats", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    return count


# ---------------------------------------------------------------------------
# 5. ingest_rosters
# ---------------------------------------------------------------------------

def ingest_rosters(db: Session, season: int, week: int) -> int:
    """
    Ingest weekly roster depth chart positions for a given season/week.

    Parameters
    ----------
    season : int
        NFL season year.
    week : int
        NFL week number.

    Returns
    -------
    int
        Number of records processed.
    """
    import nfl_data_py as nfl

    logger.info("Ingesting rosters for season %d week %d...", season, week)
    started_at = datetime.utcnow()

    try:
        df: pd.DataFrame = nfl.import_seasonal_rosters([season])
    except Exception as exc:  # noqa: BLE001
        logger.error("nfl_data_py.import_seasonal_rosters([%d]) failed: %s", season, exc)
        _record_pipeline_run(
            db, f"ingest_rosters_{season}_w{week}", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    if df is None or df.empty:
        logger.warning("import_rosters([%d]) returned empty DataFrame for rosters.", season)
        return 0

    team_id_map = _get_team_id_map(db)
    player_id_map: dict[str, int] = {
        gsis: pid
        for gsis, pid in db.query(Player.gsis_id, Player.id)
        .filter(Player.gsis_id.isnot(None))
        .all()
    }

    # Filter to the current week if a 'week' column exists
    if "week" in df.columns:
        df = df[df["week"] == week]
    # If no week column, treat the full roster as the current week's snapshot

    count = 0
    try:
        for _, row in df.iterrows():
            gsis_id = _safe_str(row.get("player_id"))
            if not gsis_id:
                continue

            player_db_id = player_id_map.get(gsis_id)
            if not player_db_id:
                continue

            team_abbr = _safe_str(row.get("team"))
            team_db_id = team_id_map.get(team_abbr) if team_abbr else None
            if not team_db_id:
                continue

            depth_chart_position = _safe_str(row.get("depth_chart_position"))
            depth_chart_rank = _safe_int(row.get("depth_chart_rank"))

            existing: Optional[Roster] = (
                db.query(Roster)
                .filter(
                    Roster.player_id == player_db_id,
                    Roster.team_id == team_db_id,
                    Roster.season == season,
                    Roster.week == week,
                )
                .first()
            )

            if existing:
                existing.depth_chart_position = depth_chart_position
                existing.depth_chart_rank = depth_chart_rank
            else:
                roster = Roster(
                    player_id=player_db_id,
                    team_id=team_db_id,
                    season=season,
                    week=week,
                    depth_chart_position=depth_chart_position,
                    depth_chart_rank=depth_chart_rank,
                )
                db.add(roster)

            count += 1

            if count % 500 == 0:
                db.commit()

        db.commit()
        logger.info("Rosters ingested for S%d W%d: %d records.", season, week, count)
        _record_pipeline_run(
            db, f"ingest_rosters_{season}_w{week}", PipelineStatus.success,
            records_processed=count, started_at=started_at
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Error committing rosters S%d W%d: %s", season, week, exc)
        _record_pipeline_run(
            db, f"ingest_rosters_{season}_w{week}", PipelineStatus.failure,
            error_message=str(exc), started_at=started_at
        )
        return 0

    return count


# ---------------------------------------------------------------------------
# 6. run_full_historical_ingestion
# ---------------------------------------------------------------------------

def run_full_historical_ingestion(
    start_season: int = 2015, end_season: Optional[int] = None
) -> dict:
    """
    Bootstrap the database with NFL data from start_season to end_season (inclusive).

    Order of operations:
      1. Teams (once)
      2. Schedule (all seasons at once)
      3. Players (per season)
      4. Player stats (all seasons at once)

    This function manages its own DB sessions and prints progress to stdout.

    Parameters
    ----------
    start_season : int
        First season to ingest (default: 2015).
    end_season : int | None
        Last season to ingest (default: current season).

    Returns
    -------
    dict
        {
          'teams': int,
          'schedule': int,
          'players': {season: int, ...},
          'player_stats': int,
          'errors': [str, ...],
        }
    """
    if end_season is None:
        end_season = _current_season()

    seasons = list(range(start_season, end_season + 1))
    summary: dict = {
        "teams": 0,
        "schedule": 0,
        "players": {},
        "player_stats": 0,
        "errors": [],
    }

    print(f"[PROPCAST] Starting full historical ingestion: {start_season}–{end_season}")

    # --- Step 1: Teams ---
    print("[PROPCAST] Step 1/4: Ingesting teams...")
    with SessionLocal() as db:
        try:
            summary["teams"] = ingest_teams(db)
            print(f"[PROPCAST]   Teams: {summary['teams']} records.")
        except Exception as exc:  # noqa: BLE001
            msg = f"Teams ingestion error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    # --- Step 2: Schedule ---
    print(f"[PROPCAST] Step 2/4: Ingesting schedule ({start_season}–{end_season})...")
    with SessionLocal() as db:
        try:
            summary["schedule"] = ingest_schedule(db, seasons)
            print(f"[PROPCAST]   Schedule: {summary['schedule']} records.")
        except Exception as exc:  # noqa: BLE001
            msg = f"Schedule ingestion error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    # --- Step 3: Players (per season) ---
    print("[PROPCAST] Step 3/4: Ingesting players (per season)...")
    for season in seasons:
        with SessionLocal() as db:
            try:
                n = ingest_players(db, season)
                summary["players"][season] = n
                print(f"[PROPCAST]   Season {season}: {n} players.")
            except Exception as exc:  # noqa: BLE001
                msg = f"Players ingestion error (season {season}): {exc}"
                logger.error(msg)
                summary["errors"].append(msg)
                summary["players"][season] = 0

    # --- Step 4: Player stats (all seasons at once) ---
    print(f"[PROPCAST] Step 4/4: Ingesting weekly player stats ({start_season}–{end_season})...")
    print("[PROPCAST]   (This may take several minutes and download significant data.)")
    with SessionLocal() as db:
        try:
            summary["player_stats"] = ingest_player_stats(db, seasons)
            print(f"[PROPCAST]   Player stats: {summary['player_stats']} records.")
        except Exception as exc:  # noqa: BLE001
            msg = f"Player stats ingestion error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    print("[PROPCAST] Full historical ingestion complete.")
    if summary["errors"]:
        print(f"[PROPCAST] Errors encountered: {len(summary['errors'])}")
        for err in summary["errors"]:
            print(f"  ⚠  {err}")

    return summary


# ---------------------------------------------------------------------------
# 7. run_current_week_ingestion
# ---------------------------------------------------------------------------

def run_current_week_ingestion() -> dict:
    """
    Run ingestion for the current NFL season and week.

    Suitable for the weekly refresh job (every Tuesday).

    Returns
    -------
    dict
        {
          'season': int,
          'week': int,
          'teams': int,
          'schedule': int,
          'players': int,
          'rosters': int,
          'player_stats': int,
          'errors': [str, ...],
        }
    """
    season = _current_season()
    week = _current_nfl_week(season)

    summary: dict = {
        "season": season,
        "week": week,
        "teams": 0,
        "schedule": 0,
        "players": 0,
        "rosters": 0,
        "player_stats": 0,
        "errors": [],
    }

    logger.info("Current week ingestion: Season %d, Week %d", season, week)

    # Teams
    with SessionLocal() as db:
        try:
            summary["teams"] = ingest_teams(db)
        except Exception as exc:  # noqa: BLE001
            msg = f"Teams error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    # Schedule (current season only)
    with SessionLocal() as db:
        try:
            summary["schedule"] = ingest_schedule(db, [season])
        except Exception as exc:  # noqa: BLE001
            msg = f"Schedule error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    # Players
    with SessionLocal() as db:
        try:
            summary["players"] = ingest_players(db, season)
        except Exception as exc:  # noqa: BLE001
            msg = f"Players error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    # Rosters
    with SessionLocal() as db:
        try:
            summary["rosters"] = ingest_rosters(db, season, week)
        except Exception as exc:  # noqa: BLE001
            msg = f"Rosters error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    # Player stats
    with SessionLocal() as db:
        try:
            summary["player_stats"] = ingest_player_stats(db, [season])
        except Exception as exc:  # noqa: BLE001
            msg = f"Player stats error: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

    logger.info("Current week ingestion complete: %s", summary)
    return summary
