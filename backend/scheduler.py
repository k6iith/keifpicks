"""
PROPCAST – Background Task Scheduler
Configures recurring APScheduler cron jobs for automated data updates.
- Hourly: injuries, weather, odds
- Tuesday 6am: weekly schedule and roster refresh
- Monday 3am: final game stats and model performance recalculation
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.db.database import SessionLocal
from backend.ingestion.injuries import ingest_injuries
from backend.ingestion.weather import ingest_weather_for_upcoming_games
from backend.ingestion.odds import ingest_market_lines
from backend.ingestion.nfl_data import ingest_schedule, ingest_players
from backend.models.performance import score_completed_games

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def sync_hourly_rosters(db: SessionLocal):
    """Sync rosters, trade moves, and depth charts from official NFL telemetry."""
    try:
        import nfl_data_py as nfl
        from backend.db.models import Player, Team, Roster
        
        teams = {t.abbreviation: t.id for t in db.query(Team).all()}
        df = nfl.import_seasonal_rosters([2024])
        for _, row in df.iterrows():
            pid = row.get("player_id")
            if not pid:
                continue
            player = db.query(Player).filter(Player.gsis_id == pid).first()
            if not player:
                continue
            new_team = row.get("team")
            new_status = row.get("status")
            if new_team in teams and player.team_id != teams[new_team]:
                player.team_id = teams[new_team]
            if new_status and player.status != new_status:
                player.status = new_status
        db.commit()
        logger.info("Hourly roster and status synchronization complete.")
    except Exception as exc:
        logger.warning("Roster synchronization skipped: %s", exc)


def run_hourly_updates():
    """Execute hourly jobs: injuries, weather, odds, and roster/status synchronization."""
    logger.info("⏰ Starting scheduled hourly update (rosters, injuries, weather, odds)...")
    db = SessionLocal()
    try:
        sync_hourly_rosters(db)

        injuries_count = ingest_injuries(db)
        logger.info("Hourly injuries update: %d records", injuries_count)

        weather_count = ingest_weather_for_upcoming_games(db)
        logger.info("Hourly weather update: %d records", weather_count)

        odds_count = ingest_market_lines(db)
        logger.info("Hourly odds update: %d records", odds_count)
    except Exception as exc:
        logger.error("Error during scheduled hourly update: %s", exc)
    finally:
        db.close()


def run_weekly_refresh():
    """Execute weekly schedule and roster refresh."""
    logger.info("⏰ Starting scheduled weekly refresh...")
    db = SessionLocal()
    try:
        from datetime import datetime
        year = datetime.utcnow().year
        ingest_schedule(db, [year])
        ingest_players(db, year)
    except Exception as exc:
        logger.error("Error during weekly refresh: %s", exc)
    finally:
        db.close()


def run_scoring_job():
    """Score completed games and recalculate model metrics."""
    logger.info("⏰ Starting post-game evaluation and scoring...")
    db = SessionLocal()
    try:
        res = score_completed_games(db)
        logger.info("Scoring completed: %s", res)
    except Exception as exc:
        logger.error("Error during scoring job: %s", exc)
    finally:
        db.close()


def start_scheduler():
    """Initialize and start background jobs."""
    if not scheduler.running:
        # Every hour
        scheduler.add_job(
            run_hourly_updates,
            IntervalTrigger(hours=1),
            id="hourly_updates",
            replace_existing=True,
        )

        # Weekly roster/schedule refresh: Tuesday 6:00 AM
        scheduler.add_job(
            run_weekly_refresh,
            CronTrigger(day_of_week="tue", hour=6, minute=0),
            id="weekly_refresh",
            replace_existing=True,
        )

        # Monday 3:00 AM scoring
        scheduler.add_job(
            run_scoring_job,
            CronTrigger(day_of_week="mon", hour=3, minute=0),
            id="monday_scoring",
            replace_existing=True,
        )

        scheduler.start()
        logger.info("✅ APScheduler started with 3 background pipelines.")


def shutdown_scheduler():
    """Graceful shutdown of scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 APScheduler shut down.")
