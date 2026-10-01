"""
PROPCAST – Background Task Scheduler
Configures recurring APScheduler cron jobs for automated data updates.
- Hourly: injuries, weather (NOT odds: see run_hourly_updates)
- Tuesday 6am: weekly schedule and roster refresh
- Wednesday 5am: model retraining against the newly-completed week
- Wednesday 12:00pm (US Eastern by default): new-week refresh — schedule,
  box scores, injuries, odds, depth charts and predictions for the new week
- Friday 8am: mid-week injuries refresh + prediction regeneration
- Monday 3am: final game stats and model performance recalculation
"""
import functools
import gc
import logging
import threading
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.config import settings
from backend.db.database import SessionLocal
from backend.db.models import Game, PipelineRun, PipelineStatus, Prediction
from backend.ingestion.injuries import ingest_injuries
from backend.ingestion.weather import ingest_weather_for_upcoming_games
from backend.ingestion.odds import ingest_market_lines
from backend.ingestion.nfl_data import ingest_schedule, ingest_players, ingest_player_stats
from backend.ingestion.espn_sync import sync_espn_rosters_and_depth_charts
from backend.models.performance import score_completed_games
from backend.models.retrain import retrain_all_models

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

# The heavy jobs (feature building, prediction regeneration, retraining) each
# peak around 350-400 MB. On a 512 MB free instance two of them overlapping
# can run the process out of memory, which kills it mid-job with nothing in
# the logs but a restart. Only one heavy job runs at a time.
_heavy_job_lock = threading.Lock()

# Task names whose successful run produces the current week's predictions.
_WEEK_BUILDERS = ("weekly_refresh", "wednesday_refresh", "midweek_refresh", "catch_up_refresh", "manual_refresh")


# ---------------------------------------------------------------------------
# Run tracking (pipeline_runs table), so job outcomes survive restarts and
# show on /admin. A run left as "running" after a restart was interrupted —
# almost always the process being killed (out of memory or a redeploy).
# ---------------------------------------------------------------------------

def _start_run(task_name: str):
    db = SessionLocal()
    try:
        run = PipelineRun(task_name=task_name, started_at=datetime.utcnow(), status=PipelineStatus.running)
        db.add(run)
        db.commit()
        return run.id
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not record start of %s: %s", task_name, exc)
        return None
    finally:
        db.close()


def _finish_run(run_id, errors: list) -> None:
    if run_id is None:
        return
    db = SessionLocal()
    try:
        run = db.get(PipelineRun, run_id)
        if run is not None:
            run.completed_at = datetime.utcnow()
            run.status = PipelineStatus.failure if errors else PipelineStatus.success
            run.error_message = "; ".join(errors)[:2000] if errors else None
            db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not record end of run %s: %s", run_id, exc)
    finally:
        db.close()


def _tracked(task_name: str, heavy: bool = False, skip_if_busy: bool = False):
    """
    Record a job's start/end/errors in pipeline_runs. A job may return a list
    of error strings for steps it handled itself. Heavy jobs take the heavy-job
    lock; with skip_if_busy they skip instead of waiting for it.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if heavy:
                if not _heavy_job_lock.acquire(blocking=not skip_if_busy):
                    logger.info("Skipping %s: another heavy job is running.", task_name)
                    return None
            run_id = _start_run(task_name)
            errors: list = []
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, list):
                    errors = result
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s failed: %s", task_name, exc)
                errors = [f"{type(exc).__name__}: {exc}"]
            finally:
                _finish_run(run_id, errors)
                if heavy:
                    _heavy_job_lock.release()
                gc.collect()
            return errors
        return wrapper
    return decorator


def _mark_interrupted_runs() -> None:
    """On startup, close out runs a previous process never finished."""
    db = SessionLocal()
    try:
        stale = (
            db.query(PipelineRun)
            .filter(PipelineRun.status == PipelineStatus.running, PipelineRun.completed_at.is_(None))
            .all()
        )
        for run in stale:
            run.status = PipelineStatus.failure
            run.completed_at = datetime.utcnow()
            run.error_message = (
                "Interrupted: the server restarted before this job finished "
                "(usually out of memory, or a redeploy while it ran)."
            )
        if stale:
            db.commit()
            logger.warning("Marked %d interrupted job run(s) as failed.", len(stale))
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not check for interrupted runs: %s", exc)
    finally:
        db.close()


def current_week_prediction_count():
    """(season, week, number of current predictions for that week's games)."""
    from backend.ingestion.nfl_data import _current_nfl_week, _current_season

    season = _current_season()
    db = SessionLocal()
    try:
        week = _current_nfl_week(season, db)
        n = (
            db.query(Prediction.id)
            .join(Game, Game.id == Prediction.game_id)
            .filter(Game.season == season, Game.week == week, Prediction.is_current.is_(True))
            .count()
        )
        has_games = db.query(Game.id).filter(Game.season == season, Game.week == week).first() is not None
        return season, week, n, has_games
    finally:
        db.close()


def sync_hourly_rosters(db: SessionLocal):
    """Sync rosters, trade moves, and depth charts from official NFL telemetry."""
    try:
        import nfl_data_py as nfl
        from datetime import datetime
        from backend.db.models import Player, Team, Roster

        # Was hardcoded to [2024], so every hourly sync kept re-applying a
        # frozen 2024 roster snapshot no matter what year it actually ran in.
        current_year = datetime.utcnow().year

        teams = {t.abbreviation: t.id for t in db.query(Team).all()}
        df = nfl.import_seasonal_rosters([current_year])
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


@_tracked("hourly_updates", heavy=True, skip_if_busy=True)
def run_hourly_updates():
    """
    Execute hourly jobs: injuries, weather, and roster/status synchronization.

    Sportsbook odds are deliberately NOT pulled here. Each odds pull costs
    ~96 Odds API credits (6 markets x 16 games); hourly that's ~2,300 a day
    against a 500-credit monthly plan. Odds are pulled once a week in the
    Wednesday refresh instead.
    """
    logger.info("⏰ Starting scheduled hourly update (rosters, injuries, weather)...")
    db = SessionLocal()
    try:
        sync_hourly_rosters(db)

        injuries_count = ingest_injuries(db)
        logger.info("Hourly injuries update: %d records", injuries_count)

        weather_count = ingest_weather_for_upcoming_games(db)
        logger.info("Hourly weather update: %d records", weather_count)

    except Exception as exc:
        logger.error("Error during scheduled hourly update: %s", exc)
    finally:
        db.close()


@_tracked("weekly_refresh", heavy=True)
def run_weekly_refresh():
    """
    Execute weekly schedule, roster, and player-stats refresh.

    Previously this only refreshed the schedule and player list — it never
    actually pulled the prior week's box-score stats into player_game_stats,
    so every player's "recent form" features silently stopped updating after
    the last season that got backfilled by hand. ingest_player_stats() is
    what actually keeps that table current, so it belongs here too.
    """
    logger.info("⏰ Starting scheduled weekly refresh...")
    db = SessionLocal()
    try:
        from datetime import datetime
        year = datetime.utcnow().year
        ingest_schedule(db, [year])
        ingest_players(db, year)
        stats_count = ingest_player_stats(db, [year])
        logger.info("Weekly player stats refresh: %d records", stats_count)
    except Exception as exc:
        logger.error("Error during weekly refresh: %s", exc)
    finally:
        db.close()

    # Depth charts + prediction regeneration: separate step (manages its own
    # DB session/commits) so a failure here doesn't roll back the ingestion
    # above. This also used to only ever run when someone manually triggered
    # an ESPN sync — predictions never actually refreshed on their own.
    try:
        sync_espn_rosters_and_depth_charts()
    except Exception as exc:
        logger.error("Error during weekly depth-chart/prediction refresh: %s", exc)


def _new_week_refresh(label: str) -> list:
    """
    Roll the site over to the current week. Returns a list of errors.

    By Wednesday midday Monday night's game is final, the week's first
    official injury report is out, and lines for the new week are posted.
    This pulls all of that and rebuilds predictions for the current week:
      1. schedule (game statuses, spreads/totals for the new week)
      2. box scores (Tuesday's 6am UTC run can land before Monday-night
         stats are published)
      3. injury reports and sportsbook lines
      4. depth charts + prediction regeneration for the current week
    and then checks that the week actually has predictions, because the
    ESPN/prediction step logs its own failures instead of raising them.
    """
    logger.info("⏰ Starting %s new-week refresh...", label)
    from backend.ingestion.nfl_data import _current_season

    errors: list = []
    season = _current_season()
    steps = [
        ("schedule", lambda db: ingest_schedule(db, [season])),
        ("player stats", lambda db: ingest_player_stats(db, [season])),
        ("injuries", ingest_injuries),
        ("odds", ingest_market_lines),
    ]
    db = SessionLocal()
    try:
        # Each step on its own, so e.g. an injury-feed outage doesn't also
        # skip pulling the new week's lines.
        for name, step in steps:
            try:
                result = step(db)
                logger.info("%s %s refresh: %s", label, name, result)
            except Exception as exc:
                db.rollback()
                logger.error("Error during %s %s refresh: %s", label, name, exc)
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    finally:
        db.close()
    gc.collect()

    # Separate step with its own session, same as the other refreshes, so an
    # ingestion failure above doesn't block rebuilding the week's predictions.
    try:
        sync_espn_rosters_and_depth_charts()
    except Exception as exc:
        logger.error("Error during %s depth-chart/prediction refresh: %s", label, exc)
        errors.append(f"depth charts/predictions: {type(exc).__name__}: {exc}")

    try:
        season, week, n, _ = current_week_prediction_count()
        logger.info("%s refresh: week %d now has %d current predictions.", label, week, n)
        if n == 0:
            errors.append(f"No predictions were generated for week {week} (see the ESPN sync lines in the logs).")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"verification: {exc}")
    return errors


@_tracked("wednesday_refresh", heavy=True)
def run_wednesday_refresh():
    """Wednesday noon refresh: roll the site over to the new week."""
    return _new_week_refresh("Wednesday")


@_tracked("manual_refresh", heavy=True, skip_if_busy=True)
def run_manual_refresh():
    """Same as the Wednesday refresh, started from /admin."""
    return _new_week_refresh("Manual")


def start_manual_refresh() -> bool:
    """Start the new-week refresh in the background. False if a heavy job is already running."""
    if _heavy_job_lock.locked():
        return False
    threading.Thread(target=run_manual_refresh, name="manual-refresh", daemon=True).start()
    return True


@_tracked("catch_up_refresh", heavy=True, skip_if_busy=True)
def _run_catch_up_refresh():
    return _new_week_refresh("Catch-up")


def run_catch_up_if_stale():
    """
    Self-heal: if the current week has games but no predictions (the
    scheduled refresh never ran or died partway, e.g. the process was
    restarted or ran out of memory), run the new-week refresh now.

    Won't fire if any week-building job started in the last 6 hours, so a
    refresh that keeps crashing can't turn into a restart loop.
    """
    try:
        season, week, n, has_games = current_week_prediction_count()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Catch-up check failed: %s", exc)
        return
    if not has_games or n > 0:
        return
    db = SessionLocal()
    try:
        recent = (
            db.query(PipelineRun.id)
            .filter(
                PipelineRun.task_name.in_(_WEEK_BUILDERS),
                PipelineRun.started_at >= datetime.utcnow() - timedelta(hours=6),
            )
            .first()
        )
    finally:
        db.close()
    if recent:
        logger.info("Week %d has no predictions, but a refresh ran in the last 6h; not retrying yet.", week)
        return
    logger.warning("Week %d has no predictions yet: running catch-up refresh.", week)
    _run_catch_up_refresh()


@_tracked("midweek_refresh", heavy=True)
def run_midweek_refresh():
    """
    Friday morning refresh: re-sync injuries and regenerate predictions
    from whatever's changed since Tuesday, without re-pulling the full
    schedule/player-stats tables (that's what Tuesday's run_weekly_refresh
    is for).

    Predictions previously only ever refreshed once a week, on Tuesday.
    That's fine right after a game week ends, but by Friday — after a full
    week of practice reports, questionable/out designations firming up, and
    lines moving — Tuesday's predictions can be noticeably stale. This
    doesn't touch box-score stats (there's nothing new there mid-week); it
    just pulls the latest injury picture and rebuilds this week's
    predictions against it. (Odds are not re-pulled here: the weekly Odds API
    budget only covers the Wednesday pull. Lines stored Wednesday are reused.)
    """
    logger.info("⏰ Starting scheduled mid-week refresh...")
    db = SessionLocal()
    try:
        injuries_count = ingest_injuries(db)
        logger.info("Mid-week injuries update: %d records", injuries_count)
    except Exception as exc:
        logger.error("Error during mid-week refresh: %s", exc)
    finally:
        db.close()

    # Same reasoning as run_weekly_refresh: keep this in its own step/session
    # so a failure ingesting injuries/odds above doesn't block the
    # depth-chart/prediction regeneration below, and vice versa.
    try:
        sync_espn_rosters_and_depth_charts()
    except Exception as exc:
        logger.error("Error during mid-week depth-chart/prediction refresh: %s", exc)


@_tracked("monday_scoring", heavy=True)
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


@_tracked("weekly_retraining", heavy=True)
def run_retraining_job():
    """
    Retrain every prop model against all data on file, including whatever
    week just finished. Each model only goes live if it's not meaningfully
    worse than the one it would replace — see retrain_all_models()'s
    docstring for the guardrail and backup mechanics. Runs the morning
    after the Tuesday refresh so the just-completed week's stats are
    already in player_game_stats by the time this runs.
    """
    logger.info("⏰ Starting scheduled model retraining...")
    db = SessionLocal()
    try:
        results = retrain_all_models(db)
        logger.info("Retraining results: %s", results)
    except Exception as exc:
        logger.error("Error during retraining job: %s", exc)
    finally:
        db.close()


def start_scheduler():
    """Initialize and start background jobs."""
    if not scheduler.running:
        _mark_interrupted_runs()

        # Self-heal check: a few minutes after startup (lets a first-time
        # database seed finish), then hourly.
        scheduler.add_job(
            run_catch_up_if_stale,
            IntervalTrigger(hours=1, start_date=datetime.now() + timedelta(minutes=3)),
            id="catch_up_check",
            replace_existing=True,
        )

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

        # Wednesday 5:00 AM model retraining (after Tuesday's stats refresh)
        scheduler.add_job(
            run_retraining_job,
            CronTrigger(day_of_week="wed", hour=5, minute=0),
            id="weekly_retraining",
            replace_existing=True,
        )

        # Wednesday 12:00 PM new-week refresh (injuries, lines, new week's
        # predictions). Pinned to settings.scheduler_timezone (US Eastern by
        # default) rather than the server clock, which is UTC on Render.
        scheduler.add_job(
            run_wednesday_refresh,
            CronTrigger(day_of_week="wed", hour=12, minute=0, timezone=settings.scheduler_timezone),
            id="wednesday_refresh",
            replace_existing=True,
            misfire_grace_time=3 * 60 * 60,
            coalesce=True,
        )

        # Friday 8:00 AM mid-week refresh (injuries/odds firm up by Friday;
        # this regenerates predictions against the freshest picture without
        # waiting for next Tuesday)
        scheduler.add_job(
            run_midweek_refresh,
            CronTrigger(day_of_week="fri", hour=8, minute=0),
            id="midweek_refresh",
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
        logger.info("✅ APScheduler started with 7 background pipelines.")


def shutdown_scheduler():
    """Graceful shutdown of scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 APScheduler shut down.")
