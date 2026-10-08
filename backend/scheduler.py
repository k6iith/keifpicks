"""
PROPCAST – Background Task Scheduler
Configures recurring APScheduler cron jobs for automated data updates.
- Daily 12:00am Louisiana time (US Central): rosters, injuries, weather
  (NOT odds: see run_daily_updates), then the missed-week catch-up check
- Tuesday 6am: weekly schedule and roster refresh
- Wednesday 5am: model retraining against the newly-completed week (pulls
  the latest box scores first; also re-run after a restart if the live
  models are more than a week old)
- Wednesday 12:00pm (US Eastern by default): new-week refresh — schedule,
  box scores, injuries, odds, depth charts and predictions for the new week
- Friday 8am: mid-week injuries refresh + prediction regeneration
- Sunday 11:45am ET (10:45am Louisiana): game-day injury check
- Monday 3am: final game stats and model performance recalculation
"""
import functools
import gc
import logging
import threading
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from backend.config import settings
from backend.db.database import SessionLocal
from backend.db.models import Game, PipelineRun, PipelineStatus, Prediction
from backend.ingestion.injuries import ingest_injuries
from backend.ingestion.weather import ingest_weather_for_upcoming_games
from backend.ingestion.odds import ingest_market_lines
from backend.ingestion.nfl_data import ingest_schedule, ingest_players, ingest_player_stats
from backend.ingestion.espn_sync import sync_espn_rosters_and_depth_charts
from backend.models.performance import score_completed_games
from backend.models.retrain import latest_model_trained_at, restore_models_from_db, retrain_all_models

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

# The heavy jobs (feature building, prediction regeneration, retraining) each
# peak around 350-400 MB. On a 512 MB free instance two of them overlapping
# can run the process out of memory, which kills it mid-job with nothing in
# the logs but a restart. Only one heavy job runs at a time.
_heavy_job_lock = threading.Lock()

# Task names whose successful run produces the current week's predictions.
_WEEK_BUILDERS = ("weekly_refresh", "wednesday_refresh", "midweek_refresh", "catch_up_refresh", "manual_refresh")

# Task names that retrain the models.
_RETRAINERS = ("weekly_retraining", "manual_retraining", "catch_up_retraining")


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
        # One query for every player instead of one per roster row (~3,000):
        # each round trip keeps the hosted database awake and billed.
        ids = [pid for pid in df["player_id"].dropna().unique().tolist() if pid]
        by_gsis = {p.gsis_id: p for p in db.query(Player).filter(Player.gsis_id.in_(ids)).all()} if ids else {}
        for _, row in df.iterrows():
            pid = row.get("player_id")
            if not pid:
                continue
            player = by_gsis.get(pid)
            if not player:
                continue
            new_team = row.get("team")
            new_status = row.get("status")
            if new_team in teams and player.team_id != teams[new_team]:
                player.team_id = teams[new_team]
            if new_status and player.status != new_status:
                player.status = new_status
        db.commit()
        logger.info("Daily roster and status synchronization complete.")
    except Exception as exc:
        logger.warning("Roster synchronization skipped: %s", exc)


@_tracked("daily_updates", heavy=True)
def run_daily_updates():
    """
    Daily job (midnight US Central): injuries, weather, and roster/status
    synchronization.

    This ran every hour, which kept the hosted Postgres (Neon) awake and
    billed around the clock and exhausted the free plan's compute quota.
    Injuries are also refreshed by the Wednesday and Friday jobs.

    Sportsbook odds are deliberately NOT pulled here. Each odds pull costs
    ~96 Odds API credits (6 markets x 16 games); daily would be ~2,900 a
    month against a 500-credit plan. Odds are pulled once a week in the
    Wednesday refresh instead.
    """
    logger.info("⏰ Starting daily update (rosters, injuries, weather)...")
    db = SessionLocal()
    try:
        sync_hourly_rosters(db)

        injuries_count = ingest_injuries(db)
        logger.info("Daily injuries update: %d records", injuries_count)

        weather_count = ingest_weather_for_upcoming_games(db)
        logger.info("Daily weather update: %d records", weather_count)

    except Exception as exc:
        logger.error("Error during daily update: %s", exc)
    finally:
        db.close()


@_tracked("sunday_injuries", heavy=True)
def run_sunday_injury_check():
    """
    Game-day injury check, Sunday 11:45 AM ET. Teams post official inactives
    90 minutes before kickoff, so this catches the 1:00 PM games (most of the
    slate). Only pulls the injury report: the props API hides anyone whose
    latest report rules them out right away, so no prediction rebuild (and
    no heavy database work) is needed.
    """
    logger.info("⏰ Starting Sunday game-day injury check...")
    db = SessionLocal()
    try:
        count = ingest_injuries(db)
        logger.info("Sunday injuries update: %d records", count)
    finally:
        db.close()


def run_daily_job():
    """Midnight job: daily updates, then the missed-week and stale-model catch-up checks."""
    run_daily_updates()
    run_catch_up_if_stale()
    run_retrain_if_stale()


def run_startup_checks():
    """
    A few minutes after startup: restore retrained models wiped by the
    restart, retrain if they're stale anyway, then make sure this week has
    predictions (a catch-up retrain rebuilds them itself).
    """
    restore_saved_models()
    run_retrain_if_stale()
    run_catch_up_if_stale()


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
    from backend.ingestion.nfl_data import _current_season

    db = SessionLocal()
    try:
        # Season, not calendar year: January/February games belong to the
        # previous year's season.
        year = _current_season()
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


def _pull_odds_unless_fresh(db) -> str:
    """Pull sportsbook lines, unless this week's were pulled in the last 24h (saves ~96 credits)."""
    from backend.ingestion.odds import week_lines_are_fresh

    if week_lines_are_fresh(db):
        logger.info("Skipping odds pull: this week's lines were pulled within the last 24 hours.")
        return "skipped (lines pulled within 24h)"
    return f"{ingest_market_lines(db)} lines"


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
        ("odds", _pull_odds_unless_fresh),
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


def _retrain(label: str, regenerate_predictions: bool) -> list:
    """
    Retrain every prop model against all data on file, including whatever
    week just finished. Each model only goes live if it's not meaningfully
    worse than the one it would replace — see retrain_all_models()'s
    docstring for the guardrail, backup and persistence mechanics.
    Returns a list of errors.
    """
    from backend.ingestion.nfl_data import _current_season

    logger.info("⏰ Starting %s model retraining...", label)
    errors: list = []
    db = SessionLocal()
    try:
        # Pull box scores first: Tuesday's 6am run can land before
        # Monday-night stats are published, and those games are exactly
        # what the retrain is supposed to learn from.
        try:
            n = ingest_player_stats(db, [_current_season()])
            logger.info("Pre-retrain player stats refresh: %d records", n)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.error("Pre-retrain stats refresh failed (retraining on what's on file): %s", exc)
            errors.append(f"player stats: {type(exc).__name__}: {exc}")

        results = retrain_all_models(db)
        logger.info("Retraining results: %s", results)
    finally:
        db.close()
    gc.collect()

    if results.get("status") in ("error", "INSUFFICIENT_DATA"):
        errors.append(f"retrain: {results.get('error') or results['status']}")
        return errors
    for prop, r in results.items():
        if isinstance(r, dict) and r.get("status") == "error":
            errors.append(f"{prop}: {r.get('error')}")
    if results.get("persist_error"):
        errors.append(f"saving models: {results['persist_error']}")

    accepted = [p for p, r in results.items() if isinstance(r, dict) and r.get("status") == "accepted"]
    if accepted and regenerate_predictions:
        # Off-schedule retrains don't have the Wednesday noon refresh coming
        # right after them, so rebuild this week's predictions now.
        try:
            sync_espn_rosters_and_depth_charts()
        except Exception as exc:  # noqa: BLE001
            logger.error("Prediction refresh after %s retrain failed: %s", label, exc)
            errors.append(f"predictions: {type(exc).__name__}: {exc}")
    return errors


@_tracked("weekly_retraining", heavy=True)
def run_retraining_job():
    """
    Wednesday 5am: runs the morning after the Tuesday refresh, so the
    just-completed week is in player_game_stats. Predictions are rebuilt by
    the Wednesday noon refresh.
    """
    return _retrain("scheduled", regenerate_predictions=False)


@_tracked("manual_retraining", heavy=True, skip_if_busy=True)
def run_manual_retraining():
    """Same as the weekly retrain, started from /admin, then rebuilds predictions."""
    return _retrain("manual", regenerate_predictions=True)


def start_manual_retraining() -> bool:
    """Start a retrain in the background. False if a heavy job is already running."""
    if _heavy_job_lock.locked():
        return False
    threading.Thread(target=run_manual_retraining, name="manual-retrain", daemon=True).start()
    return True


@_tracked("catch_up_retraining", heavy=True, skip_if_busy=True)
def _run_catch_up_retraining():
    return _retrain("catch-up", regenerate_predictions=True)


def restore_saved_models() -> None:
    """Put models from past retrains back on disk (it's wiped on restart/redeploy)."""
    db = SessionLocal()
    try:
        restore_models_from_db(db)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("Could not restore saved models from the database: %s", exc)
    finally:
        db.close()


def run_retrain_if_stale():
    """
    Self-heal: if the live models are over a week old while games are being
    played (the Wednesday retrain was missed because the process was asleep
    or restarting, or it ran but couldn't save), retrain now.

    Won't fire if a retrain started in the last 24 hours, so a retrain the
    guardrail keeps rejecting (which leaves the old model in place) runs at
    most once a day instead of on every restart.
    """
    trained_at = latest_model_trained_at()
    now = datetime.utcnow()
    if trained_at is not None and now - trained_at < timedelta(days=7):
        return
    db = SessionLocal()
    try:
        recent_final = (
            db.query(Game.id)
            .filter(Game.status == "final", Game.kickoff_time >= now - timedelta(days=10))
            .first()
        )
        recent_retrain = (
            db.query(PipelineRun.id)
            .filter(
                PipelineRun.task_name.in_(_RETRAINERS),
                PipelineRun.started_at >= now - timedelta(hours=24),
            )
            .first()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stale-model check failed: %s", exc)
        return
    finally:
        db.close()
    if recent_final is None or recent_retrain is not None:
        return
    logger.warning("Live models were last trained %s: running catch-up retrain.", trained_at)
    _run_catch_up_retraining()


def start_scheduler():
    """Initialize and start background jobs."""
    if not scheduler.running:
        _mark_interrupted_runs()

        # Self-heal checks once, a few minutes after startup (lets a
        # first-time database seed finish). They also run at the end of the
        # daily job. Not hourly: every check wakes the hosted database.
        scheduler.add_job(
            run_startup_checks,
            DateTrigger(run_date=datetime.now() + timedelta(minutes=3)),
            id="catch_up_check",
            replace_existing=True,
        )

        # Daily at 12:00 AM Louisiana time (US Central; follows DST).
        scheduler.add_job(
            run_daily_job,
            CronTrigger(hour=0, minute=0, timezone=settings.daily_job_timezone),
            id="daily_updates",
            replace_existing=True,
            misfire_grace_time=3 * 60 * 60,
            coalesce=True,
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
            misfire_grace_time=6 * 60 * 60,
            coalesce=True,
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

        # Sunday 11:45 AM ET game-day injury check (inactives come out 90
        # minutes before the 1:00 PM kickoffs).
        scheduler.add_job(
            run_sunday_injury_check,
            CronTrigger(day_of_week="sun", hour=11, minute=45, timezone=settings.scheduler_timezone),
            id="sunday_injuries",
            replace_existing=True,
            misfire_grace_time=60 * 60,
            coalesce=True,
        )

        # Monday 3:00 AM scoring
        scheduler.add_job(
            run_scoring_job,
            CronTrigger(day_of_week="mon", hour=3, minute=0),
            id="monday_scoring",
            replace_existing=True,
        )

        scheduler.start()
        logger.info("✅ APScheduler started: %d scheduled jobs.", len(scheduler.get_jobs()))


def shutdown_scheduler():
    """Graceful shutdown of scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 APScheduler shut down.")
