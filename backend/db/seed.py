"""
PROPCAST – One-time seed of a fresh PostgreSQL database from propcast.db.

On Render's free plan the disk is ephemeral: anything the scheduled jobs write
to the bundled SQLite file (injuries, lines, refreshed predictions) is lost on
every restart or deploy. Pointing DATABASE_URL at a hosted PostgreSQL fixes
that, and this module fills the new, empty database with everything already
in the repo's propcast.db the first time the app starts against it, so
nothing has to be run by hand.

It only runs when the target database is not SQLite and has no teams yet; any
later start sees data and does nothing.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from sqlalchemy import create_engine, func, select, text

from backend.db.database import Base, engine

logger = logging.getLogger(__name__)

SQLITE_SEED_PATH = Path(__file__).resolve().parents[2] / "propcast.db"
_CHUNK = 5000


def seed_from_sqlite_if_empty(sqlite_path: Path = SQLITE_SEED_PATH) -> int:
    """Copy every table from propcast.db into an empty non-SQLite database. Returns rows copied."""
    from backend.db import models  # noqa: F401 – register tables

    if engine.dialect.name == "sqlite":
        return 0
    if not sqlite_path.exists():
        logger.warning("Seed skipped: %s not found.", sqlite_path)
        return 0

    teams = Base.metadata.tables["teams"]
    with engine.connect() as conn:
        if conn.execute(select(func.count()).select_from(teams)).scalar():
            return 0

    logger.info("Empty %s database detected: seeding from %s ...", engine.dialect.name, sqlite_path.name)
    started = time.time()
    src = create_engine(f"sqlite:///{sqlite_path}")
    total = 0
    # One transaction: if anything fails the target stays empty and the seed
    # is simply retried on the next start.
    with engine.begin() as dst, src.connect() as s:
        src_tables = {r[0] for r in s.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        for table in Base.metadata.sorted_tables:  # parents before children
            if table.name not in src_tables:
                continue
            src_cols = {r[1] for r in s.execute(text(f"PRAGMA table_info({table.name})"))}
            cols = [c for c in table.columns if c.name in src_cols]
            result = s.execution_options(yield_per=_CHUNK).execute(select(*cols))
            n = 0
            for chunk in result.partitions(_CHUNK):
                dst.execute(table.insert(), [dict(r._mapping) for r in chunk])
                n += len(chunk)
            total += n
            logger.info("  seeded %-20s %7d rows", table.name, n)

        # Rows were inserted with their original ids; move each id sequence
        # past them so new inserts don't collide.
        if engine.dialect.name == "postgresql":
            for table in Base.metadata.sorted_tables:
                if "id" in table.columns and table.name in src_tables:
                    dst.execute(text(
                        f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                        f"COALESCE((SELECT MAX(id) FROM {table.name}), 0) + 1, false)"
                    ))

    logger.info("✅ Seeded %d rows from %s in %.0fs.", total, sqlite_path.name, time.time() - started)
    return total


def seed_in_background() -> None:
    """Run the seed off the startup path so Render sees the port open right away."""
    def _run() -> None:
        try:
            seed_from_sqlite_if_empty()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Seeding from propcast.db failed (will retry on next start): %s", exc)

    threading.Thread(target=_run, name="sqlite-seed", daemon=True).start()
