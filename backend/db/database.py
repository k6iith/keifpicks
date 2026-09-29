"""
PROPCAST – Database Engine & Session Factory
Supports SQLite (dev) and PostgreSQL (prod) via DATABASE_URL env var.
"""
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# Hosted Postgres providers hand out "postgres://..." or "postgresql://..."
# URLs. SQLAlchemy rejects the first, and the default driver for the second
# differs between SQLAlchemy versions, so pin the installed psycopg (v3) driver.
_database_url = settings.database_url
for _prefix in ("postgres://", "postgresql://"):
    if _database_url.startswith(_prefix):
        _database_url = "postgresql+psycopg://" + _database_url[len(_prefix):]
        break

# PostgreSQL returns SUM()/COALESCE() over integer columns as NUMERIC, which
# psycopg hands back as decimal.Decimal; the feature code does pandas/numpy
# math on those results and breaks on Decimal (SQLite returned floats). Load
# NUMERIC as float everywhere instead: nothing in this app needs exact
# decimal arithmetic.
if _database_url.startswith("postgresql+psycopg://"):
    import psycopg
    from psycopg.types.numeric import FloatLoader

    psycopg.adapters.register_loader("numeric", FloatLoader)

_connect_args: dict = {}
if "sqlite" in _database_url:
    # SQLite: disable the same-thread check so FastAPI workers can share
    _connect_args["check_same_thread"] = False

engine = create_engine(
    _database_url,
    connect_args=_connect_args,
    # Echo SQL in development; silence in production
    echo=(settings.environment == "development"),
    # Pool settings (SQLite ignores these; good for Postgres later)
    pool_pre_ping=True,
)

# Enable WAL mode for SQLite to improve concurrent read performance
if "sqlite" in _database_url:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---------------------------------------------------------------------------
# Declarative base (shared by all models)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def get_db():
    """Yield a SQLAlchemy session and close it when done."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table creation helper (called at startup)
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Import all models so SQLAlchemy knows about them, then create tables."""
    from backend.db import models  # noqa: F401 – side-effect import

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")
