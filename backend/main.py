"""
PROPCAST – FastAPI Application Entry Point
Configures middleware, mounts all routers, and initialises the database on startup.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.db.database import init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("propcast")


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("🏈 PROPCAST API starting up (env=%s)", settings.environment)
    try:
        init_db()
        logger.info("✅ Database initialised successfully.")
    except Exception as exc:
        logger.error("❌ Database initialisation failed: %s", exc)

    try:
        from backend.scheduler import start_scheduler, shutdown_scheduler
        start_scheduler()
    except Exception as exc:
        logger.error("Failed to start scheduler: %s", exc)

    yield

    try:
        from backend.scheduler import shutdown_scheduler
        shutdown_scheduler()
    except Exception as exc:
        logger.error("Failed to shut down scheduler: %s", exc)

    logger.info("🏈 PROPCAST API shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="KEIFS PICKS API",
    description=(
        "Professional NFL player prop ML analytics platform. "
        "All projections are statistical estimates — see /api/props/* disclaimers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
_cors_origins = (
    ["*"]
    if settings.environment == "development"
    else [
        "https://propcast.io",
        "https://www.propcast.io",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "status": "NOT_FOUND",
            "message": f"Resource not found: {request.url.path}",
        },
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.exception("Unhandled 500 error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "SERVER_ERROR",
            "message": "An internal server error occurred. Please try again later.",
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from backend.api.routes import (  # noqa: E402 – imported after app creation
    games,
    health,
    injuries,
    model_performance,
    players,
    props,
)

app.include_router(health.router, prefix="/api")
app.include_router(games.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(props.router, prefix="/api")
app.include_router(injuries.router, prefix="/api")
app.include_router(model_performance.router, prefix="/api")


# ---------------------------------------------------------------------------
# Frontend Dashboard & Static Mount
# ---------------------------------------------------------------------------
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", tags=["Dashboard"])
def serve_dashboard():
    """Serve the PROPCAST NFL Player Prop Analytics Web Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "PROPCAST API Live",
        "docs": "/docs",
        "disclaimer": (
            "Model probabilities are statistical estimates, not guarantees. "
            "Sports outcomes contain substantial randomness."
        ),
    }
