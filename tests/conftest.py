"""
Shared setup for the CI checks in .github/workflows/.

Every test runs against a throwaway copy of the bundled propcast.db, never the
repo file itself, and never against a hosted database: DATABASE_URL is set
here, before anything imports backend.config.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="keifpicks-ci-"))
shutil.copy(ROOT / "propcast.db", _tmp / "propcast.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'propcast.db'}"
os.environ["ENVIRONMENT"] = "test"
os.environ["ADMIN_KEY"] = "ci-test-key"
os.environ["ODDS_API_KEY"] = ""

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # Not used as a context manager, so the app's startup (seeding, the
    # scheduler and its refresh jobs) never runs in CI.
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


@pytest.fixture()
def db():
    from backend.db.database import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
