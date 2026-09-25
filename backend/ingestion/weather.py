"""
PROPCAST – Open-Meteo Weather Ingestion
Fetches weather data for NFL game venues using the free Open-Meteo API.
No API key required.  Handles dome stadiums, historical games, and forecasts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from backend.db.models import Game, Weather

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stadium reference data
# ---------------------------------------------------------------------------

# (latitude, longitude) for each NFL stadium
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Allegiant Stadium": (36.0909, -115.1833),
    "Arrowhead Stadium": (39.0489, -94.4839),
    "AT&T Stadium": (32.7479, -97.0931),
    "Bank of America Stadium": (35.2258, -80.8528),
    "Caesars Superdome": (29.9511, -90.0812),
    "Century Link Field": (47.5952, -122.3316),
    "Empower Field": (39.7439, -105.0201),
    "EverBank Stadium": (30.3239, -81.6373),
    "FedExField": (38.9076, -76.8644),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "Ford Field": (42.3400, -83.0456),
    "Gillette Stadium": (42.0909, -71.2643),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "Highmark Stadium": (42.7738, -78.7870),
    "Huntington Bank Field": (41.5061, -81.6995),
    "Lambeau Field": (44.5013, -88.0622),
    "Levi's Stadium": (37.4033, -121.9694),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "M&T Bank Stadium": (39.2780, -76.6227),
    "Mercedes-Benz Stadium": (33.7553, -84.4006),
    "MetLife Stadium": (40.8128, -74.0742),
    "NRG Stadium": (29.6847, -95.4107),
    "Nissan Stadium": (36.1665, -86.7713),
    "Paycor Stadium": (39.0954, -84.5161),
    "Raymond James Stadium": (27.9759, -82.5033),
    "SoFi Stadium": (33.9535, -118.3392),
    "Soldier Field": (41.8623, -87.6167),
    "State Farm Stadium": (33.5277, -112.2626),
    "TIAA Bank Field": (30.3239, -81.6373),
    "US Bank Stadium": (44.9736, -93.2575),
}

# Fully enclosed / climate-controlled stadiums — weather is irrelevant
DOME_STADIUMS: set[str] = {
    "Allegiant Stadium",
    "AT&T Stadium",
    "Caesars Superdome",
    "Ford Field",
    "Lucas Oil Stadium",
    "Mercedes-Benz Stadium",
    "State Farm Stadium",
    "US Bank Stadium",
    "NRG Stadium",
}

# ---------------------------------------------------------------------------
# Open-Meteo endpoints
# ---------------------------------------------------------------------------
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

# WMO weather interpretation codes → human-readable condition labels
_WMO_CONDITIONS: dict[int, str] = {
    0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy Fog",
    51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Moderate Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Moderate Snow", 75: "Heavy Snow", 77: "Snow Grains",
    80: "Light Showers", 81: "Moderate Showers", 82: "Violent Showers",
    85: "Slight Snow Showers", 86: "Heavy Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ Hail", 99: "Thunderstorm w/ Heavy Hail",
}

_HTTP_TIMEOUT = 20.0

# Celsius to Fahrenheit
def _c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32

# km/h to mph
def _kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371

# mm to inches
def _mm_to_in(mm: float) -> float:
    return mm / 25.4

# Wind degrees → cardinal direction
def _wind_degrees_to_direction(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


def _find_stadium_coords(stadium_name: Optional[str]) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for the stadium, doing a flexible partial-key match."""
    if not stadium_name:
        return None
    # Exact match first
    if stadium_name in STADIUM_COORDS:
        return STADIUM_COORDS[stadium_name]
    # Partial match (handles slight naming variants)
    sl = stadium_name.lower()
    for key, coords in STADIUM_COORDS.items():
        if key.lower() in sl or sl in key.lower():
            return coords
    return None


def _is_dome(stadium_name: Optional[str]) -> bool:
    """Return True if the stadium is a dome / fully enclosed."""
    if not stadium_name:
        return False
    if stadium_name in DOME_STADIUMS:
        return True
    sl = stadium_name.lower()
    for dome in DOME_STADIUMS:
        if dome.lower() in sl:
            return True
    return False


def _parse_hourly_weather(data: dict, target_hour: int) -> dict:
    """
    Extract weather values at the hour closest to kickoff from an
    Open-Meteo hourly response.

    Parameters
    ----------
    data : dict
        Parsed Open-Meteo JSON response.
    target_hour : int
        The hour-of-day index (0-23) of kickoff in UTC.

    Returns
    -------
    dict with keys: temperature_f, wind_mph, wind_direction, humidity_pct,
                    precipitation, condition
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return {}

    # Find the best matching index
    idx = min(range(len(times)), key=lambda i: abs(i % 24 - target_hour))

    def _safe(key: str, default=None):
        vals = hourly.get(key, [])
        return vals[idx] if idx < len(vals) else default

    temp_c = _safe("temperature_2m")
    wind_kmh = _safe("windspeed_10m")
    wind_dir_deg = _safe("winddirection_10m")
    humidity = _safe("relativehumidity_2m")
    precip_mm = _safe("precipitation")
    wmo_code = _safe("weathercode")

    return {
        "temperature_f": round(_c_to_f(temp_c), 1) if temp_c is not None else None,
        "wind_mph": round(_kmh_to_mph(wind_kmh), 1) if wind_kmh is not None else None,
        "wind_direction": _wind_degrees_to_direction(wind_dir_deg) if wind_dir_deg is not None else None,
        "humidity_pct": round(float(humidity), 1) if humidity is not None else None,
        "precipitation": round(_mm_to_in(precip_mm), 3) if precip_mm is not None else None,
        "condition": _WMO_CONDITIONS.get(int(wmo_code), "Unknown") if wmo_code is not None else None,
    }


def fetch_weather_for_game(game: Game) -> Optional[dict]:
    """
    Retrieve weather data for a single game from Open-Meteo.

    Uses the *archive* API for historical games and *forecast* API for
    upcoming games (within the 16-day forecast window).

    Parameters
    ----------
    game : Game
        SQLAlchemy Game instance with ``stadium`` and ``kickoff_time`` populated.

    Returns
    -------
    dict | None
        Keys: temperature_f, wind_mph, wind_direction, humidity_pct,
              precipitation, condition, is_dome.
        Returns None if weather cannot be determined.
    """
    dome = _is_dome(game.stadium)

    if dome:
        logger.debug("Game %s is at a dome — returning dome record.", game.game_id_str)
        return {
            "temperature_f": 72.0,
            "wind_mph": 0.0,
            "wind_direction": "N/A",
            "humidity_pct": 50.0,
            "precipitation": 0.0,
            "condition": "Dome (Controlled)",
            "is_dome": True,
        }

    coords = _find_stadium_coords(game.stadium)
    if not coords:
        logger.debug(
            "No coordinates found for stadium '%s' (game %s) — skipping weather.",
            game.stadium,
            game.game_id_str,
        )
        return None

    if not game.kickoff_time:
        logger.debug("Game %s has no kickoff_time — skipping weather.", game.game_id_str)
        return None

    lat, lon = coords
    kickoff_utc = game.kickoff_time
    if kickoff_utc.tzinfo is not None:
        kickoff_utc = kickoff_utc.astimezone(timezone.utc).replace(tzinfo=None)

    now_utc = datetime.utcnow()
    days_until_game = (kickoff_utc.date() - now_utc.date()).days
    date_str = kickoff_utc.strftime("%Y-%m-%d")

    common_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation,weathercode,windspeed_10m,winddirection_10m",
        "timezone": "UTC",
    }

    # Select historical vs forecast endpoint
    if days_until_game < 0:
        # Past game — use archive API
        url = _HISTORICAL_URL
        params = {**common_params, "start_date": date_str, "end_date": date_str}
    elif days_until_game <= 16:
        # Upcoming game within forecast window
        url = _FORECAST_URL
        params = {**common_params, "forecast_days": min(days_until_game + 1, 16)}
    else:
        logger.debug(
            "Game %s is >16 days away (%d days) — weather not available yet.",
            game.game_id_str,
            days_until_game,
        )
        return None

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Open-Meteo API HTTP %s for game %s.",
            exc.response.status_code,
            game.game_id_str,
        )
        return None
    except httpx.RequestError as exc:
        logger.warning(
            "Open-Meteo request error (%s) for game %s.",
            type(exc).__name__,
            game.game_id_str,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected weather fetch error for game %s: %s", game.game_id_str, exc)
        return None

    weather = _parse_hourly_weather(data, target_hour=kickoff_utc.hour)
    if not weather:
        return None

    weather["is_dome"] = False
    return weather


def ingest_weather_for_upcoming_games(db: Session) -> int:
    """
    Fetch and upsert weather for all games with status 'scheduled' or
    games within the next 16 days that don't already have fresh weather.

    Parameters
    ----------
    db : Session
        Active SQLAlchemy session.

    Returns
    -------
    int
        Number of weather records upserted (0 on any failure).
    """
    now_utc = datetime.utcnow()
    cutoff = now_utc + timedelta(days=16)

    try:
        # Include 'scheduled' games and any recent game without weather data
        games_to_update = (
            db.query(Game)
            .filter(
                Game.status == "scheduled",
                Game.kickoff_time <= cutoff,
            )
            .all()
        )
        # Also refresh dome games that may be missing a Weather row
        total_games = len(games_to_update)
        logger.info("Weather ingestion: processing %d upcoming games.", total_games)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error querying games for weather ingestion: %s", exc)
        return 0

    count = 0
    for game in games_to_update:
        try:
            weather_data = fetch_weather_for_game(game)
            if weather_data is None:
                continue

            existing: Optional[Weather] = (
                db.query(Weather).filter(Weather.game_id == game.id).first()
            )

            if existing:
                for field, value in weather_data.items():
                    setattr(existing, field, value)
                existing.fetched_at = datetime.utcnow()
            else:
                weather_row = Weather(
                    game_id=game.id,
                    fetched_at=datetime.utcnow(),
                    **weather_data,
                )
                db.add(weather_row)

            count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error processing weather for game %s: %s", game.game_id_str, exc)
            continue

    try:
        db.commit()
        logger.info("Weather ingestion complete: %d records upserted.", count)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Weather ingestion commit failed: %s", exc)
        return 0

    return count
