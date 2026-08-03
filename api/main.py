"""
EnergyLens — FastAPI backend API.

Serves spot prices, forecasts, pipeline health, and quality metrics
to the React dashboard frontend.

v0.5.0 — Adds: Firebase Auth, Firestore user registration, admin panel,
          structured JSON logging, pipeline tracing.

Data freshness logic:
  - Price endpoints first try the requested date range (e.g. last 48h)
  - If no rows match (data is stale), falls back to the most recent
    N rows available and marks the response as data_status="stale"
  - If data matches the requested range, data_status="fresh"
  - The dashboard and logs always show which mode is active

Run locally: uvicorn api.main:app --reload --port 8000
"""

import sys
import asyncio
import sqlite3
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "energylens"))

from datetime import datetime, timedelta, timezone
from typing import Optional
from api.accuracy_routes import register_accuracy_routes

from fastapi import FastAPI, Query, HTTPException, Depends
from pipeline.ingest import IngestionPipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.forecast_service import ForecastService

# --- Structured logging (replaces basicConfig) --------------------------------
from api.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger("energylens.api")

# --- Auth & Admin imports -----------------------------------------------------
from api.auth import get_current_user, require_auth, require_admin, increment_prediction_count
from api.admin import register_admin_routes


app = FastAPI(
    title="EnergyLens API",
    description="Energy market forecasting platform API",
    version="0.5.0",
)

# CORS — allow React dev server and deployed frontend
import os
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",") + ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/energylens.db"

# Restore database from GCS on startup (survives redeploys)
try:
    from api.gcs_sync import download_db_from_gcs
    _db_restore = download_db_from_gcs(DB_PATH)
    logger.info(f"DB restore: {_db_restore.get('status')}")
except Exception as _e:
    logger.warning(f"DB restore skipped: {_e}")

forecast_svc = ForecastService(db_path=DB_PATH, model_dir="models")

# Accuracy tracking & SHAP explainability
accuracy_engine, shap_engine = register_accuracy_routes(app)

# Admin routes
register_admin_routes(app)


# --- SQLite helpers -----------------------------------------------------------

def query_db(sql: str, params: tuple = ()) -> list[dict]:
    """Run a query against SQLite and return list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_data_bounds() -> dict:
    """Get the min/max valid_time and total count in spot_prices."""
    rows = query_db(
        "SELECT MIN(valid_time) as oldest, MAX(valid_time) as newest, COUNT(*) as total FROM spot_prices"
    )
    if rows and rows[0]["total"] > 0:
        return rows[0]
    return {"oldest": None, "newest": None, "total": 0}


def get_zone_prices(zone: str, start_iso: str, end_iso: str) -> tuple[list[dict], str]:
    """
    Fetch prices for a zone within a date range.
    If no rows match, falls back to the most recent data available.
    Returns (records, data_status) where data_status is 'fresh' or 'stale'.
    """
    # First try: exact date range
    records = query_db(
        """
        SELECT valid_time, zone, price_eur_mwh, price_dkk_mwh
        FROM spot_prices
        WHERE zone = ? AND valid_time >= ? AND valid_time <= ?
        ORDER BY valid_time ASC
        """,
        (zone, start_iso, end_iso),
    )

    if records:
        logger.info(
            f"[FRESH] {zone}: {len(records)} rows from {records[0]['valid_time']} "
            f"to {records[-1]['valid_time']}"
        )
        return records, "fresh"

    # Fallback: get the most recent N rows regardless of date
    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)
        requested_hours = max(int((end_dt - start_dt).total_seconds() / 3600), 24)
    except Exception:
        requested_hours = 48

    records = query_db(
        """
        SELECT valid_time, zone, price_eur_mwh, price_dkk_mwh
        FROM spot_prices
        WHERE zone = ?
        ORDER BY valid_time DESC
        LIMIT ?
        """,
        (zone, requested_hours),
    )
    records = list(reversed(records))  # chronological order

    if records:
        bounds = get_data_bounds()
        logger.warning(
            f"[STALE] {zone}: No data in requested range ({start_iso[:10]} to {end_iso[:10]}). "
            f"Falling back to latest {len(records)} rows. "
            f"DB newest: {bounds['newest']}, DB oldest: {bounds['oldest']}. "
            f"Run auto_refresh.py or download fresh data!"
        )
        return records, "stale"

    logger.error(f"[EMPTY] {zone}: No data at all in spot_prices table!")
    return [], "empty"


def get_record_counts() -> dict:
    """Get row counts for all data tables."""
    counts = {}
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        for (table,) in cursor:
            if table == "predictions":
                continue
            count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            counts[table] = count
    finally:
        conn.close()
    return counts


# --- Startup ------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    db = Path(DB_PATH)
    if not db.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return

    bounds = get_data_bounds()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    if bounds["total"] == 0:
        logger.warning("spot_prices table is EMPTY — dashboard will show no data")
    else:
        newest = bounds["newest"]
        logger.info(f"Database: {bounds['total']} spot price rows")
        logger.info(f"Date range: {bounds['oldest']} -> {newest}")

        # Check staleness
        try:
            newest_dt = datetime.fromisoformat(newest.replace("T", " ").split("+")[0])
            age_hours = (datetime.utcnow() - newest_dt).total_seconds() / 3600
            if age_hours > 48:
                logger.warning(
                    f"DATA IS STALE — newest record is {age_hours:.0f}h old ({newest}). "
                    f"Charts will use fallback mode. Run: python auto_refresh.py"
                )
            else:
                logger.info(f"Data is FRESH — newest record is {age_hours:.1f}h old")
        except Exception:
            pass

    counts = get_record_counts()
    logger.info(f"All tables: {counts}")

    # Auto-refresh on cold start so we never serve stale baked-in data
    asyncio.create_task(_startup_refresh())


async def _startup_refresh():
    """Run ingestion pipeline on cold start with gap-aware spot price fetch."""
    await asyncio.sleep(3)
    logger.info("Cold-start refresh: checking for data gaps...")
    try:
        # Detect how stale spot prices are
        bounds = get_data_bounds()
        days_back = 7  # default
        if bounds["newest"]:
            try:
                newest_str = bounds["newest"].replace("T", " ").split("+")[0]
                newest_dt = datetime.fromisoformat(newest_str)
                gap_days = (datetime.utcnow() - newest_dt).total_seconds() / 86400
                if gap_days > 2:
                    days_back = min(int(gap_days) + 2, 30)
                    logger.info(f"Spot price gap: {gap_days:.1f} days — fetching {days_back} days")
            except Exception:
                pass

        pipeline = IngestionPipeline()
        # Widen the spot price fetch for this startup run
        pipeline._spot_days_override = days_back
        await pipeline.run_latest()
        logger.info(f"Cold-start refresh complete — fetched {days_back} days of spot prices")
    except Exception as e:
        logger.warning(f"Cold-start refresh error: {e}")


# ==============================================================================
# AUTH
# ==============================================================================

@app.get("/api/auth/me")
async def auth_me(user=Depends(require_auth)):
    """Return the authenticated user's profile (role, email, etc.)."""
    return user


# ==============================================================================
# HEALTH (public — no auth required)
# ==============================================================================

@app.get("/api/health")
async def health():
    """Pipeline health overview."""
    counts = get_record_counts()
    bounds = get_data_bounds()
    ml_status = forecast_svc.status()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": counts,
        "data_range": {
            "oldest": bounds["oldest"],
            "newest": bounds["newest"],
            "total_spot_prices": bounds["total"],
        },
        "ml_models": ml_status,
    }

@app.get("/api/pipeline/status")
async def pipeline_status():
    """Return last-refresh timestamps and record counts for each data pipeline."""
    import sqlite3
    try:
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "energylens.db")
        if not os.path.exists(db_path):
            db_path = "/tmp/energylens.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        status = {}

        # Nord Pool — Spot Prices
        cursor.execute("SELECT COUNT(*) FROM spot_prices")
        spot_count = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(valid_time) FROM spot_prices")
        spot_latest_data = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(created_at) FROM spot_prices")
        spot_last_refresh = cursor.fetchone()[0]
        status["nordpool"] = {
            "name": "Nord Pool — Spot prices",
            "records": spot_count,
            "latest_data": spot_latest_data,
            "last_refresh": spot_last_refresh,
        }

        # Open-Meteo — Weather
        cursor.execute("SELECT COUNT(*) FROM weather_forecasts")
        weather_count = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(valid_time) FROM weather_forecasts")
        weather_latest = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(created_at) FROM weather_forecasts")
        weather_refresh = cursor.fetchone()[0]
        status["weather"] = {
            "name": "Open-Meteo — Weather",
            "records": weather_count,
            "latest_data": weather_latest,
            "last_refresh": weather_refresh,
        }

        # ENTSO-E — Generation
        cursor.execute("SELECT COUNT(*) FROM generation")
        gen_count = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(valid_time) FROM generation")
        gen_latest = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(created_at) FROM generation")
        gen_refresh = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT generation_type) FROM generation")
        gen_types = cursor.fetchone()[0]
        status["entsoe"] = {
            "name": "ENTSO-E — Generation",
            "records": gen_count,
            "latest_data": gen_latest,
            "last_refresh": gen_refresh,
            "generation_types": gen_types,
        }

        # Quality Gate
        cursor.execute("SELECT COUNT(*) FROM quality_quarantine")
        quarantine_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM forecast_log")
        forecast_count = cursor.fetchone()[0]

        status["quality_gate"] = {
            "name": "Quality gate",
            "forecast_log_count": forecast_count,
            "quarantined": quarantine_count,
        }

        conn.close()
        return {"status": "ok", "pipelines": status}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================================================================
# SPOT PRICES — with stale-data fallback (public for dashboard loading)
# ==============================================================================

@app.get("/api/prices")
async def get_prices(
    zone: str = Query("DK1", description="Bidding zone (DK1, DK2)"),
    days: int = Query(7, ge=1, le=365, description="Days of history"),
):
    """Get spot prices for a bidding zone."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    records, data_status = get_zone_prices(zone, start.isoformat(), end.isoformat())

    return {
        "zone": zone,
        "count": len(records),
        "data_status": data_status,
        "start": records[0]["valid_time"] if records else start.isoformat(),
        "end": records[-1]["valid_time"] if records else end.isoformat(),
        "records": records,
    }


@app.get("/api/prices/latest")
async def get_latest_prices():
    """Get the most recent 24 prices for all active zones."""
    result = {}
    for zone in ["DK1", "DK2"]:
        records = query_db(
            """
            SELECT valid_time, zone, price_eur_mwh, price_dkk_mwh
            FROM spot_prices
            WHERE zone = ?
            ORDER BY valid_time DESC
            LIMIT 24
            """,
            (zone,),
        )
        records = list(reversed(records))
        if records:
            logger.info(f"/prices/latest {zone}: {len(records)} rows, newest={records[-1]['valid_time']}")
        result[zone] = records

    return result


@app.get("/api/prices/compare")
async def compare_zones(
    days: int = Query(2, ge=1, le=365),
):
    """Compare DK1 vs DK2 prices for the chart overlay."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    result = {}
    data_status = "fresh"

    for zone in ["DK1", "DK2"]:
        records, zone_status = get_zone_prices(zone, start.isoformat(), end.isoformat())
        result[zone] = records
        if zone_status != "fresh":
            data_status = zone_status

    result["_meta"] = {
        "data_status": data_status,
        "requested_days": days,
        "note": "STALE: showing most recent available data" if data_status == "stale" else "Showing live data",
    }

    return result


# ==============================================================================
# FORECASTS (Phase 3) — authenticated, tracks prediction count
# ==============================================================================

@app.get("/api/forecast")
async def forecast(
    zone: str = "DK1",
    hours: int = 24,
    user=Depends(get_current_user),  # Optional auth — works with or without token
):
    result = forecast_svc.forecast(zone=zone, hours=hours)

    # Track prediction count for authenticated users
    if user and "error" not in result:
        try:
            increment_prediction_count(user["uid"])
        except Exception:
            pass

    # Log predictions for accuracy tracking
    try:
        accuracy_engine.log_forecast(
            zone=zone,
            forecasts=result.get("forecasts", []),
            per_model=result.get("per_model", {}),
            confidence=result.get("confidence", 0)
        )
    except Exception as e:
        logger.warning(f"AccuracyLog: {e}")

    return result


@app.get("/api/forecast/models")
async def get_forecast_models():
    """Show loaded model status and ensemble weights."""
    status = forecast_svc.status()
    if not status:
        return {
            "loaded": False,
            "message": "No models loaded yet. They load on first forecast request.",
            "zones": {},
        }
    return {"loaded": True, "zones": status}


@app.post("/api/forecast/reload")
async def reload_models(
    zone: Optional[str] = Query(None, description="Zone to reload, or all if omitted"),
    user=Depends(require_admin),  # Admin only
):
    """Pull latest models from GCS, then reload from disk."""
    from api.gcs_sync import sync_models_from_gcs
    gcs_result = sync_models_from_gcs(local_dir="models", zone=zone)
    forecast_svc.reload_models(zone)
    logger.info(
        f"Models reloaded by admin {user['email']}",
        extra={"event": "model_reload", "zone": zone or "all"},
    )
    return {"status": "reloaded", "zone": zone or "all", "gcs_sync": gcs_result}


# ==============================================================================
# DATA REFRESH (called by Cloud Scheduler — no auth, uses service account)
# ==============================================================================

@app.post("/api/refresh")
async def trigger_refresh():
    """Trigger full data refresh — spot prices + weather + ENTSO-E generation.
    Also generates and logs forecasts so accuracy/backtest accumulate data."""
    try:
        pipeline = IngestionPipeline()
        await pipeline.run_latest()
        health = pipeline.get_health()

        # Generate and log forecasts for accuracy tracking
        forecast_results = {}
        for zone in ["DK1"]:
            try:
                result = forecast_svc.forecast(zone=zone, hours=24)
                if "error" not in result:
                    accuracy_engine.log_forecast(
                        zone=zone,
                        forecasts=result.get("forecasts", []),
                        per_model=result.get("per_model", {}),
                        confidence=result.get("confidence", 0)
                    )
                    forecast_results[zone] = {
                        "logged": True,
                        "confidence": result.get("confidence"),
                        "models_used": result.get("models_used"),
                    }
                    logger.info(f"Refresh: logged {zone} forecast, confidence={result.get('confidence')}%")
                else:
                    forecast_results[zone] = {"logged": False, "error": result["error"]}
            except Exception as e:
                logger.warning(f"Refresh: {zone} forecast failed: {e}")
                forecast_results[zone] = {"logged": False, "error": str(e)}

        # Backup database to GCS after successful refresh
        try:
            from api.gcs_sync import upload_db_to_gcs
            db_backup = upload_db_to_gcs(DB_PATH)
            logger.info(f"DB backup: {db_backup.get('status')}")
        except Exception as backup_err:
            logger.warning(f"DB backup skipped: {backup_err}")
            db_backup = {"status": "skipped"}

        return {
            "status": "ok",
            "sources": health,
            "forecast_logging": forecast_results,
            "db_backup": db_backup,
            "message": "Full pipeline refresh complete (spot + weather + ENTSO-E + forecast logged + DB backed up)"
        }
    except Exception as e:
        logger.error(f"Pipeline refresh failed: {e}")
        # Fallback to spot-only refresh
        result = subprocess.run(
            ["python", "auto_refresh.py"],
            capture_output=True, text=True, timeout=120
        )
        return {
            "status": "partial",
            "message": f"Pipeline failed ({e}), fell back to spot-only refresh",
            "stdout": result.stdout[-500:] if result.stdout else "",
        }


# ==============================================================================
# STATIC FILES — Serve React build (MUST be last — catch-all route)
# ==============================================================================

STATIC_DIR = Path("/app/static")

if STATIC_DIR.exists():
    # Serve JS/CSS/images from /assets/
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_react(path: str):
        """Serve React app — catch-all for non-API routes."""
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
