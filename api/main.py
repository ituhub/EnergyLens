"""
EnergyLens — FastAPI backend API.

Serves spot prices, forecasts, pipeline health, and quality metrics
to the React dashboard frontend.

Run locally: uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "energylens"))

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.database import Database
from api.forecast_service import ForecastService

app = FastAPI(
    title="EnergyLens API",
    description="Energy market forecasting platform API",
    version="0.2.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()
forecast_svc = ForecastService(db_path="data/energylens.db", model_dir="models")


@app.on_event("startup")
async def startup():
    try:
        db.initialize()
    except Exception as e:
        import logging
        logging.warning(f"Database init skipped (not critical for forecasts): {e}")


# ═══════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """Pipeline health overview."""
    counts = db.get_record_counts()
    ml_status = forecast_svc.status()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": counts,
        "ml_models": ml_status,
    }


# ═══════════════════════════════════════════════════════════════════════
# SPOT PRICES (Phase 1)
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/prices")
async def get_prices(
    zone: str = Query("DK1", description="Bidding zone (DK1, DK2)"),
    days: int = Query(7, ge=1, le=365, description="Days of history"),
):
    """Get spot prices for a bidding zone."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    records = db.get_prices_as_of(
        zone=zone,
        valid_start=start.isoformat(),
        valid_end=end.isoformat(),
    )

    return {
        "zone": zone,
        "count": len(records),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "records": records,
    }


@app.get("/api/prices/latest")
async def get_latest_prices():
    """Get the most recent prices for all active zones."""
    zones = ["DK1", "DK2"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=48)

    result = {}
    for zone in zones:
        records = db.get_prices_as_of(
            zone=zone,
            valid_start=start.isoformat(),
            valid_end=end.isoformat(),
        )
        result[zone] = records[-24:] if len(records) > 24 else records

    return result


@app.get("/api/prices/compare")
async def compare_zones(
    days: int = Query(2, ge=1, le=30),
):
    """Compare DK1 vs DK2 prices for the chart overlay."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    result = {}
    for zone in ["DK1", "DK2"]:
        records = db.get_prices_as_of(
            zone=zone,
            valid_start=start.isoformat(),
            valid_end=end.isoformat(),
        )
        result[zone] = records

    return result


# ═══════════════════════════════════════════════════════════════════════
# FORECASTS (Phase 3)
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/forecast")
async def get_forecast(
    zone: str = Query("DK1", description="Bidding zone"),
    hours: int = Query(24, ge=1, le=72, description="Forecast horizon in hours"),
):
    """
    Generate a price forecast for the next N hours.

    Returns hourly predicted prices with confidence scoring and
    per-model breakdown from the 8-model neural ensemble.
    """
    result = forecast_svc.forecast(zone=zone, hours=hours)

    if result.get("error"):
        raise HTTPException(
            status_code=503,
            detail=result["error"],
        )

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
    return {
        "loaded": True,
        "zones": status,
    }


@app.post("/api/forecast/reload")
async def reload_models(
    zone: Optional[str] = Query(None, description="Zone to reload, or all if omitted"),
):
    """Force-reload models from disk (e.g. after retraining)."""
    forecast_svc.reload_models(zone)
    return {"status": "reloaded", "zone": zone or "all"}
