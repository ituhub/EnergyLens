"""
EnergyLens — Forecast service.

Bridges the FastAPI layer with the ML module.
  • Lazy-loads trained models on first request
  • Pulls the latest data from SQLite for feature building
  • Returns structured forecasts with confidence scoring

Usage from FastAPI:
    from api.forecast_service import ForecastService
    svc = ForecastService(db_path="data/energylens.db", model_dir="models")
    result = svc.forecast(zone="DK1", hours=24)
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from ml.training import load_models
from ml.features import build_energy_features
from ml.ensemble import ensemble_predict, multi_step_forecast, calculate_confidence

logger = logging.getLogger(__name__)


class ForecastService:
    """Stateful service that caches loaded models between requests."""

    def __init__(self, db_path: str = "data/energylens.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = model_dir
        self._cache: dict[str, dict] = {}   # zone → {models, config, scaler, loaded_at}
        self._lock = Lock()

    # ── Model loading (lazy, cached) ─────────────────────────────────

    def _load_zone(self, zone: str) -> dict | None:
        """Load models for a zone, caching the result."""
        with self._lock:
            cached = self._cache.get(zone)
            if cached is not None:
                return cached

            logger.info(f"Loading models for zone {zone} from {self.model_dir}/")
            models, config = load_models(zone=zone, base_dir=self.model_dir)

            if not models:
                logger.warning(f"No trained models found for zone {zone}")
                return None

            scaler = config.pop("scaler", None)
            entry = {
                "models": models,
                "config": config,
                "scaler": scaler,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
            self._cache[zone] = entry
            logger.info(f"Cached {len(models)} models for {zone}")
            return entry

    def reload_models(self, zone: str | None = None):
        """Force-reload models (e.g. after retraining)."""
        with self._lock:
            if zone:
                self._cache.pop(zone, None)
            else:
                self._cache.clear()

    # ── Data loading ─────────────────────────────────────────────────

    def _load_recent_data(self, zone: str, lookback_hours: int = 200) -> pd.DataFrame | None:
        """
        Load the most recent data from SQLite for feature building.
        We need enough history for the rolling windows (168h weekly lag).
        """
        db = Path(self.db_path)
        if not db.exists():
            logger.error(f"Database not found: {db}")
            return None

        conn = sqlite3.connect(str(db))

        try:
            prices_df = pd.read_sql_query(
                """
                SELECT HourUTC, SpotPriceEUR, SpotPriceDKK
                FROM spot_prices
                WHERE PriceArea = ?
                ORDER BY HourUTC DESC
                LIMIT ?
                """,
                conn,
                params=(zone, lookback_hours),
                parse_dates=["HourUTC"],
            )
            prices_df = prices_df.sort_values("HourUTC").set_index("HourUTC")
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
            conn.close()
            return None

        try:
            weather_df = pd.read_sql_query(
                """
                SELECT time, temperature_2m, wind_speed_10m, shortwave_radiation
                FROM weather_data
                ORDER BY time DESC
                LIMIT ?
                """,
                conn,
                params=(lookback_hours,),
                parse_dates=["time"],
            )
            weather_df = weather_df.sort_values("time").set_index("time")
            weather_hourly = weather_df.resample("h").mean()
        except Exception:
            weather_hourly = pd.DataFrame()

        conn.close()

        if not weather_hourly.empty:
            merged = prices_df.join(weather_hourly, how="left")
        else:
            merged = prices_df

        merged = merged.sort_index().dropna(subset=["SpotPriceEUR"])
        logger.info(f"Loaded {len(merged)} recent rows for {zone}")
        return merged

    # ── Core forecast method ─────────────────────────────────────────

    def forecast(self, zone: str = "DK1", hours: int = 24) -> dict:
        """
        Generate an hourly price forecast.

        Returns:
            {
                "zone": str,
                "hours": int,
                "forecasts": [{"hour": int, "timestamp_utc": str, "price_eur": float}, ...],
                "current_price": float,
                "confidence": float,
                "models_used": int,
                "models_total": int,
                "per_model": {name: price, ...},
                "price_range": [min, max],
                "generated_at": str,
            }
        """
        # 1. Load models
        entry = self._load_zone(zone)
        if entry is None:
            return self._error_response(zone, hours, "No trained models found. Run: python -m ml.run_training")

        models = entry["models"]
        config = entry["config"]
        scaler = entry["scaler"]

        time_step = config.get("time_step", 48)
        used_features = config.get("used_features", [])
        price_range = config.get("price_range")
        cv_weights = config.get("ensemble_weights")

        # 2. Load recent data
        # We need time_step + buffer for feature building (lags up to 168h)
        raw_df = self._load_recent_data(zone, lookback_hours=max(time_step + 200, 400))
        if raw_df is None or len(raw_df) < time_step + 10:
            return self._error_response(
                zone, hours,
                f"Insufficient data ({len(raw_df) if raw_df is not None else 0} rows). "
                f"Need at least {time_step + 10}."
            )

        # 3. Build features
        featured_df = build_energy_features(raw_df)

        # Filter to used_features (same columns model was trained on)
        available = [c for c in used_features if c in featured_df.columns]
        if "Close" not in available:
            return self._error_response(zone, hours, "Close column missing from features")

        featured_df = featured_df[available]

        # 4. Scale and create the last sequence
        from sklearn.preprocessing import RobustScaler

        if scaler is not None:
            scaled = scaler.transform(featured_df.values)
        else:
            temp_scaler = RobustScaler()
            scaled = temp_scaler.fit_transform(featured_df.values)

        if len(scaled) < time_step:
            return self._error_response(zone, hours, f"Not enough scaled data: {len(scaled)} < {time_step}")

        last_seq = scaled[-time_step:].reshape(1, time_step, -1)

        # Current price (last known actual)
        current_price = float(featured_df["Close"].iloc[-1])
        last_timestamp = featured_df.index[-1]

        # 5. Run single-step ensemble (for per-model breakdown + confidence)
        next_price, per_model = ensemble_predict(
            models, last_seq, scaler, zone,
            cv_weights=cv_weights,
            price_range=price_range,
            current_price=current_price,
        )

        confidence = calculate_confidence(per_model, zone) if per_model else 50.0

        # 6. Multi-step forecast
        forecast_prices = multi_step_forecast(
            models, last_seq, scaler,
            steps=hours, zone=zone,
            cv_weights=cv_weights,
            price_range=price_range,
        )

        # 7. Build response
        forecasts = []
        for i, price in enumerate(forecast_prices):
            ts = pd.Timestamp(last_timestamp) + pd.Timedelta(hours=i + 1)
            forecasts.append({
                "hour": i + 1,
                "timestamp_utc": ts.isoformat(),
                "price_eur": round(price, 2),
            })

        return {
            "zone": zone,
            "hours": hours,
            "forecasts": forecasts,
            "current_price": round(current_price, 2),
            "confidence": round(confidence, 1),
            "models_used": len(per_model),
            "models_total": len(models),
            "per_model": {k: round(v, 2) for k, v in per_model.items()},
            "price_range": list(price_range) if price_range else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Status check ─────────────────────────────────────────────────

    def status(self) -> dict:
        """Return loaded model status for /api/health."""
        info = {}
        for zone, entry in self._cache.items():
            info[zone] = {
                "models_loaded": len(entry["models"]),
                "model_names": sorted(entry["models"].keys()),
                "loaded_at": entry["loaded_at"],
            }
        return info

    # ── Error helper ─────────────────────────────────────────────────

    @staticmethod
    def _error_response(zone: str, hours: int, message: str) -> dict:
        return {
            "zone": zone,
            "hours": hours,
            "forecasts": [],
            "current_price": None,
            "confidence": 0,
            "models_used": 0,
            "models_total": 0,
            "per_model": {},
            "price_range": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error": message,
        }
