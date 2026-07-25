"""
EnergyLens — Forecast service.

Bridges the FastAPI layer with the ML module.
  - Lazy-loads trained models on first request
  - Pulls the latest data from SQLite for feature building
  - Resamples 15-min DayAheadPrices to hourly (models trained on hourly)
  - Handles near-zero/negative prices (common in high-wind Nordic periods)
  - Returns structured forecasts with confidence scoring

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
from api.quality_gate import evaluate_quality_gate

logger = logging.getLogger(__name__)


class ForecastService:
    """Stateful service that caches loaded models between requests."""

    def __init__(self, db_path: str = "data/energylens.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = model_dir
        self._cache: dict[str, dict] = {}
        self._lock = Lock()

    # -- Model loading (lazy, cached) -----------------------------------------

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

    # -- Data loading ----------------------------------------------------------

    def _load_recent_data(self, zone: str, lookback_hours: int = 200) -> pd.DataFrame | None:
        """
        Load the most recent data from SQLite for feature building.
        We need enough history for the rolling windows (168h weekly lag).
        Resamples 15-min data to hourly since models were trained on hourly.
        """
        db = Path(self.db_path)
        if not db.exists():
            logger.error(f"Database not found: {db}")
            return None

        conn = sqlite3.connect(str(db))

        try:
            # Fetch 4x rows to account for 15-min intervals
            prices_df = pd.read_sql_query(
                """
                SELECT valid_time AS HourUTC,
                       price_eur_mwh AS SpotPriceEUR,
                       price_dkk_mwh AS SpotPriceDKK
                FROM spot_prices
                WHERE zone = ?
                ORDER BY valid_time DESC
                LIMIT ?
                """,
                conn,
                params=(zone, lookback_hours * 4),
                parse_dates=["HourUTC"],
            )
            prices_df = prices_df.sort_values("HourUTC").set_index("HourUTC")
            # Resample to hourly (models trained on hourly data)
            prices_df = prices_df.resample("h").mean().dropna()
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
            conn.close()
            return None

        try:
            # Check what columns the weather table has
            weather_cols_cursor = conn.execute("PRAGMA table_info(weather_forecasts)")
            weather_col_names = [r[1] for r in weather_cols_cursor]

            # Build query based on available columns
            select_cols = ["valid_time AS time"]
            for col in ["temperature_2m", "wind_speed_10m", "shortwave_radiation",
                         "temp_c", "wind_speed_ms", "solar_radiation"]:
                if col in weather_col_names:
                    select_cols.append(col)

            weather_df = pd.read_sql_query(
                f"""
                SELECT {', '.join(select_cols)}
                FROM weather_forecasts
                ORDER BY valid_time DESC
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


        # Load generation data (ENTSO-E)
        gen_hourly = pd.DataFrame()
        try:
            gen_raw = pd.read_sql_query(
                """
                SELECT valid_time AS time, generation_type, value_mw
                FROM generation
                WHERE zone = ?
                ORDER BY valid_time DESC
                LIMIT ?
                """,
                conn,
                params=(zone, lookback_hours * 10),
                parse_dates=["time"],
            )
            if len(gen_raw) > 0:
                gen_raw = gen_raw.sort_values("time")
                gen_raw["time"] = pd.to_datetime(gen_raw["time"], utc=True).dt.tz_localize(None)
                gen_pivot = gen_raw.pivot_table(
                    index="time", columns="generation_type",
                    values="value_mw", aggfunc="mean"
                )
                wind_cols = [c for c in gen_pivot.columns if "wind" in c.lower()]
                solar_cols = [c for c in gen_pivot.columns if "solar" in c.lower()]
                gen_hourly = pd.DataFrame(index=gen_pivot.index)
                if wind_cols:
                    gen_hourly["wind_generation_mw"] = gen_pivot[wind_cols].sum(axis=1)
                if solar_cols:
                    gen_hourly["solar_generation_mw"] = gen_pivot[solar_cols].sum(axis=1)
                renewable_cols = wind_cols + solar_cols
                if renewable_cols:
                    gen_hourly["renewable_generation"] = gen_pivot[renewable_cols].sum(axis=1)
                gen_hourly["total_generation_mw"] = gen_pivot.sum(axis=1)
                for col in gen_pivot.columns:
                    gen_hourly[f"gen_{col}"] = gen_pivot[col]
                gen_hourly = gen_hourly.resample("h").mean()
                logger.info(f"Loaded {len(gen_hourly)} generation rows for {zone}")
        except Exception as e:
            logger.debug(f"Generation data not available: {e}")

        conn.close()


        merged = prices_df

        if not weather_hourly.empty:
            merged = merged.join(weather_hourly, how="left")

        if not gen_hourly.empty:
            merged = merged.join(gen_hourly, how="left")

        merged = merged.sort_index().dropna(subset=["SpotPriceEUR"])
        logger.info(f"Loaded {len(merged)} hourly rows for {zone}")
        return merged

    # -- Core forecast method --------------------------------------------------

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
        use_differencing = config.get("use_differencing", False)

        # 2. Load recent data
        # We need time_step + buffer for feature building (lags up to 168h)
        raw_df = self._load_recent_data(zone, lookback_hours=max(time_step + 200, 400))
        if raw_df is None or len(raw_df) < time_step + 10:
            return self._error_response(
                zone, hours,
                f"Insufficient data ({len(raw_df) if raw_df is not None else 0} rows). "
                f"Need at least {time_step + 10}."
            )

        # IMPORTANT: Capture current price BEFORE feature transform
        # build_energy_features modifies the DataFrame in-place
        current_price = float(raw_df["SpotPriceEUR"].iloc[-1])

        # Use actual last data timestamp (not wall-clock) for forecast stepping
        last_data_ts = raw_df.index[-1]
        if hasattr(last_data_ts, 'to_pydatetime'):
            last_timestamp = last_data_ts.to_pydatetime()
        else:
            last_timestamp = pd.Timestamp(last_data_ts).to_pydatetime()
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

        logger.info(f"Current price for {zone}: EUR {current_price:.2f}")

         # ── Compute recent 24h price range for smarter catastrophic filter ──
        # Nordic intraday swings can be 10x (€170 → €10 same day).
        # Anchoring the filter only on current_price kills all models
        # during these swings.
        recent_24h = raw_df["SpotPriceEUR"].iloc[-24:] if len(raw_df) >= 24 else raw_df["SpotPriceEUR"]
        recent_price_range = (float(recent_24h.min()), float(recent_24h.max()))
        logger.info(
            f"Recent 24h range for {zone}: €{recent_price_range[0]:.2f} – €{recent_price_range[1]:.2f}"
        )

        # When price is near zero/negative (common in high-wind Nordic periods),
        # use median of price_range for ensemble filtering to avoid
        # false CATASTROPHIC exclusions
        filter_price = current_price
        if price_range and abs(current_price) < 5.0:
            filter_price = (price_range[0] + price_range[1]) / 2
            logger.info(
                f"Near-zero price detected ({current_price:.2f} EUR), "
                f"using filter_price={filter_price:.2f} for ensemble outlier check"
            )

        # 3. Build features
        featured_df = build_energy_features(raw_df)

        # Filter to used_features (same columns model was trained on)
        # Pad missing features with zeros so the scaler gets the expected shape
        available = [c for c in used_features if c in featured_df.columns]
        if "Close" not in available:
            return self._error_response(zone, hours, "Close column missing from features")

        # Create aligned DataFrame with all used_features, filling missing with 0
        aligned_df = pd.DataFrame(0.0, index=featured_df.index, columns=used_features)
        for col in available:
            aligned_df[col] = featured_df[col]
        featured_df = aligned_df

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

        # 5. Run single-step ensemble (for per-model breakdown + confidence)
        next_price, per_model = ensemble_predict(
            models, last_seq, scaler, zone,
            cv_weights=cv_weights,
            price_range=None,
            current_price=current_price,
            recent_price_range=recent_price_range,
        )

        # If models were trained with differencing, their output is a price
        # CHANGE, not an absolute price. Add current_price to reconstruct.
        if use_differencing:
            logger.info(f"Differencing mode: adjusting predictions by base price €{current_price:.2f}")
            next_price = current_price + next_price
            per_model = {k: current_price + v for k, v in per_model.items()}

        confidence = calculate_confidence(per_model, zone) if per_model else 50.0

        # 6. Multi-step forecast
        forecast_prices = multi_step_forecast(
            models, last_seq, scaler,
            steps=hours, zone=zone,
            cv_weights=cv_weights,
            price_range=None,  # disabled: avoids CATASTROPHIC on near-zero prices
            feature_names=used_features,
            last_timestamp=last_timestamp,
            current_price=current_price,
            recent_price_range=recent_price_range,
        )

        # For differencing: each forecast step is a change — accumulate
        if use_differencing:
            accumulated = []
            base = current_price
            for change in forecast_prices:
                base = base + change
                accumulated.append(base)
            forecast_prices = accumulated

        # 6b. Naive fallback when all ML models fail
        if not forecast_prices:
            logger.warning("All ML models failed — generating naive fallback forecast")
            # Use same-hour-yesterday prices as baseline, fall back to current_price
            fallback_prices = []
            for i in range(1, hours + 1):
                target_hour = (last_timestamp.hour + i) % 24
                same_hour = raw_df["SpotPriceEUR"].iloc[-48:][
                    raw_df.index[-48:].hour == target_hour
                ]
                if len(same_hour) > 0:
                    fallback_prices.append(float(same_hour.mean()))
                else:
                    fallback_prices.append(current_price)
            forecast_prices = fallback_prices
            per_model["_fallback"] = "naive_same_hour_avg"
            confidence = 25.0
            logger.info(
                f"Naive fallback: {len(forecast_prices)} steps, "
                f"range €{min(forecast_prices):.2f}–€{max(forecast_prices):.2f}"
            )    

        # 7. Quality Gate evaluation
        quality_gate = evaluate_quality_gate(
            confidence=confidence,
            per_model_preds=per_model,
            forecast_prices=forecast_prices,
            current_price=current_price,
            last_data_timestamp=last_timestamp,
            zone=zone,
        )

        # 8. Build response
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
            "quality_gate": quality_gate,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # -- Status check ----------------------------------------------------------

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

    # -- Error helper ----------------------------------------------------------

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