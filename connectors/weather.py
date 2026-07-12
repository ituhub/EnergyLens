"""
EnergyLens — Weather forecast connector via Open-Meteo.

Fetches wind speed, solar irradiance, and temperature forecasts for
Danish weather reference points. These are critical features for
energy price prediction (wind → generation → price impact).

Data source: https://open-meteo.com/en/docs
Free tier: No API key required, 10,000 requests/day
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base import BaseDataProvider
from config.constants import OPEN_METEO_URL, WEATHER_LOCATIONS

logger = logging.getLogger("energylens.connectors.weather")


class WeatherProvider(BaseDataProvider):
    """
    Fetches weather forecasts from Open-Meteo API.

    Key variables for energy price forecasting:
    - wind_speed_10m / wind_speed_100m: Wind generation correlation
    - direct_radiation / diffuse_radiation: Solar generation
    - temperature_2m: Heating/cooling demand
    - cloud_cover: Solar generation suppression
    """

    def __init__(self, **kwargs):
        # Weather data changes slowly — cache for 30 minutes
        kwargs.setdefault("cache_ttl_open", 1800)
        kwargs.setdefault("cache_ttl_closed", 3600)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "weather"

    @property
    def base_url(self) -> str:
        return OPEN_METEO_URL

    def parse_response(self, raw_data: Any, endpoint: str) -> dict:
        """Parse Open-Meteo JSON into structured weather records."""
        if isinstance(raw_data, str):
            import json
            raw_data = json.loads(raw_data)

        hourly = raw_data.get("hourly", {})
        times = hourly.get("time", [])

        records = []
        for i, t in enumerate(times):
            record = {
                "timestamp_utc": t,
                "latitude": raw_data.get("latitude"),
                "longitude": raw_data.get("longitude"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            # Extract all hourly variables at this index
            for key in hourly:
                if key != "time" and i < len(hourly[key]):
                    record[key] = hourly[key][i]
            records.append(record)

        return {
            "location": {
                "lat": raw_data.get("latitude"),
                "lon": raw_data.get("longitude"),
                "elevation": raw_data.get("elevation"),
                "timezone": raw_data.get("timezone"),
            },
            "records": records,
            "units": raw_data.get("hourly_units", {}),
        }

    async def get_forecast(
        self,
        location_key: str = "DK1_wind",
        forecast_days: int = 3,
    ) -> dict:
        """
        Fetch weather forecast for a location.

        Args:
            location_key: Key from WEATHER_LOCATIONS (e.g. "DK1_wind")
            forecast_days: Number of days ahead (max 16)

        Returns:
            Dict with location metadata, hourly records, and units
        """
        loc = WEATHER_LOCATIONS.get(location_key)
        if not loc:
            raise ValueError(f"Unknown location: {location_key}. Available: {list(WEATHER_LOCATIONS.keys())}")

        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "cloud_cover",
                "direct_radiation",
                "diffuse_radiation",
                "precipitation",
                "pressure_msl",
            ]),
            "forecast_days": min(forecast_days, 16),
            "timezone": "UTC",
        }

        result = await self.fetch("forecast", params=params)
        # Tag with location key for downstream use
        result["location_key"] = location_key
        result["location_name"] = loc["name"]
        return result

    async def get_all_danish_forecasts(self, forecast_days: int = 3) -> dict[str, dict]:
        """
        Fetch forecasts for all Danish weather reference points.

        Returns:
            Dict mapping location_key → forecast data
        """
        results = {}
        for key in WEATHER_LOCATIONS:
            try:
                results[key] = await self.get_forecast(key, forecast_days)
                logger.info(f"Weather forecast fetched: {key}")
            except Exception as e:
                logger.error(f"Weather fetch failed for {key}: {e}")
                # Try fallback
                fallback = self.get_last_known_good("forecast", {"location": key})
                if fallback:
                    results[key] = fallback

        return results

    async def get_historical_weather(
        self,
        location_key: str = "DK1_wind",
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
    ) -> dict:
        """
        Fetch historical weather data for model training.
        Uses Open-Meteo Historical API (free, ERA5 reanalysis data).
        """
        loc = WEATHER_LOCATIONS.get(location_key)
        if not loc:
            raise ValueError(f"Unknown location: {location_key}")

        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "hourly": ",".join([
                "temperature_2m",
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_10m",
                "cloud_cover",
                "direct_radiation",
                "diffuse_radiation",
                "precipitation",
            ]),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }

        # Historical endpoint is different
        url = "https://archive-api.open-meteo.com/v1/archive"
        raw = await self._http_get(url, params=params)
        self._archive_raw(f"historical_{location_key}", raw)
        parsed = self.parse_response(raw, "historical")
        parsed["location_key"] = location_key
        return parsed
