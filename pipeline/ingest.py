"""
EnergyLens — Ingestion orchestrator.

Coordinates data fetching from all sources, quality validation,
and storage. Can be run as a one-shot script or scheduled via cron.

Usage:
    # One-shot fetch (latest data)
    python -m pipeline.ingest

    # Historical backfill
    python -m pipeline.ingest --backfill --days 365
"""

import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors import NordPoolProvider, WeatherProvider, ENTSOEProvider
from core.database import Database
from core.quality_gate import QualityGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("energylens.pipeline.ingest")


class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline:
    1. Fetch from all data sources
    2. Validate through quality gates
    3. Store in bitemporal database
    4. Quarantine failed records
    """

    def __init__(self):
        self.db = Database()
        self.quality_gate = QualityGate()
        self.nordpool = NordPoolProvider()
        self.weather = WeatherProvider()
        self.entsoe = ENTSOEProvider()

        # Stats tracking
        self.stats = {
            "fetched": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
        }

    async def run_latest(self):
        """Fetch latest data from all sources."""
        logger.info("=" * 60)
        logger.info("EnergyLens Ingestion Pipeline — Latest Data")
        logger.info("=" * 60)

        # Initialize database
        self.db.initialize()

        # Run all fetches concurrently
        results = await asyncio.gather(
            self._ingest_spot_prices(),
            self._ingest_weather(),
            self._ingest_generation(),
            return_exceptions=True,
        )

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source = ["spot_prices", "weather", "generation"][i]
                logger.error(f"Ingestion failed for {source}: {result}")

        self._log_summary()

    async def run_backfill(self, days: int = 365):
        """Backfill historical data for model training."""
        logger.info("=" * 60)
        logger.info(f"EnergyLens Ingestion Pipeline — Backfill ({days} days)")
        logger.info("=" * 60)

        self.db.initialize()

        # Historical spot prices
        await self._backfill_spot_prices(days)

        # Historical weather
        await self._backfill_weather(days)

        # Historical generation (ENTSO-E)
        await self._backfill_generation(days)

        self._log_summary()

    # ─── Spot Prices ───

    async def _ingest_spot_prices(self) -> int:
        """Fetch and store latest spot prices."""
        logger.info("Fetching spot prices...")

        try:
            records = await self.nordpool.get_day_ahead_prices(days_back=2)
        except Exception as e:
            logger.error(f"Nord Pool fetch failed: {e}")
            # Try fallback
            records = self.nordpool.get_last_known_good(
                "dataset/Elspotprices", {}
            )
            if not records:
                return 0

        self.stats["fetched"] += len(records)

        # Validate each record
        validated = []
        for record in records:
            report = self.quality_gate.validate(record, data_type="spot_price")
            record["quality_passed"] = report.passed
            record["quality_report"] = report.summary

            if report.passed:
                validated.append(record)
                self.stats["passed"] += 1
                if report.has_warnings:
                    self.stats["warnings"] += 1
            else:
                self.stats["failed"] += 1
                self.db.quarantine_record(record, report.summary)

        # Store validated records
        count = self.db.insert_spot_prices(validated)
        logger.info(f"Spot prices: {count} inserted, {self.stats['failed']} quarantined")
        return count

    async def _backfill_spot_prices(self, days: int) -> int:
        """Backfill historical spot prices."""
        logger.info(f"Backfilling {days} days of spot prices...")

        records = await self.nordpool.get_historical_prices(days=days)
        self.stats["fetched"] += len(records)

        # For backfill, skip freshness check (data is intentionally old)
        validated = []
        for record in records:
            report = self.quality_gate.validate(record, data_type="spot_price")
            # Override freshness gate for historical data
            record["quality_passed"] = all(
                g.gate == "freshness" or g.result.value != "fail"
                for g in report.gates
            )

            if record["quality_passed"]:
                validated.append(record)
                self.stats["passed"] += 1
            else:
                self.stats["failed"] += 1

        count = self.db.insert_spot_prices(validated)
        logger.info(f"Backfill: {count} spot price records inserted")
        return count

    # ─── Weather ───

    async def _ingest_weather(self) -> int:
        """Fetch and store weather forecasts for all Danish locations."""
        logger.info("Fetching weather forecasts...")
        total = 0

        try:
            forecasts = await self.weather.get_all_danish_forecasts(forecast_days=3)
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
            return 0

        for location_key, forecast_data in forecasts.items():
            records = forecast_data.get("records", [])
            self.stats["fetched"] += len(records)

            # Validate
            validated = []
            for record in records:
                report = self.quality_gate.validate(record, data_type="weather")
                if report.passed:
                    validated.append(record)
                    self.stats["passed"] += 1
                else:
                    self.stats["failed"] += 1

            count = self.db.insert_weather(validated, location_key)
            total += count

        logger.info(f"Weather: {total} records inserted across all locations")
        return total

    async def _backfill_weather(self, days: int) -> int:
        """Backfill historical weather data."""
        logger.info(f"Backfilling {days} days of weather data...")
        total = 0

        from config.constants import WEATHER_LOCATIONS

        for location_key in WEATHER_LOCATIONS:
            try:
                from datetime import timedelta
                start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
                data = await self.weather.get_historical_weather(
                    location_key=location_key,
                    start_date=start,
                )
                records = data.get("records", [])
                self.stats["fetched"] += len(records)
                count = self.db.insert_weather(records, location_key)
                total += count
                logger.info(f"Weather backfill: {count} records for {location_key}")
            except Exception as e:
                logger.error(f"Weather backfill failed for {location_key}: {e}")

        return total

    # ─── Generation ───

    async def _ingest_generation(self) -> int:
        """Fetch generation forecasts from ENTSO-E."""
        logger.info("Fetching generation data...")

        from config.constants import ACTIVE_ZONES

        total = 0
        for zone in ACTIVE_ZONES:
            try:
                records = await self.entsoe.get_generation_forecast(zone=zone)
                self.stats["fetched"] += len(records)
                self.stats["passed"] += len(records)
                if records:
                    count = self.db.insert_generation(records, zone)
                    total += count
                logger.info(f"Generation: {len(records)} fetched, {count if records else 0} inserted for {zone}")
            except Exception as e:
                logger.warning(f"ENTSO-E fetch failed for {zone}: {e}")

        return total

    async def _backfill_generation(self, days: int) -> int:
        """Backfill historical generation data from ENTSO-E."""
        logger.info(f"Backfilling {days} days of generation data...")
        from config.constants import ACTIVE_ZONES
        from datetime import timedelta
        total = 0
        now = datetime.now(timezone.utc)
        for zone in ACTIVE_ZONES:
            for day_offset in range(days):
                target = now - timedelta(days=day_offset)
                try:
                    records = await self.entsoe.get_generation_forecast(zone=zone, date=target)
                    if records:
                        total += self.db.insert_generation(records, zone)
                    actual = await self.entsoe.get_actual_generation(zone=zone, date=target)
                    if actual:
                        for r in actual: r["is_forecast"] = False
                        total += self.db.insert_generation(actual, zone)
                    if day_offset % 30 == 0:
                        logger.info(f"  Gen backfill: {zone} day {day_offset}/{days}, total={total}")
                except Exception as e:
                    logger.warning(f"Gen backfill failed {zone} day {day_offset}: {e}")
                await asyncio.sleep(0.3)
        logger.info(f"Generation backfill complete: {total} records")
        return total

    # ─── Health & Summary ───

    def get_health(self) -> dict:
        """Get health status of all connectors."""
        return {
            "nordpool": self.nordpool.health,
            "weather": self.weather.health,
            "entsoe": self.entsoe.health,
            "database": self.db.get_record_counts(),
            "quality_gate": self.quality_gate.get_stats(),
        }

    def _log_summary(self):
        """Log ingestion summary."""
        logger.info("=" * 60)
        logger.info("Ingestion Summary")
        logger.info(f"  Fetched:     {self.stats['fetched']}")
        logger.info(f"  Passed:      {self.stats['passed']}")
        logger.info(f"  Warnings:    {self.stats['warnings']}")
        logger.info(f"  Failed:      {self.stats['failed']}")
        logger.info(f"  DB records:  {self.db.get_record_counts()}")
        logger.info("=" * 60)


# ─── CLI Entry Point ───

async def main():
    parser = argparse.ArgumentParser(description="EnergyLens Ingestion Pipeline")
    parser.add_argument("--backfill", action="store_true", help="Run historical backfill")
    parser.add_argument("--days", type=int, default=365, help="Days to backfill (default: 365)")
    args = parser.parse_args()

    pipeline = IngestionPipeline()

    if args.backfill:
        await pipeline.run_backfill(days=args.days)
    else:
        await pipeline.run_latest()


if __name__ == "__main__":
    asyncio.run(main())
