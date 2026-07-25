"""
EnergyLens - Backfill ENTSO-E generation data for model retraining.

Usage:
    python backfill_generation.py                  # Default: 365 days
    python backfill_generation.py --days 180       # Custom range
    python backfill_generation.py --check          # Just check what is in the DB

Requires ENTSOE_API_KEY environment variable.
"""

import asyncio
import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from connectors import ENTSOEProvider
from core.database import Database
from config.constants import ACTIVE_ZONES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("energylens.backfill_generation")


async def backfill(days: int):
    """Backfill generation data day by day."""
    db = Database()
    db.initialize()
    entsoe = ENTSOEProvider()

    now = datetime.now(timezone.utc)
    total = 0
    errors = 0

    for zone in ACTIVE_ZONES:
        logger.info(f"Backfilling {zone} - {days} days")

        for day_offset in range(days):
            target = now - timedelta(days=day_offset)

            try:
                # Day-ahead generation forecast
                forecast_records = await entsoe.get_generation_forecast(
                    zone=zone, date=target
                )
                if forecast_records:
                    count = db.insert_generation(forecast_records, zone)
                    total += count

                # Actual generation by type
                actual_records = await entsoe.get_actual_generation(
                    zone=zone, date=target
                )
                if actual_records:
                    for r in actual_records:
                        r["is_forecast"] = False
                    count = db.insert_generation(actual_records, zone)
                    total += count

            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Failed {zone} day {day_offset} ({target.date()}): {e}")
                elif errors == 6:
                    logger.warning("Suppressing further error messages...")

            # Progress report every 30 days
            if day_offset > 0 and day_offset % 30 == 0:
                logger.info(
                    f"  {zone}: {day_offset}/{days} days done, "
                    f"{total} records total, {errors} errors"
                )

            # Rate limiting (ENTSO-E: ~400 req/min)
            await asyncio.sleep(0.3)

    logger.info(f"Backfill complete: {total} records inserted, {errors} errors")


def check_db():
    """Print current generation data summary."""
    db_path = Path("data/energylens.db")
    if not db_path.exists():
        print("Database not found")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    print("=== Generation Data Summary ===\n")

    cur.execute("SELECT COUNT(*) FROM generation")
    total = cur.fetchone()[0]
    print(f"Total records: {total}")

    if total > 0:
        cur.execute("SELECT MIN(valid_time), MAX(valid_time) FROM generation")
        row = cur.fetchone()
        print(f"Time range: {row[0]} -> {row[1]}")

        print("\nBy zone and type:")
        cur.execute("""
            SELECT zone, generation_type, COUNT(*), is_forecast
            FROM generation
            GROUP BY zone, generation_type, is_forecast
            ORDER BY zone, generation_type, is_forecast
        """)
        for row in cur.fetchall():
            kind = "forecast" if row[3] else "actual"
            print(f"  {row[0]} | {row[1]:20s} | {row[2]:6d} records ({kind})")

    print("\n=== Spot Prices ===")
    cur.execute("SELECT COUNT(*), zone, MIN(valid_time), MAX(valid_time) FROM spot_prices GROUP BY zone")
    for row in cur.fetchall():
        print(f"  {row[1]}: {row[0]} records ({row[2]} -> {row[3]})")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill ENTSO-E generation data")
    parser.add_argument("--days", type=int, default=365, help="Days to backfill (default: 365)")
    parser.add_argument("--check", action="store_true", help="Check current DB contents")
    args = parser.parse_args()

    if args.check:
        check_db()
    else:
        asyncio.run(backfill(args.days))


if __name__ == "__main__":
    main()
