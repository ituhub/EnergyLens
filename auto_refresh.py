"""
EnergyLens — Auto-refresh spot prices from Energi Data Service.

Pulls the latest spot prices at regular intervals (default: every 6 hours)
and inserts them into SQLite. Small enough requests to avoid 429 rate limits.

Logs clearly show:
  - What data was fetched and from when
  - How many new rows were inserted vs skipped (duplicates)
  - Current DB freshness after each refresh

Usage:
    python auto_refresh.py                     # Run once
    python auto_refresh.py --loop              # Run every 6 hours
    python auto_refresh.py --loop --interval 2 # Run every 2 hours
    python auto_refresh.py --days 14           # Pull last 14 days (one-time catch-up)

Can be left running in a background terminal during demos.
"""

import argparse
import json
import logging
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("energylens.refresh")

DB_PATH = "data/energylens.db"
API_BASE = "https://api.energidataservice.dk/dataset/DayAheadPrices"
ZONES = ["DK1", "DK2"]


def ensure_table(conn: sqlite3.Connection):
    """Create spot_prices table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spot_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            valid_time TEXT NOT NULL,
            knowledge_time TEXT NOT NULL,
            zone TEXT NOT NULL,
            price_eur_mwh REAL,
            price_dkk_mwh REAL,
            source TEXT,
            quality_passed INTEGER DEFAULT 1,
            quality_report TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Ensure unique index for dedup
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_spot_unique
        ON spot_prices (valid_time, zone)
    """)
    conn.commit()


def get_db_status(conn: sqlite3.Connection) -> dict:
    """Get current DB freshness info."""
    row = conn.execute(
        "SELECT MIN(valid_time) as oldest, MAX(valid_time) as newest, COUNT(*) as total FROM spot_prices"
    ).fetchone()
    return {"oldest": row[0], "newest": row[1], "total": row[2]}


def fetch_prices(start: datetime, end: datetime, zones: list[str], limit: int = 2000) -> list[dict]:
    """
    Fetch spot prices from Energi Data Service JSON API.
    Uses the JSON endpoint (not CSV) to avoid encoding issues.
    """
    import urllib.parse
    zone_filter = json.dumps({"PriceArea": zones}, separators=(",", ":"))
    params = urllib.parse.urlencode({
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": zone_filter,
        "sort": "TimeUTC asc",
        "limit": limit,
    })
    url = API_BASE + "?" + params

    logger.info(f"Fetching: {start.strftime('%Y-%m-%d %H:%M')} -> {end.strftime('%Y-%m-%d %H:%M')} "
                f"zones={zones} limit={limit}")

    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited (429). Waiting 60 seconds...")
            time.sleep(60)
            # Retry once
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        else:
            logger.error(f"HTTP error {e.code}: {e.reason}")
            raise

    records = data.get("records", [])
    total_available = data.get("total", 0)
    logger.info(f"API returned {len(records)} records (total available: {total_available})")

    return records


def insert_records(conn: sqlite3.Connection, records: list[dict]) -> tuple[int, int]:
    """Insert records into SQLite, skipping duplicates. Returns (inserted, skipped)."""
    if not records:
        return 0, 0

    batch = []
    for r in records:
        batch.append((
            r["TimeUTC"],                    # valid_time
            r["TimeUTC"],                    # knowledge_time
            r["PriceArea"],                  # zone
            r.get("DayAheadPriceEUR"),       # price_eur_mwh
            r.get("DayAheadPriceDKK"),       # price_dkk_mwh
            "energidataservice_auto",        # source
            1,                               # quality_passed
        ))

    cursor = conn.executemany(
        """
        INSERT OR IGNORE INTO spot_prices
            (valid_time, knowledge_time, zone, price_eur_mwh, price_dkk_mwh, source, quality_passed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )
    inserted = cursor.rowcount
    skipped = len(batch) - inserted
    conn.commit()
    return inserted, skipped


def refresh(days: int = 3) -> dict:
    """
    Main refresh: pull the last N days of data and insert into SQLite.

    Default 3 days: enough to keep data fresh with overlap for dedup,
    small enough to avoid rate limits (~144 records per zone per day).
    """
    logger.info("=" * 60)
    logger.info("EnergyLens Auto-Refresh starting")
    logger.info("=" * 60)

    # Ensure DB and table exist
    db = Path(DB_PATH)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    ensure_table(conn)

    # Log current state
    before = get_db_status(conn)
    logger.info(f"DB before: {before['total']} rows, newest={before['newest']}")

    # Calculate time range
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    # Fetch from API
    try:
        records = fetch_prices(start, now, ZONES, limit=days * 192 + 100)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        conn.close()
        return {"status": "error", "error": str(e)}

    if not records:
        logger.warning("API returned 0 records — check date range or API status")
        conn.close()
        return {"status": "no_data"}

    # Log what we got
    hour_utcs = [r["TimeUTC"] for r in records]
    zones_seen = set(r["PriceArea"] for r in records)
    logger.info(f"Received {len(records)} records for zones {zones_seen}")
    logger.info(f"Data spans: {min(hour_utcs)} -> {max(hour_utcs)}")

    # Insert
    inserted, skipped = insert_records(conn, records)
    logger.info(f"Inserted: {inserted} new rows | Skipped: {skipped} duplicates")

    # Log updated state
    after = get_db_status(conn)
    logger.info(f"DB after: {after['total']} rows, newest={after['newest']}")

    # Check freshness
    try:
        newest_dt = datetime.fromisoformat(
            after["newest"].replace("T", " ").split("+")[0]
        )
        age_hours = (datetime.utcnow() - newest_dt).total_seconds() / 3600
        if age_hours < 48:
            logger.info(f"DATA STATUS: FRESH (newest record is {age_hours:.1f}h old)")
        else:
            logger.warning(f"DATA STATUS: STALE (newest record is {age_hours:.0f}h old)")
    except Exception:
        pass

    conn.close()
    logger.info("Refresh complete")
    logger.info("=" * 60)

    return {
        "status": "ok",
        "fetched": len(records),
        "inserted": inserted,
        "skipped": skipped,
        "db_total": after["total"],
        "newest": after["newest"],
    }


def run_loop(interval_hours: int = 6, days: int = 3):
    """Run refresh on a loop. Meant for background terminal during demos."""
    logger.info(f"Starting auto-refresh loop: every {interval_hours}h, pulling last {days} days")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    while True:
        try:
            result = refresh(days=days)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Refresh failed: {e}")

        next_run = datetime.now() + timedelta(hours=interval_hours)
        logger.info(f"Next refresh at {next_run.strftime('%H:%M:%S')} "
                    f"(sleeping {interval_hours}h)")
        logger.info("")

        try:
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EnergyLens Auto-Refresh — pull fresh spot prices from Energi Data Service"
    )
    parser.add_argument(
        "--days", type=int, default=3,
        help="Days of history to fetch (default: 3)"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run continuously on an interval"
    )
    parser.add_argument(
        "--interval", type=int, default=6,
        help="Hours between refreshes when using --loop (default: 6)"
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})"
    )
    args = parser.parse_args()

    DB_PATH = args.db

    if args.loop:
        run_loop(interval_hours=args.interval, days=args.days)
    else:
        result = refresh(days=args.days)
        print(f"\nResult: {json.dumps(result, indent=2)}")
