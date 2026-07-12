"""
EnergyLens — Quick backfill script.
Fetches recent spot prices from Energi Data Service and inserts into SQLite.
Handles rate limiting with proper 180-second cooldowns.

Usage:
    python quick_backfill.py              # Default: 7 days
    python quick_backfill.py --days 30    # 30 days
"""

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from urllib.error import HTTPError

DB_PATH = "data/energylens.db"
API_URL = "https://api.energidataservice.dk/dataset/Elspotprices"


def fetch_prices(start_date: str, end_date: str, offset: int = 0, limit: int = 2000):
    """Fetch spot prices from Energi Data Service."""
    params = (
        f"?start={start_date}&end={end_date}"
        f"&limit={limit}&offset={offset}"
        f"&sort=HourUTC asc"
    )
    url = API_URL + params
    print(f"  Fetching: {start_date} to {end_date} (offset={offset})...")

    req = Request(url, headers={"User-Agent": "EnergyLens/1.0"})

    for attempt in range(5):
        try:
            with urlopen(req, timeout=30) as resp:
                if resp.status == 429:
                    wait = 180
                    print(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                data = json.loads(resp.read().decode())
                records = data.get("records", [])
                total = data.get("total", 0)
                print(f"  Got {len(records)} records (total available: {total})")
                return records, total
        except HTTPError as e:
            if e.code == 429:
                wait = 180
                print(f"  Rate limited (429). Waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                print(f"  HTTP error {e.code}: {e.reason}")
                return [], 0
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < 4:
                wait = 30
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return [], 0

    return [], 0


def insert_records(conn, records):
    """Insert price records into SQLite."""
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    cur = conn.cursor()

    for r in records:
        hour_utc = r.get("HourUTC")
        zone = r.get("PriceArea")
        price_eur = r.get("SpotPriceEUR")
        price_dkk = r.get("SpotPriceDKK")

        if not hour_utc or not zone:
            continue

        # Skip duplicates
        cur.execute(
            "SELECT COUNT(*) FROM spot_prices WHERE valid_time = ? AND zone = ?",
            (hour_utc, zone)
        )
        if cur.fetchone()[0] > 0:
            continue

        cur.execute("""
            INSERT INTO spot_prices
                (valid_time, knowledge_time, zone, price_eur_mwh, price_dkk_mwh, source, quality_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (hour_utc, now, zone, price_eur, price_dkk, "nordpool_backfill", 1))
        count += 1

    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="EnergyLens quick backfill")
    parser.add_argument("--days", type=int, default=7, help="Days to backfill")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_str = start.strftime("%Y-%m-%dT00:00")
    end_str = end.strftime("%Y-%m-%dT%H:00")

    print("=" * 60)
    print(f"EnergyLens Quick Backfill — {args.days} days")
    print(f"Period: {start_str} to {end_str}")
    print(f"Database: {DB_PATH}")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # Check current state
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM spot_prices")
    before = cur.fetchone()[0]
    cur.execute("SELECT MAX(valid_time) FROM spot_prices")
    max_time = cur.fetchone()[0]
    print(f"\nBefore: {before} records, latest: {max_time}")

    # Fetch in chunks to avoid rate limiting
    total_inserted = 0
    offset = 0
    chunk_size = 2000  # Smaller than 5000 to be gentler on the API

    while True:
        records, total = fetch_prices(start_str, end_str, offset=offset, limit=chunk_size)

        if not records:
            break

        inserted = insert_records(conn, records)
        total_inserted += inserted
        offset += len(records)
        print(f"  Inserted {inserted} new records (total new: {total_inserted})")

        if len(records) < chunk_size:
            # Got all records
            break

        if offset < total:
            # More records to fetch — wait for rate limit
            wait = 180
            print(f"\n  Waiting {wait}s for API rate limit before next batch...")
            time.sleep(wait)
        else:
            break

    # Final stats
    cur.execute("SELECT COUNT(*) FROM spot_prices")
    after = cur.fetchone()[0]
    cur.execute("SELECT MAX(valid_time) FROM spot_prices")
    new_max = cur.fetchone()[0]
    cur.execute("SELECT MIN(valid_time) FROM spot_prices WHERE valid_time > '2026-01-01'")
    newest_start = cur.fetchone()[0]

    print("\n" + "=" * 60)
    print(f"BACKFILL COMPLETE")
    print(f"Records: {before} → {after} (+{total_inserted} new)")
    print(f"Latest timestamp: {new_max}")
    if newest_start:
        print(f"Newest data from: {newest_start}")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
