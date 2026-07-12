"""
EnergyLens — Import spot prices from CSV files downloaded from Energi Data Service.

Usage:
    python import_csv.py data/Elspotprices_DK1.csv data/Elspotprices_DK2.csv
    python import_csv.py data/*.csv

Download CSVs from (no rate limits):
    DK1: https://api.energidataservice.dk/dataset/Elspotprices/download?format=csv&start=2026-06-12T00:00&end=2026-07-12T23:00&filter={"PriceArea":"DK1"}&sort=HourUTC asc
    DK2: https://api.energidataservice.dk/dataset/Elspotprices/download?format=csv&start=2026-06-12T00:00&end=2026-07-12T23:00&filter={"PriceArea":"DK2"}&sort=HourUTC asc
"""

import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = "data/energylens.db"


def import_csv(csv_path: str, conn: sqlite3.Connection) -> int:
    """Import a single CSV file into spot_prices table."""
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    skipped = 0
    cur = conn.cursor()

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        # Try to detect delimiter
        sample = f.read(2048)
        f.seek(0)
        
        # Energi Data Service uses semicolon delimiter
        if ";" in sample and sample.count(";") > sample.count(","):
            delimiter = ";"
        else:
            delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Print detected columns
        print(f"  Columns: {reader.fieldnames}")

        for row in reader:
            # Handle different column name formats
            hour_utc = row.get("HourUTC") or row.get("hourutc") or row.get("Hour UTC")
            zone = row.get("PriceArea") or row.get("pricearea") or row.get("Price Area")
            
            # Price columns - try multiple formats
            price_eur = row.get("SpotPriceEUR") or row.get("spotpriceeur") or row.get("Spot Price EUR")
            price_dkk = row.get("SpotPriceDKK") or row.get("spotpricedkk") or row.get("Spot Price DKK")

            if not hour_utc or not zone:
                continue

            # Clean and convert prices
            try:
                price_eur = float(price_eur.replace(",", ".")) if price_eur and price_eur.strip() else None
            except (ValueError, AttributeError):
                price_eur = None

            try:
                price_dkk = float(price_dkk.replace(",", ".")) if price_dkk and price_dkk.strip() else None
            except (ValueError, AttributeError):
                price_dkk = None

            # Skip duplicates
            cur.execute(
                "SELECT COUNT(*) FROM spot_prices WHERE valid_time = ? AND zone = ?",
                (hour_utc, zone)
            )
            if cur.fetchone()[0] > 0:
                skipped += 1
                continue

            cur.execute("""
                INSERT INTO spot_prices
                    (valid_time, knowledge_time, zone, price_eur_mwh, price_dkk_mwh, source, quality_passed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (hour_utc, now, zone, price_eur, price_dkk, "csv_import", 1))
            count += 1

    conn.commit()
    print(f"  Imported: {count} new records, skipped: {skipped} duplicates")
    return count


def main():
    if len(sys.argv) < 2:
        print("Usage: python import_csv.py <csv_file> [csv_file2] ...")
        print("\nDownload CSVs from Energi Data Service:")
        print("  https://www.energidataservice.dk/tso-electricity/elspotprices")
        sys.exit(1)

    csv_files = sys.argv[1:]
    conn = sqlite3.connect(DB_PATH)

    # Check before
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM spot_prices")
    before = cur.fetchone()[0]
    cur.execute("SELECT MAX(valid_time) FROM spot_prices")
    max_before = cur.fetchone()[0]
    print(f"Database before: {before} records, latest: {max_before}\n")

    total = 0
    for csv_path in csv_files:
        p = Path(csv_path)
        if not p.exists():
            print(f"File not found: {csv_path}")
            continue
        print(f"Importing {p.name}...")
        total += import_csv(str(p), conn)

    # Check after
    cur.execute("SELECT COUNT(*) FROM spot_prices")
    after = cur.fetchone()[0]
    cur.execute("SELECT MAX(valid_time) FROM spot_prices")
    max_after = cur.fetchone()[0]
    cur.execute("SELECT MIN(valid_time) FROM spot_prices")
    min_after = cur.fetchone()[0]

    print(f"\n{'='*60}")
    print(f"IMPORT COMPLETE")
    print(f"Records: {before} -> {after} (+{total} new)")
    print(f"Date range: {min_after} to {max_after}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
