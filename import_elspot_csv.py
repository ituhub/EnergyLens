"""
Import Elspotprices.csv from Energi Data Service into EnergyLens SQLite DB.

Handles:
- Semicolon delimiter
- Danish decimal format (comma -> dot)
- Column mapping to existing schema:
    HourUTC       -> valid_time
    PriceArea     -> zone
    SpotPriceEUR  -> price_eur_mwh
    SpotPriceDKK  -> price_dkk_mwh
- knowledge_time set to valid_time (day-ahead publication)
- Skips duplicates on (valid_time, zone)

Usage:
    python import_elspot_csv.py Elspotprices.csv --db data/energylens.db
"""

import csv
import sqlite3
import argparse
import os


def parse_danish_decimal(value: str) -> float:
    """Convert Danish number format (comma decimal) to float."""
    return float(value.replace(",", "."))


def import_csv(csv_path: str, db_path: str):
    """Import Elspotprices CSV into SQLite."""
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)

    # Check existing row count
    existing = conn.execute("SELECT COUNT(*) FROM spot_prices").fetchone()[0]
    print(f"Existing rows in spot_prices: {existing}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        batch = []
        for row in reader:
            valid_time = row["HourUTC"]
            zone = row["PriceArea"]
            price_eur = parse_danish_decimal(row["SpotPriceEUR"])
            price_dkk = parse_danish_decimal(row["SpotPriceDKK"])

            batch.append((
                valid_time,       # valid_time
                valid_time,       # knowledge_time (= valid_time for spot)
                zone,             # zone
                price_eur,        # price_eur_mwh
                price_dkk,        # price_dkk_mwh
                "energidataservice_csv",  # source
                1,                # quality_passed
                None,             # quality_report
            ))

    # INSERT OR IGNORE skips rows where (valid_time, zone) already exists
    # Using a unique index check via conflict on the natural key
    cursor = conn.executemany(
        """
        INSERT INTO spot_prices
            (valid_time, knowledge_time, zone, price_eur_mwh, price_dkk_mwh,
             source, quality_passed, quality_report)
        SELECT ?, ?, ?, ?, ?, ?, ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM spot_prices
            WHERE valid_time = ? AND zone = ?
        )
        """,
        # Append valid_time and zone again for the WHERE clause
        [row + (row[0], row[2]) for row in batch],
    )
    inserted = cursor.rowcount
    skipped = len(batch) - inserted

    conn.commit()

    # Verify
    total = conn.execute("SELECT COUNT(*) FROM spot_prices").fetchone()[0]
    date_range = conn.execute(
        "SELECT MIN(valid_time), MAX(valid_time) FROM spot_prices"
    ).fetchone()
    zone_counts = conn.execute(
        "SELECT zone, COUNT(*) FROM spot_prices GROUP BY zone"
    ).fetchall()

    print(f"\nImport complete:")
    print(f"  Rows in CSV:      {len(batch)}")
    print(f"  Inserted:         {inserted}")
    print(f"  Skipped (dupes):  {skipped}")
    print(f"  Total in DB:      {total}")
    print(f"  Date range:       {date_range[0]} -> {date_range[1]}")
    print(f"  Per zone:         {dict(zone_counts)}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import Elspotprices CSV into EnergyLens SQLite"
    )
    parser.add_argument("csv_file", help="Path to Elspotprices.csv")
    parser.add_argument(
        "--db",
        default="data/energylens.db",
        help="Path to SQLite database (default: data/energylens.db)",
    )
    args = parser.parse_args()

    print(f"Importing {args.csv_file} -> {args.db}")
    import_csv(args.csv_file, args.db)
