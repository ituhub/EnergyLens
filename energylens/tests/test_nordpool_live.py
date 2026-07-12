"""
Quick test: Verify Nord Pool connector fetches real spot prices.
Run from project root: python tests/test_nordpool_live.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.nordpool import NordPoolProvider
from core.quality_gate import QualityGate


async def test_live_fetch():
    print("=" * 60)
    print("EnergyLens — Nord Pool Live Fetch Test")
    print("=" * 60)

    provider = NordPoolProvider()
    gate = QualityGate()

    # 1. Fetch latest prices
    print("\n1. Fetching day-ahead prices (DK1, DK2, last 2 days)...")
    records = await provider.get_day_ahead_prices(days_back=2)
    print(f"   Fetched {len(records)} records")

    if not records:
        print("   ERROR: No records returned!")
        return

    # 2. Show a sample record
    print(f"\n2. Sample record:")
    sample = records[0]
    for key, value in sample.items():
        print(f"   {key}: {value}")

    # 3. Run quality gate on all records
    print(f"\n3. Quality Gate validation:")
    passed = 0
    failed = 0
    for record in records:
        report = gate.validate(record, data_type="spot_price")
        if report.passed:
            passed += 1
        else:
            failed += 1
            print(f"   FAILED: {report.failed_gates} for {record.get('timestamp_utc')}")

    print(f"   Passed: {passed}/{len(records)}")
    print(f"   Failed: {failed}/{len(records)}")

    # 4. Show price range
    prices = [r["price_eur_mwh"] for r in records if r.get("price_eur_mwh") is not None]
    if prices:
        print(f"\n4. Price range (EUR/MWh):")
        print(f"   Min:  {min(prices):.2f}")
        print(f"   Max:  {max(prices):.2f}")
        print(f"   Mean: {sum(prices)/len(prices):.2f}")

    # 5. Show per-zone breakdown
    print(f"\n5. Per-zone breakdown:")
    zones = set(r.get("zone") for r in records)
    for zone in sorted(zones):
        zone_records = [r for r in records if r.get("zone") == zone]
        zone_prices = [r["price_eur_mwh"] for r in zone_records if r.get("price_eur_mwh") is not None]
        if zone_prices:
            print(f"   {zone}: {len(zone_records)} records, avg {sum(zone_prices)/len(zone_prices):.2f} EUR/MWh")

    # 6. Connector health
    print(f"\n6. Connector health:")
    health = provider.health
    for key, value in health.items():
        print(f"   {key}: {value}")

    # 7. Check raw archive
    print(f"\n7. Raw archive:")
    archive_dir = Path("data/raw/nordpool")
    if archive_dir.exists():
        files = list(archive_dir.rglob("*.json"))
        print(f"   {len(files)} files archived")
    else:
        print(f"   Archive dir not found (expected at {archive_dir})")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_live_fetch())
