"""
EnergyLens — Nord Pool day-ahead spot price connector.

Fetches hourly day-ahead prices for DK1/DK2 from the Energi Data Service API,
which mirrors Nord Pool data and is freely accessible without authentication.

Data source: https://api.energidataservice.dk/dataset/Elspotprices
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base import BaseDataProvider
from config.constants import ENERGI_DATA_SERVICE_URL, ACTIVE_ZONES

logger = logging.getLogger("energylens.connectors.nordpool")


class NordPoolProvider(BaseDataProvider):
    """
    Fetches Nord Pool day-ahead electricity spot prices via Energi Data Service.

    Why Energi Data Service instead of Nord Pool directly:
    - Free, no API key required
    - Clean REST/JSON API (Nord Pool's own API is less documented)
    - Covers all Nordic zones
    - Includes both DKK and EUR prices
    - Historical data available back to 2017+
    """

    @property
    def source_name(self) -> str:
        return "nordpool"

    @property
    def base_url(self) -> str:
        return ENERGI_DATA_SERVICE_URL

    def parse_response(self, raw_data: Any, endpoint: str) -> list[dict]:
        """Parse Energi Data Service JSON into standardized price records."""
        if isinstance(raw_data, str):
            import json
            raw_data = json.loads(raw_data)

        records = raw_data.get("records", [])
        parsed = []

        for record in records:
            try:
                parsed.append({
                    "timestamp_utc": record.get("HourUTC"),
                    "timestamp_dk": record.get("HourDK"),
                    "zone": record.get("PriceArea"),
                    "price_eur_mwh": record.get("SpotPriceEUR"),
                    "price_dkk_mwh": record.get("SpotPriceDKK"),
                    "source": "nordpool_via_energidataservice",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                })
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping malformed record: {e}")
                continue

        logger.info(f"Parsed {len(parsed)} price records from {len(records)} raw records")
        return parsed

    async def get_day_ahead_prices(
        self,
        zones: Optional[list[str]] = None,
        date: Optional[datetime] = None,
        days_back: int = 1,
    ) -> list[dict]:
        """
        Fetch day-ahead spot prices.

        Args:
            zones: Bidding zones to fetch (default: ACTIVE_ZONES = ["DK1", "DK2"])
            date: Target date (default: today)
            days_back: How many days of history to include

        Returns:
            List of price records with zone, timestamp, EUR and DKK prices
        """
        zones = zones or ACTIVE_ZONES
        zone_filter = ",".join(f'"{z}"' for z in zones)

        params = {
            "filter": f'{{"PriceArea":[{zone_filter}]}}',
            "sort": "HourUTC DESC",
            "limit": 500,
        }

        return await self.fetch("dataset/Elspotprices", params=params)

    async def get_historical_prices(
        self,
        zones: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 365,
    ) -> list[dict]:
        """
        Fetch historical spot prices for model training.

        Args:
            zones: Bidding zones
            start_date: Start date string "YYYY-MM-DD" (default: {days} ago)
            end_date: End date string "YYYY-MM-DD" (default: today)
            days: Days of history if start_date not specified

        Returns:
            List of hourly price records
        """
        zones = zones or ACTIVE_ZONES
        zone_filter = ",".join(f'"{z}"' for z in zones)

        all_records = []
        offset = 0
        limit = 5000  # API max per request

        while True:
            params = {
                "filter": f'{{"PriceArea":[{zone_filter}]}}',
                "sort": "HourUTC DESC",
                "limit": limit,
                "offset": offset,
            }

            # Bypass cache for historical bulk loads
            batch = await self.fetch(
                "dataset/Elspotprices",
                params=params,
                use_cache=False,
            )

            if not batch:
                break

            all_records.extend(batch)
            logger.info(f"Historical fetch: {len(all_records)} records so far (offset={offset})")

            if len(batch) < limit:
                break

            offset += limit

        logger.info(
            f"Historical load complete: {len(all_records)} records for zones {zones}"
        )
        return all_records

    async def get_latest_price(self, zone: str = "DK1") -> Optional[dict]:
        """Get the most recent available price for a single zone."""
        params = {
            "filter": f'{{"PriceArea":["{zone}"]}}',
            "sort": "HourUTC DESC",
            "limit": 1,
        }

        records = await self.fetch("dataset/Elspotprices", params=params)
        return records[0] if records else None