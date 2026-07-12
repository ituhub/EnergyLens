"""
EnergyLens — ENTSO-E Transparency Platform connector.

Fetches generation forecasts, actual generation per type, cross-border
flows, and installed capacity. Requires free API key (register at
transparency.entsoe.eu).

Data source: https://web-api.tp.entsoe.eu/
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from connectors.base import BaseDataProvider
from config.constants import ENTSOE_BASE_URL, BIDDING_ZONES
from config.settings import ENTSOE_API_KEY

logger = logging.getLogger("energylens.connectors.entsoe")

# ENTSO-E document type codes
DOC_TYPES = {
    "generation_forecast": "A71",    # Wind/solar generation forecast
    "actual_generation": "A75",       # Actual generation per type
    "load_forecast": "A65",           # Day-ahead load forecast
    "actual_load": "A67",             # Actual total load
    "cross_border_flows": "A11",      # Cross-border physical flows
}

# Generation type codes (PSR types)
PSR_TYPES = {
    "B16": "solar",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B01": "biomass",
    "B02": "fossil_brown_coal",
    "B04": "fossil_gas",
    "B05": "fossil_hard_coal",
    "B06": "fossil_oil",
    "B09": "geothermal",
    "B10": "hydro_pumped",
    "B11": "hydro_river",
    "B12": "hydro_reservoir",
    "B14": "nuclear",
    "B17": "solar_thermal",
    "B20": "other",
}


class ENTSOEProvider(BaseDataProvider):
    """
    Fetches data from ENTSO-E Transparency Platform.

    Key data for energy price forecasting:
    - Wind/solar generation forecasts (supply side → price impact)
    - Actual generation by type (model validation)
    - Cross-border flows (import/export → price convergence)
    - Load forecasts (demand side)
    """

    def __init__(self, **kwargs):
        if not ENTSOE_API_KEY:
            logger.warning(
                "ENTSOE_API_KEY not set. Register free at "
                "https://transparency.entsoe.eu/ and add to .env"
            )
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "entsoe"

    @property
    def base_url(self) -> str:
        return ENTSOE_BASE_URL

    def _format_period(self, dt: datetime) -> str:
        """Format datetime for ENTSO-E API (YYYYMMddHHmm)."""
        return dt.strftime("%Y%m%d%H%M")

    def _get_zone_code(self, zone: str) -> str:
        """Get ENTSO-E area code for a bidding zone."""
        zone_info = BIDDING_ZONES.get(zone)
        if not zone_info:
            raise ValueError(f"Unknown zone: {zone}. Available: {list(BIDDING_ZONES.keys())}")
        return zone_info["entsoe_code"]

    def parse_response(self, raw_data: Any, endpoint: str) -> list[dict]:
        """Parse ENTSO-E XML response into structured records."""
        if not isinstance(raw_data, str):
            raw_data = str(raw_data)

        records = []
        try:
            root = ET.fromstring(raw_data)
            # ENTSO-E uses namespaces
            ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

            # Try multiple namespace patterns (ENTSO-E varies by doc type)
            for ns_uri in [
                "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0",
                "urn:iec62325.351:tc57wg16:451-2:publicationdocument:3:0",
                "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3",
            ]:
                ns = {"ns": ns_uri}
                time_series = root.findall(".//ns:TimeSeries", ns)
                if time_series:
                    break

            for ts in time_series:
                # Get PSR type if present (generation type)
                psr_elem = ts.find(".//ns:psrType", ns)
                psr_code = psr_elem.text if psr_elem is not None else None
                gen_type = PSR_TYPES.get(psr_code, psr_code)

                # Get area
                area_elem = ts.find(".//ns:inBiddingZone_Domain.mRID", ns)
                if area_elem is None:
                    area_elem = ts.find(".//ns:in_Domain.mRID", ns)
                area = area_elem.text if area_elem is not None else "unknown"

                # Parse periods
                for period in ts.findall(".//ns:Period", ns):
                    start_elem = period.find(".//ns:start", ns)
                    resolution_elem = period.find(".//ns:resolution", ns)

                    if start_elem is None:
                        continue

                    start_time = datetime.fromisoformat(
                        start_elem.text.replace("Z", "+00:00")
                    )
                    resolution = resolution_elem.text if resolution_elem is not None else "PT60M"
                    step_minutes = 60 if "60M" in resolution else 15

                    for point in period.findall(".//ns:Point", ns):
                        pos = int(point.find("ns:position", ns).text)
                        qty_elem = point.find("ns:quantity", ns)
                        quantity = float(qty_elem.text) if qty_elem is not None else None

                        timestamp = start_time + timedelta(minutes=step_minutes * (pos - 1))

                        records.append({
                            "timestamp_utc": timestamp.isoformat(),
                            "area": area,
                            "generation_type": gen_type,
                            "value_mw": quantity,
                            "resolution_minutes": step_minutes,
                            "source": "entsoe",
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        })

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        except Exception as e:
            logger.error(f"ENTSO-E parse error: {e}")

        logger.info(f"Parsed {len(records)} records from ENTSO-E XML")
        return records

    async def get_generation_forecast(
        self,
        zone: str = "DK1",
        date: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch wind/solar generation forecast for a zone.
        Critical for price prediction — more wind = lower prices.
        """
        if not ENTSOE_API_KEY:
            logger.error("ENTSOE_API_KEY required. Skipping generation forecast.")
            return []

        target = date or datetime.now(timezone.utc)
        period_start = self._format_period(target.replace(hour=0, minute=0))
        period_end = self._format_period(target.replace(hour=0, minute=0) + timedelta(days=1))

        params = {
            "securityToken": ENTSOE_API_KEY,
            "documentType": DOC_TYPES["generation_forecast"],
            "processType": "A01",  # Day-ahead
            "in_Domain": self._get_zone_code(zone),
            "periodStart": period_start,
            "periodEnd": period_end,
        }

        return await self.fetch("", params=params)

    async def get_actual_generation(
        self,
        zone: str = "DK1",
        date: Optional[datetime] = None,
    ) -> list[dict]:
        """Fetch actual generation per type (for model validation)."""
        if not ENTSOE_API_KEY:
            return []

        target = date or datetime.now(timezone.utc)
        period_start = self._format_period(target.replace(hour=0, minute=0))
        period_end = self._format_period(target.replace(hour=0, minute=0) + timedelta(days=1))

        params = {
            "securityToken": ENTSOE_API_KEY,
            "documentType": DOC_TYPES["actual_generation"],
            "processType": "A16",  # Realised
            "in_Domain": self._get_zone_code(zone),
            "periodStart": period_start,
            "periodEnd": period_end,
        }

        return await self.fetch("", params=params)

    async def get_load_forecast(
        self,
        zone: str = "DK1",
        date: Optional[datetime] = None,
    ) -> list[dict]:
        """Fetch day-ahead load (demand) forecast."""
        if not ENTSOE_API_KEY:
            return []

        target = date or datetime.now(timezone.utc)
        period_start = self._format_period(target.replace(hour=0, minute=0))
        period_end = self._format_period(target.replace(hour=0, minute=0) + timedelta(days=1))

        params = {
            "securityToken": ENTSOE_API_KEY,
            "documentType": DOC_TYPES["load_forecast"],
            "processType": "A01",
            "outBiddingZone_Domain": self._get_zone_code(zone),
            "periodStart": period_start,
            "periodEnd": period_end,
        }

        return await self.fetch("", params=params)
