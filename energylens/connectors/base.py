"""
EnergyLens — Base DataProvider class.
Adapted from MarketLens FMPDataProvider pattern.

Every connector inherits from this and gets:
- Async HTTP with retry + exponential backoff
- Smart caching (market-hours vs off-hours cadence)
- Health status reporting (LIVE / STALE / DOWN)
- Raw response archiving before parsing
- Structured logging
"""

import asyncio
import aiohttp
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import (
    RAW_ARCHIVE_DIR,
    CACHE_TTL_MARKET_OPEN,
    CACHE_TTL_MARKET_CLOSED,
    LOG_LEVEL,
)


logger = logging.getLogger("energylens.connectors")
logger.setLevel(getattr(logging, LOG_LEVEL))


class HealthStatus:
    LIVE = "LIVE"
    STALE = "STALE"
    DOWN = "DOWN"


class CacheEntry:
    """A cached data item with timestamp."""
    def __init__(self, data: Any, timestamp: float):
        self.data = data
        self.timestamp = timestamp

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class BaseDataProvider(ABC):
    """
    Abstract base class for all data source connectors.

    Subclasses implement:
        - source_name: str property
        - base_url: str property
        - fetch_raw(endpoint, params) -> raw response data
        - parse_response(raw_data) -> structured data
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        cache_ttl_open: int = CACHE_TTL_MARKET_OPEN,
        cache_ttl_closed: int = CACHE_TTL_MARKET_CLOSED,
    ):
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.cache_ttl_open = cache_ttl_open
        self.cache_ttl_closed = cache_ttl_closed
        self._cache: dict[str, CacheEntry] = {}
        self._health = HealthStatus.DOWN
        self._last_success: Optional[float] = None
        self._consecutive_failures: int = 0

        # Ensure raw archive directory exists for this source
        self.archive_dir = RAW_ARCHIVE_DIR / self.source_name
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source, e.g. 'nordpool', 'entsoe'."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the API."""
        ...

    @abstractmethod
    def parse_response(self, raw_data: Any, endpoint: str) -> Any:
        """Parse raw API response into structured data."""
        ...

    # ─── Core Fetch with Retry ───

    async def _http_get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ) -> Any:
        """Execute HTTP GET with retry and exponential backoff."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, params=params, headers=headers) as resp:
                        if resp.status == 200:
                            content_type = resp.headers.get("Content-Type", "")
                            if "json" in content_type:
                                return await resp.json()
                            elif "xml" in content_type or "text" in content_type:
                                return await resp.text()
                            else:
                                return await resp.text()

                        elif resp.status == 429:
                            # Rate limited — back off longer
                            wait = 2 ** attempt * 2
                            logger.warning(
                                f"[{self.source_name}] Rate limited (429). "
                                f"Waiting {wait}s (attempt {attempt}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait)
                            continue

                        else:
                            last_error = f"HTTP {resp.status}: {await resp.text()}"
                            logger.warning(
                                f"[{self.source_name}] {last_error} "
                                f"(attempt {attempt}/{self.max_retries})"
                            )

            except asyncio.TimeoutError:
                last_error = "Request timed out"
                logger.warning(
                    f"[{self.source_name}] Timeout "
                    f"(attempt {attempt}/{self.max_retries})"
                )
            except aiohttp.ClientError as e:
                last_error = str(e)
                logger.warning(
                    f"[{self.source_name}] Connection error: {e} "
                    f"(attempt {attempt}/{self.max_retries})"
                )

            # Exponential backoff
            if attempt < self.max_retries:
                wait = 2 ** attempt
                await asyncio.sleep(wait)

        # All retries exhausted
        self._consecutive_failures += 1
        self._health = HealthStatus.DOWN
        raise ConnectionError(
            f"[{self.source_name}] Failed after {self.max_retries} attempts: {last_error}"
        )

    # ─── Caching ───

    def _get_cache_ttl(self) -> int:
        """Return appropriate cache TTL based on whether markets are active."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()
        # Nordic markets roughly active 06:00–18:00 UTC, Mon–Fri
        if weekday < 5 and 6 <= hour <= 18:
            return self.cache_ttl_open
        return self.cache_ttl_closed

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Return cached data if still fresh, else None."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        if entry.age_seconds > self._get_cache_ttl():
            return None
        return entry.data

    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with current timestamp."""
        self._cache[cache_key] = CacheEntry(data=data, timestamp=time.time())

    # ─── Raw Archive ───

    def _archive_raw(self, endpoint: str, raw_data: Any) -> Path:
        """
        Write raw API response to immutable archive.
        File path: data/raw/{source}/{date}/{timestamp}_{endpoint}.json
        """
        now = datetime.now(timezone.utc)
        day_dir = self.archive_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        safe_endpoint = endpoint.replace("/", "_").replace("?", "_")[:80]
        filename = f"{now.strftime('%H%M%S')}_{safe_endpoint}.json"
        filepath = day_dir / filename

        archive_record = {
            "source": self.source_name,
            "endpoint": endpoint,
            "arrival_utc": now.isoformat(),
            "data": raw_data,
        }

        with open(filepath, "w") as f:
            json.dump(archive_record, f, default=str, ensure_ascii=False)

        return filepath

    # ─── Public Interface ───

    async def fetch(self, endpoint: str, params: Optional[dict] = None, use_cache: bool = True) -> Any:
        """
        Fetch data from the source with caching, retry, and archiving.

        1. Check cache → return if fresh
        2. HTTP GET with retry
        3. Archive raw response
        4. Parse into structured data
        5. Update cache
        6. Return structured data
        """
        cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"

        # 1. Check cache
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                logger.debug(f"[{self.source_name}] Cache hit: {endpoint}")
                return cached

        # 2. Fetch
        url = f"{self.base_url}/{endpoint}" if not endpoint.startswith("http") else endpoint
        raw_data = await self._http_get(url, params=params)

        # 3. Archive
        try:
            self._archive_raw(endpoint, raw_data)
        except Exception as e:
            logger.error(f"[{self.source_name}] Archive failed: {e}")

        # 4. Parse
        parsed = self.parse_response(raw_data, endpoint)

        # 5. Update cache + health
        self._update_cache(cache_key, parsed)
        self._last_success = time.time()
        self._consecutive_failures = 0
        self._health = HealthStatus.LIVE

        logger.info(f"[{self.source_name}] Fetched: {endpoint}")
        return parsed

    def get_last_known_good(self, endpoint: str, params: Optional[dict] = None) -> Optional[Any]:
        """Return cached data regardless of age — fallback mode."""
        cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        entry = self._cache.get(cache_key)
        if entry:
            logger.warning(
                f"[{self.source_name}] Serving stale data "
                f"(age: {entry.age_seconds:.0f}s): {endpoint}"
            )
            self._health = HealthStatus.STALE
            return entry.data
        return None

    @property
    def health(self) -> dict:
        """Report connector health status."""
        return {
            "source": self.source_name,
            "status": self._health,
            "last_success": (
                datetime.fromtimestamp(self._last_success, tz=timezone.utc).isoformat()
                if self._last_success
                else None
            ),
            "consecutive_failures": self._consecutive_failures,
            "cache_entries": len(self._cache),
        }
