"""
EnergyLens — Database module with bitemporal schema.

Implements the bitemporal data layer:
- valid_time: when the fact was true in the real world
- knowledge_time: when we learned about it

This enables point-in-time queries like:
"What did we know at 06:00 on March 5th about prices for March 6th?"

For local dev, falls back to SQLite if PostgreSQL is unavailable.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import DATABASE_URL, DATA_DIR, ENVIRONMENT

logger = logging.getLogger("energylens.core.database")

# Try PostgreSQL first, fall back to SQLite for local dev
try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    logger.info("psycopg2 not available — using SQLite for local development")


# ─── Schema Definitions ───

SCHEMA_SQL_POSTGRES = """
-- Spot prices with bitemporal timestamps
CREATE TABLE IF NOT EXISTS spot_prices (
    id              BIGSERIAL PRIMARY KEY,
    valid_time      TIMESTAMPTZ NOT NULL,     -- when this price applies
    knowledge_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- when we learned it
    zone            VARCHAR(10) NOT NULL,
    price_eur_mwh   DOUBLE PRECISION,
    price_dkk_mwh   DOUBLE PRECISION,
    source          VARCHAR(50),
    quality_passed  BOOLEAN DEFAULT TRUE,
    quality_report  JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_spot_valid_zone
    ON spot_prices (zone, valid_time DESC);

CREATE INDEX IF NOT EXISTS idx_spot_knowledge
    ON spot_prices (knowledge_time DESC);

CREATE INDEX IF NOT EXISTS idx_spot_bitemporal
    ON spot_prices (zone, valid_time, knowledge_time DESC);

-- Weather forecasts (bitemporal: forecast_for vs fetched_at)
CREATE TABLE IF NOT EXISTS weather_forecasts (
    id              BIGSERIAL PRIMARY KEY,
    valid_time      TIMESTAMPTZ NOT NULL,     -- when this forecast is for
    knowledge_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- when we got the forecast
    location_key    VARCHAR(20) NOT NULL,
    temperature_2m  DOUBLE PRECISION,
    wind_speed_10m  DOUBLE PRECISION,
    wind_speed_100m DOUBLE PRECISION,
    wind_direction  DOUBLE PRECISION,
    cloud_cover     DOUBLE PRECISION,
    direct_radiation DOUBLE PRECISION,
    diffuse_radiation DOUBLE PRECISION,
    precipitation   DOUBLE PRECISION,
    source          VARCHAR(50) DEFAULT 'open_meteo',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_valid_loc
    ON weather_forecasts (location_key, valid_time DESC);

CREATE INDEX IF NOT EXISTS idx_weather_bitemporal
    ON weather_forecasts (location_key, valid_time, knowledge_time DESC);

-- Generation data
CREATE TABLE IF NOT EXISTS generation (
    id              BIGSERIAL PRIMARY KEY,
    valid_time      TIMESTAMPTZ NOT NULL,
    knowledge_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    zone            VARCHAR(10) NOT NULL,
    generation_type VARCHAR(30),
    value_mw        DOUBLE PRECISION,
    is_forecast     BOOLEAN DEFAULT FALSE,
    source          VARCHAR(50) DEFAULT 'entsoe',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gen_valid_zone
    ON generation (zone, valid_time DESC);

-- Quality quarantine (failed records stored here)
CREATE TABLE IF NOT EXISTS quality_quarantine (
    id              BIGSERIAL PRIMARY KEY,
    record_source   VARCHAR(50),
    record_data     JSONB,
    failed_gates    TEXT[],
    quality_report  JSONB,
    quarantined_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Predictions log (for tracking accuracy)
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    prediction_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_time     TIMESTAMPTZ NOT NULL,
    zone            VARCHAR(10) NOT NULL,
    predicted_price DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    model_version   VARCHAR(50),
    features_hash   VARCHAR(64),
    actual_price    DOUBLE PRECISION,     -- filled in after the fact
    error           DOUBLE PRECISION,     -- filled in after the fact
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_target
    ON predictions (zone, target_time DESC);
"""

SCHEMA_SQL_SQLITE = """
CREATE TABLE IF NOT EXISTS spot_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    valid_time      TEXT NOT NULL,
    knowledge_time  TEXT NOT NULL,
    zone            TEXT NOT NULL,
    price_eur_mwh   REAL,
    price_dkk_mwh   REAL,
    source          TEXT,
    quality_passed  INTEGER DEFAULT 1,
    quality_report  TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_spot_valid_zone
    ON spot_prices (zone, valid_time);

CREATE TABLE IF NOT EXISTS weather_forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    valid_time      TEXT NOT NULL,
    knowledge_time  TEXT NOT NULL,
    location_key    TEXT NOT NULL,
    temperature_2m  REAL,
    wind_speed_10m  REAL,
    wind_speed_100m REAL,
    wind_direction  REAL,
    cloud_cover     REAL,
    direct_radiation REAL,
    diffuse_radiation REAL,
    precipitation   REAL,
    source          TEXT DEFAULT 'open_meteo',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS generation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    valid_time      TEXT NOT NULL,
    knowledge_time  TEXT NOT NULL,
    zone            TEXT NOT NULL,
    generation_type TEXT,
    value_mw        REAL,
    is_forecast     INTEGER DEFAULT 0,
    source          TEXT DEFAULT 'entsoe',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quality_quarantine (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    record_source   TEXT,
    record_data     TEXT,
    failed_gates    TEXT,
    quality_report  TEXT,
    quarantined_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_time TEXT NOT NULL,
    target_time     TEXT NOT NULL,
    zone            TEXT NOT NULL,
    predicted_price REAL,
    confidence      REAL,
    model_version   TEXT,
    features_hash   TEXT,
    actual_price    REAL,
    error           REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""


class Database:
    """
    Database manager with bitemporal query support.
    Uses PostgreSQL in production, SQLite for local dev.
    """

    def __init__(self, connection_url: Optional[str] = None):
        self.url = connection_url or DATABASE_URL
        self.use_postgres = HAS_POSTGRES and self.url.startswith("postgresql")
        self._sqlite_path = DATA_DIR / "energylens.db"

        if not self.use_postgres:
            logger.info(f"Using SQLite at {self._sqlite_path}")

    def initialize(self):
        """Create all tables if they don't exist."""
        if self.use_postgres:
            with self._pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA_SQL_POSTGRES)
                conn.commit()
            logger.info("PostgreSQL schema initialized")
        else:
            with self._sqlite_connection() as conn:
                conn.executescript(SCHEMA_SQL_SQLITE)
            logger.info("SQLite schema initialized")

    @contextmanager
    def _pg_connection(self):
        conn = psycopg2.connect(self.url)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _sqlite_connection(self):
        conn = sqlite3.connect(str(self._sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def connection(self):
        """Get a database connection (PostgreSQL or SQLite)."""
        if self.use_postgres:
            with self._pg_connection() as conn:
                yield conn
        else:
            with self._sqlite_connection() as conn:
                yield conn

    # ─── Insert Methods ───

    def insert_spot_prices(self, records: list[dict]) -> int:
        """Insert spot price records with bitemporal timestamps."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        with self.connection() as conn:
            cur = conn.cursor()
            for record in records:
                try:
                    if self.use_postgres:
                        cur.execute("""
                            INSERT INTO spot_prices
                                (valid_time, knowledge_time, zone, price_eur_mwh, price_dkk_mwh, source, quality_passed, quality_report)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            record["timestamp_utc"],
                            now,
                            record["zone"],
                            record.get("price_eur_mwh"),
                            record.get("price_dkk_mwh"),
                            record.get("source", "nordpool"),
                            record.get("quality_passed", True),
                            json.dumps(record.get("quality_report")) if record.get("quality_report") else None,
                        ))
                    else:
                        cur.execute("""
                            INSERT INTO spot_prices
                                (valid_time, knowledge_time, zone, price_eur_mwh, price_dkk_mwh, source, quality_passed, quality_report)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            record["timestamp_utc"],
                            now,
                            record["zone"],
                            record.get("price_eur_mwh"),
                            record.get("price_dkk_mwh"),
                            record.get("source", "nordpool"),
                            1 if record.get("quality_passed", True) else 0,
                            json.dumps(record.get("quality_report")) if record.get("quality_report") else None,
                        ))
                    count += 1
                except Exception as e:
                    logger.error(f"Insert failed for record: {e}")

            conn.commit()

        logger.info(f"Inserted {count} spot price records")
        return count

    def insert_weather(self, records: list[dict], location_key: str) -> int:
        """Insert weather forecast records."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        with self.connection() as conn:
            cur = conn.cursor()
            placeholder = "%s" if self.use_postgres else "?"

            for record in records:
                try:
                    sql = f"""
                        INSERT INTO weather_forecasts
                            (valid_time, knowledge_time, location_key,
                             temperature_2m, wind_speed_10m, wind_speed_100m,
                             wind_direction, cloud_cover, direct_radiation,
                             diffuse_radiation, precipitation)
                        VALUES ({', '.join([placeholder] * 11)})
                    """
                    cur.execute(sql, (
                        record.get("timestamp_utc"),
                        now,
                        location_key,
                        record.get("temperature_2m"),
                        record.get("wind_speed_10m"),
                        record.get("wind_speed_100m"),
                        record.get("wind_direction_10m"),
                        record.get("cloud_cover"),
                        record.get("direct_radiation"),
                        record.get("diffuse_radiation"),
                        record.get("precipitation"),
                    ))
                    count += 1
                except Exception as e:
                    logger.error(f"Weather insert failed: {e}")

            conn.commit()

        logger.info(f"Inserted {count} weather records for {location_key}")
        return count

    def quarantine_record(self, record: dict, quality_report: dict) -> None:
        """Store a failed record in quarantine."""
        with self.connection() as conn:
            cur = conn.cursor()
            placeholder = "%s" if self.use_postgres else "?"

            if self.use_postgres:
                cur.execute(f"""
                    INSERT INTO quality_quarantine
                        (record_source, record_data, failed_gates, quality_report)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})
                """, (
                    record.get("source", "unknown"),
                    json.dumps(record),
                    quality_report.get("failed_gates", []),
                    json.dumps(quality_report),
                ))
            else:
                cur.execute(f"""
                    INSERT INTO quality_quarantine
                        (record_source, record_data, failed_gates, quality_report)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})
                """, (
                    record.get("source", "unknown"),
                    json.dumps(record),
                    json.dumps(quality_report.get("failed_gates", [])),
                    json.dumps(quality_report),
                ))
            conn.commit()

    # ─── Bitemporal Query Methods ───

    def get_prices_as_of(
        self,
        zone: str,
        valid_start: str,
        valid_end: str,
        as_of: Optional[str] = None,
    ) -> list[dict]:
        """
        Point-in-time query: get prices as they were known at a specific time.

        "What did we know at {as_of} about prices between {valid_start} and {valid_end}?"

        If as_of is None, returns the latest known values (current view).
        """
        with self.connection() as conn:
            cur = conn.cursor()
            placeholder = "%s" if self.use_postgres else "?"

            if as_of:
                # Bitemporal query: latest knowledge_time <= as_of for each valid_time
                if self.use_postgres:
                    cur.execute("""
                        SELECT DISTINCT ON (valid_time)
                            valid_time, zone, price_eur_mwh, price_dkk_mwh,
                            knowledge_time, source
                        FROM spot_prices
                        WHERE zone = %s
                          AND valid_time BETWEEN %s AND %s
                          AND knowledge_time <= %s
                        ORDER BY valid_time, knowledge_time DESC
                    """, (zone, valid_start, valid_end, as_of))
                else:
                    cur.execute("""
                        SELECT valid_time, zone, price_eur_mwh, price_dkk_mwh,
                               knowledge_time, source
                        FROM spot_prices
                        WHERE zone = ?
                          AND valid_time BETWEEN ? AND ?
                          AND knowledge_time <= ?
                        GROUP BY valid_time
                        HAVING knowledge_time = MAX(knowledge_time)
                        ORDER BY valid_time
                    """, (zone, valid_start, valid_end, as_of))
            else:
                # Current view: latest knowledge for each valid_time
                if self.use_postgres:
                    cur.execute("""
                        SELECT DISTINCT ON (valid_time)
                            valid_time, zone, price_eur_mwh, price_dkk_mwh,
                            knowledge_time, source
                        FROM spot_prices
                        WHERE zone = %s
                          AND valid_time BETWEEN %s AND %s
                        ORDER BY valid_time, knowledge_time DESC
                    """, (zone, valid_start, valid_end))
                else:
                    cur.execute("""
                        SELECT valid_time, zone, price_eur_mwh, price_dkk_mwh,
                               knowledge_time, source
                        FROM spot_prices
                        WHERE zone = ?
                          AND valid_time BETWEEN ? AND ?
                        GROUP BY valid_time
                        HAVING knowledge_time = MAX(knowledge_time)
                        ORDER BY valid_time
                    """, (zone, valid_start, valid_end))

            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_record_counts(self) -> dict:
        """Get record counts for all tables — useful for dashboard."""
        tables = ["spot_prices", "weather_forecasts", "generation", "quality_quarantine", "predictions"]
        counts = {}

        with self.connection() as conn:
            cur = conn.cursor()
            for table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cur.fetchone()[0]
                except Exception:
                    counts[table] = 0

        return counts
