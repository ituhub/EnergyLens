"""
EnergyLens — Train ML models on backfilled data.

Usage (from energylens root):
    python -m ml.run_training                     # Default: DK1, 48h lookback
    python -m ml.run_training --zone DK2 --steps 72
    python -m ml.run_training --no-cv             # Skip cross-validation (faster)

Reads from the Phase 1 bitemporal SQLite database, builds features,
trains all 8 models, and saves to models/ directory.
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.features import build_energy_features
from ml.training import train_all_models, save_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data_from_sqlite(db_path: str, zone: str = "DK1") -> pd.DataFrame | None:
    """
    Load spot prices, weather, and generation data from SQLite database
    and merge into a single DataFrame indexed by hour.
    """
    db = Path(db_path)
    if not db.exists():
        logger.error(f"Database not found: {db}")
        return None

    conn = sqlite3.connect(str(db))

    # ── Load spot prices ─────────────────────────────────────────────
    try:
        # Try Phase 1 schema first (HourUTC, PriceArea)
        prices_df = pd.read_sql_query(
            """
            SELECT HourUTC, SpotPriceEUR, SpotPriceDKK
            FROM spot_prices
            WHERE PriceArea = ?
            ORDER BY HourUTC
            """,
            conn,
            params=(zone,),
            parse_dates=["HourUTC"],
        )
        prices_df = prices_df.set_index("HourUTC")
    except Exception:
        # Fall back to production schema (valid_time, zone)
        try:
            prices_df = pd.read_sql_query(
                """
                SELECT valid_time AS HourUTC,
                       price_eur_mwh AS SpotPriceEUR,
                       price_dkk_mwh AS SpotPriceDKK
                FROM spot_prices
                WHERE zone = ?
                ORDER BY valid_time
                """,
                conn,
                params=(zone,),
                parse_dates=["HourUTC"],
            )
            prices_df = prices_df.set_index("HourUTC")
            prices_df = prices_df.resample("h").mean().dropna()
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
            conn.close()
            return None

    logger.info(f"Loaded {len(prices_df)} price records for {zone}")

    # ── Load weather data ────────────────────────────────────────────
    weather_df = pd.DataFrame()
    for table, time_col in [("weather_data", "time"), ("weather_forecasts", "valid_time")]:
        try:
            cols_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_names = [r[1] for r in cols_info]
            select_parts = [f"{time_col} AS time"]
            for c in ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]:
                if c in col_names:
                    select_parts.append(c)
            if len(select_parts) > 1:
                q = "SELECT " + ", ".join(select_parts) + f" FROM {table} ORDER BY {time_col}"
                weather_df = pd.read_sql_query(q, conn, parse_dates=["time"])
                weather_df = weather_df.set_index("time")
                logger.info(f"Loaded {len(weather_df)} weather records from {table}")
                break
        except Exception:
            continue

    # ── Load generation data (ENTSO-E) ───────────────────────────────
    gen_df = pd.DataFrame()
    try:
        gen_raw = pd.read_sql_query(
            """
            SELECT valid_time AS time, generation_type, value_mw
            FROM generation
            WHERE zone = ?
            ORDER BY valid_time
            """,
            conn,
            params=(zone,),
            parse_dates=["time"],
        )
        if len(gen_raw) > 0:
            gen_pivot = gen_raw.pivot_table(
                index="time", columns="generation_type",
                values="value_mw", aggfunc="mean"
            )
            wind_cols = [c for c in gen_pivot.columns if "wind" in c.lower()]
            solar_cols = [c for c in gen_pivot.columns if "solar" in c.lower()]
            gen_df = pd.DataFrame(index=gen_pivot.index)
            if wind_cols:
                gen_df["wind_generation_mw"] = gen_pivot[wind_cols].sum(axis=1)
            if solar_cols:
                gen_df["solar_generation_mw"] = gen_pivot[solar_cols].sum(axis=1)
            renewable_cols = wind_cols + solar_cols
            if renewable_cols:
                gen_df["renewable_generation"] = gen_pivot[renewable_cols].sum(axis=1)
            gen_df["total_generation_mw"] = gen_pivot.sum(axis=1)
            for col in gen_pivot.columns:
                gen_df[f"gen_{col}"] = gen_pivot[col]
            gen_df = gen_df.resample("h").mean()
            logger.info(f"Loaded {len(gen_df)} generation records for {zone}")
        else:
            logger.warning(f"No generation data found for {zone}")
    except Exception as e:
        logger.warning(f"Generation data not available: {e}")

    conn.close()

    # ── Merge on hourly index ────────────────────────────────────────
    merged = prices_df
    if not weather_df.empty:
        weather_hourly = weather_df.resample("h").mean()
        merged = merged.join(weather_hourly, how="left")
    if not gen_df.empty:
        merged = merged.join(gen_df, how="left")
        logger.info(f"Added generation features: {list(gen_df.columns)}")
    merged = merged.sort_index().dropna(subset=["SpotPriceEUR"])
    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged

def main():
    parser = argparse.ArgumentParser(description="Train EnergyLens ML models")
    parser.add_argument("--zone", default="DK1", help="Nord Pool bidding zone")
    parser.add_argument("--db", default="data/energylens.db", help="SQLite database path")
    parser.add_argument("--time-step", type=int, default=48, help="Lookback window in hours")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument("--model-dir", default="models", help="Output directory for models")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EnergyLens Phase 2 — ML Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Zone: {args.zone}  |  Lookback: {args.time_step}h  |  CV: {not args.no_cv}")

    # 1. Load data
    raw_df = load_data_from_sqlite(args.db, args.zone)
    if raw_df is None or len(raw_df) < 200:
        logger.error(f"Insufficient data ({len(raw_df) if raw_df is not None else 0} rows). "
                     "Run a backfill first: python -m pipeline.ingest --backfill --days 365")
        sys.exit(1)

    # 2. Build features
    logger.info("Building energy features…")
    featured_df = build_energy_features(raw_df)
    logger.info(f"Feature matrix: {featured_df.shape[0]} rows × {featured_df.shape[1]} features")

    # 3. Train
    feature_cols = featured_df.columns.tolist()
    models, scaler, config = train_all_models(
        featured_df, feature_cols,
        zone=args.zone, time_step=args.time_step,
        run_cv=not args.no_cv,
    )

    if models is None:
        logger.error("Training failed — no models produced")
        sys.exit(1)

    # 4. Save
    save_models(models, scaler, config, zone=args.zone, base_dir=args.model_dir)

    # 5. Summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Models trained: {len(models)}  →  {sorted(models.keys())}")
    logger.info(f"Price range (dynamic): €{config['price_range'][0]:.2f} — €{config['price_range'][1]:.2f}")
    if config.get("ensemble_weights"):
        logger.info("Ensemble weights (from CV):")
        for name, w in sorted(config["ensemble_weights"].items(), key=lambda x: -x[1]):
            logger.info(f"  {name:25s}  {w:.3f}")
    logger.info(f"Artifacts saved to: {args.model_dir}/")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  • Integrate with FastAPI:  GET /api/forecast?zone=DK1&hours=24")
    logger.info("  • Add to React dashboard:  Forecast chart component")


if __name__ == "__main__":
    main()
