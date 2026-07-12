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
    Load spot prices and weather data from Phase 1 SQLite database
    and merge into a single DataFrame indexed by hour.
    """
    db = Path(db_path)
    if not db.exists():
        logger.error(f"Database not found: {db}")
        return None

    conn = sqlite3.connect(str(db))

    # ── Load spot prices ─────────────────────────────────────────────
    try:
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
        logger.info(f"Loaded {len(prices_df)} price records for {zone}")
    except Exception as e:
        logger.error(f"Failed to load prices: {e}")
        conn.close()
        return None

    # ── Load weather data ────────────────────────────────────────────
    try:
        weather_df = pd.read_sql_query(
            """
            SELECT time, temperature_2m, wind_speed_10m, shortwave_radiation
            FROM weather_data
            ORDER BY time
            """,
            conn,
            parse_dates=["time"],
        )
        weather_df = weather_df.set_index("time")
        logger.info(f"Loaded {len(weather_df)} weather records")
    except Exception as e:
        logger.warning(f"Weather data not available: {e}")
        weather_df = pd.DataFrame()

    conn.close()

    # ── Merge on hourly index ────────────────────────────────────────
    if not weather_df.empty:
        # Resample weather to hourly (in case of sub-hourly data)
        weather_hourly = weather_df.resample("h").mean()
        merged = prices_df.join(weather_hourly, how="left")
    else:
        merged = prices_df

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
