#!/usr/bin/env python3
"""
EnergyLens — Comprehensive Pre-Deploy Diagnostic
═══════════════════════════════════════════════════

Run from project root:
    python diagnose_energylens.py

Checks every layer of the stack:
  1. Project structure & imports
  2. Configuration & environment
  3. Database health & data freshness
  4. Data pipeline connectors
  5. Feature engineering pipeline
  6. Model loading & architecture validation
  7. Single-step ensemble prediction
  8. Multi-step forecast (24h)
  9. Ensemble safety rails (outlier filter, clamping, frozen detection)
 10. Confidence scoring
 11. API endpoint smoke tests (if server is running)
 12. Frontend build check

Exit codes:
  0 = All critical checks passed
  1 = One or more critical failures
"""

import sys
import os
import time
import json
import sqlite3
import importlib
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ─── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "energylens"))

# ═════════════════════════════════════════════════════════════════════
# REPORT INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════

class DiagnosticReport:
    """Collects results across all checks and prints a final summary."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"

    ICONS = {PASS: "✅", WARN: "⚠️", FAIL: "❌", SKIP: "⏭️"}
    COLORS = {PASS: "\033[92m", WARN: "\033[93m", FAIL: "\033[91m", SKIP: "\033[90m"}
    RESET = "\033[0m"

    def __init__(self):
        self.sections: list[dict] = []
        self._current_section: str = ""
        self._checks: list[dict] = []
        self.start_time = time.time()

    def section(self, title: str):
        """Start a new diagnostic section."""
        if self._current_section and self._checks:
            self.sections.append({
                "title": self._current_section,
                "checks": list(self._checks),
            })
        self._current_section = title
        self._checks = []
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    def check(self, name: str, status: str, detail: str = "", data: dict = None):
        """Record a single check result."""
        icon = self.ICONS.get(status, "?")
        color = self.COLORS.get(status, "")
        print(f"  {icon} {color}{name}{self.RESET}")
        if detail:
            for line in detail.split("\n"):
                print(f"       {line}")
        self._checks.append({
            "name": name,
            "status": status,
            "detail": detail,
            "data": data or {},
        })

    def finalize(self) -> int:
        """Print summary and return exit code."""
        # Flush last section
        if self._current_section and self._checks:
            self.sections.append({
                "title": self._current_section,
                "checks": list(self._checks),
            })

        elapsed = time.time() - self.start_time
        all_checks = [c for s in self.sections for c in s["checks"]]
        counts = defaultdict(int)
        for c in all_checks:
            counts[c["status"]] += 1

        print(f"\n{'═' * 60}")
        print(f"  ENERGYLENS DIAGNOSTIC SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Total checks:  {len(all_checks)}")
        print(f"  ✅ Passed:     {counts[self.PASS]}")
        print(f"  ⚠️  Warnings:   {counts[self.WARN]}")
        print(f"  ❌ Failed:     {counts[self.FAIL]}")
        print(f"  ⏭️  Skipped:    {counts[self.SKIP]}")
        print(f"  Time:          {elapsed:.1f}s")
        print(f"{'═' * 60}")

        # List all failures
        failures = [c for c in all_checks if c["status"] == self.FAIL]
        if failures:
            print(f"\n  ❌ CRITICAL FAILURES:")
            for f in failures:
                print(f"     • {f['name']}: {f['detail'][:120]}")
            print()

        # List all warnings
        warnings = [c for c in all_checks if c["status"] == self.WARN]
        if warnings:
            print(f"\n  ⚠️  WARNINGS:")
            for w in warnings:
                print(f"     • {w['name']}: {w['detail'][:120]}")
            print()

        if not failures:
            print("  🚀 ALL CRITICAL CHECKS PASSED — ready to deploy!\n")
            return 0
        else:
            print("  🛑 FIX FAILURES BEFORE DEPLOYING\n")
            return 1


report = DiagnosticReport()


# ═════════════════════════════════════════════════════════════════════
# 1. PROJECT STRUCTURE & IMPORTS
# ═════════════════════════════════════════════════════════════════════

def check_project_structure():
    report.section("1 · Project Structure & Imports")

    # Critical directories
    expected_dirs = [
        "ml", "api", "config", "connectors", "core",
        "pipeline", "data", "models",
    ]
    for d in expected_dirs:
        p = PROJECT_ROOT / d
        if p.is_dir():
            report.check(f"Directory: {d}/", report.PASS)
        else:
            severity = report.FAIL if d in ("ml", "api", "config", "data") else report.WARN
            report.check(f"Directory: {d}/", severity, f"Not found at {p}")

    # Critical Python files
    critical_files = {
        "ml/ensemble.py": "Ensemble prediction logic",
        "ml/models.py": "Model architectures",
        "ml/features.py": "Feature engineering",
        "api/main.py": "FastAPI application",
        "api/forecast_service.py": "Forecast service bridge",
        "config/constants.py": "Market constants",
        "config/settings.py": "Environment settings",
        "core/database.py": "Database layer",
    }
    for fpath, desc in critical_files.items():
        if (PROJECT_ROOT / fpath).exists():
            report.check(f"File: {fpath}", report.PASS, desc)
        else:
            report.check(f"File: {fpath}", report.FAIL, f"MISSING — {desc}")

    # Try critical imports
    modules_to_test = [
        ("config.constants", "Constants"),
        ("config.settings", "Settings"),
        ("ml.models", "Model definitions"),
        ("ml.features", "Feature engineering"),
        ("ml.ensemble", "Ensemble logic"),
    ]
    for mod_name, desc in modules_to_test:
        try:
            importlib.import_module(mod_name)
            report.check(f"Import: {mod_name}", report.PASS)
        except Exception as e:
            report.check(f"Import: {mod_name}", report.FAIL, str(e)[:200])

    # Check key dependencies
    deps = ["torch", "numpy", "pandas", "sklearn", "fastapi", "aiohttp"]
    for dep in deps:
        try:
            mod = importlib.import_module(dep)
            ver = getattr(mod, "__version__", "?")
            report.check(f"Dependency: {dep}", report.PASS, f"v{ver}")
        except ImportError:
            report.check(f"Dependency: {dep}", report.FAIL, "Not installed")

    try:
        import xgboost
        report.check("Dependency: xgboost", report.PASS, f"v{xgboost.__version__}")
    except ImportError:
        report.check("Dependency: xgboost", report.WARN, "Not installed — xgboost model will be unavailable")


# ═════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION & ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════

def check_configuration():
    report.section("2 · Configuration & Environment")

    try:
        from config.settings import (
            DATABASE_URL, DATA_DIR, ENTSOE_API_KEY,
            ENVIRONMENT, LOG_LEVEL,
        )
        report.check("Settings loaded", report.PASS, f"env={ENVIRONMENT}, log={LOG_LEVEL}")
        report.check("DATABASE_URL", report.PASS, DATABASE_URL[:60] + "..." if len(DATABASE_URL) > 60 else DATABASE_URL)
        report.check("DATA_DIR", report.PASS if DATA_DIR.exists() else report.FAIL, str(DATA_DIR))

        if ENTSOE_API_KEY:
            report.check("ENTSOE_API_KEY", report.PASS, f"Set ({len(ENTSOE_API_KEY)} chars)")
        else:
            report.check("ENTSOE_API_KEY", report.WARN,
                         "NOT SET — ENTSO-E generation data will be empty.\n"
                         "Register free at https://transparency.entsoe.eu/\n"
                         "Then add ENTSOE_API_KEY=... to .env")

    except Exception as e:
        report.check("Settings", report.FAIL, str(e))
        return

    try:
        from config.constants import (
            ACTIVE_ZONES, BIDDING_ZONES, ENSEMBLE_MODELS,
            QUALITY_GATES, WEATHER_LOCATIONS,
        )
        report.check("Constants loaded", report.PASS,
                     f"zones={ACTIVE_ZONES}, models={len(ENSEMBLE_MODELS)}, "
                     f"weather_locs={len(WEATHER_LOCATIONS)}")

        # Validate zone codes
        for zone in ACTIVE_ZONES:
            if zone in BIDDING_ZONES:
                code = BIDDING_ZONES[zone]["entsoe_code"]
                report.check(f"Zone {zone}", report.PASS, f"ENTSO-E: {code}")
            else:
                report.check(f"Zone {zone}", report.FAIL, "Not in BIDDING_ZONES")

    except Exception as e:
        report.check("Constants", report.FAIL, str(e))


# ═════════════════════════════════════════════════════════════════════
# 3. DATABASE HEALTH & DATA FRESHNESS
# ═════════════════════════════════════════════════════════════════════

def check_database():
    report.section("3 · Database Health & Data Freshness")

    db_path = PROJECT_ROOT / "data" / "energylens.db"
    if not db_path.exists():
        report.check("Database file", report.FAIL, f"Not found at {db_path}")
        return None
    else:
        size_mb = db_path.stat().st_size / (1024 * 1024)
        report.check("Database file", report.PASS, f"{db_path} ({size_mb:.1f} MB)")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check tables exist
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()]
    report.check("Tables", report.PASS if tables else report.FAIL, ", ".join(tables) if tables else "No tables!")

    # Row counts per table
    counts = {}
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        counts[table] = count
        status = report.PASS if count > 0 else report.WARN
        report.check(f"  {table}", status, f"{count:,} rows")

    # Spot price freshness
    if "spot_prices" in tables and counts.get("spot_prices", 0) > 0:
        row = conn.execute(
            "SELECT MIN(valid_time) AS oldest, MAX(valid_time) AS newest FROM spot_prices"
        ).fetchone()
        oldest, newest = row["oldest"], row["newest"]
        report.check("Spot price range", report.PASS, f"{oldest} → {newest}")

        try:
            newest_dt = datetime.fromisoformat(newest.replace("T", " ").split("+")[0])
            age_hours = (datetime.utcnow() - newest_dt).total_seconds() / 3600
            if age_hours > 48:
                report.check("Spot price freshness", report.WARN,
                             f"Newest record is {age_hours:.0f}h old — run: python auto_refresh.py")
            else:
                report.check("Spot price freshness", report.PASS, f"{age_hours:.1f}h old")
        except Exception:
            report.check("Spot price freshness", report.WARN, "Could not parse newest timestamp")

        # Check zone distribution
        zones = conn.execute(
            "SELECT zone, COUNT(*) as cnt FROM spot_prices GROUP BY zone"
        ).fetchall()
        for z in zones:
            report.check(f"  Zone {z['zone']}", report.PASS, f"{z['cnt']:,} rows")

        # Check for negative prices (expected in Nordic markets)
        neg = conn.execute(
            "SELECT COUNT(*) FROM spot_prices WHERE price_eur_mwh < 0"
        ).fetchone()[0]
        if neg > 0:
            report.check("Negative prices", report.PASS,
                         f"{neg} records with price < 0 (normal for Nordic wind oversupply)")

    # Weather freshness
    if "weather_forecasts" in tables and counts.get("weather_forecasts", 0) > 0:
        row = conn.execute(
            "SELECT MAX(valid_time) AS newest FROM weather_forecasts"
        ).fetchone()
        report.check("Weather data", report.PASS, f"newest: {row['newest']}")

    # Generation data
    gen_count = counts.get("generation", 0)
    if gen_count == 0:
        report.check("Generation data", report.WARN,
                     "0 rows — ENTSO-E connector needs API key or manual ingest")

    # Bitemporal check: verify knowledge_time is populated
    if "spot_prices" in tables:
        has_kt = conn.execute(
            "SELECT COUNT(*) FROM spot_prices WHERE knowledge_time IS NOT NULL AND knowledge_time != ''"
        ).fetchone()[0]
        total = counts.get("spot_prices", 0)
        if total > 0 and has_kt == total:
            report.check("Bitemporal layer", report.PASS,
                         f"All {total:,} spot prices have knowledge_time")
        elif has_kt > 0:
            report.check("Bitemporal layer", report.WARN,
                         f"{has_kt}/{total} rows have knowledge_time")
        else:
            report.check("Bitemporal layer", report.WARN, "No knowledge_time values found")

    conn.close()
    return counts


# ═════════════════════════════════════════════════════════════════════
# 4. DATA PIPELINE CONNECTORS
# ═════════════════════════════════════════════════════════════════════

def check_connectors():
    report.section("4 · Data Pipeline Connectors")

    # Nord Pool
    try:
        from connectors.nordpool import NordPoolProvider
        np_prov = NordPoolProvider()
        report.check("NordPoolProvider", report.PASS,
                     f"source={np_prov.source_name}, url={np_prov.base_url}")
    except Exception as e:
        report.check("NordPoolProvider", report.FAIL, str(e)[:200])

    # Weather
    try:
        from connectors.weather import WeatherProvider
        wp = WeatherProvider()
        report.check("WeatherProvider", report.PASS,
                     f"source={wp.source_name}, url={wp.base_url}")
    except Exception as e:
        report.check("WeatherProvider", report.FAIL, str(e)[:200])

    # ENTSO-E
    try:
        from connectors.entsoe import ENTSOEProvider
        ep = ENTSOEProvider()
        report.check("ENTSOEProvider", report.PASS,
                     f"source={ep.source_name}, url={ep.base_url}")
    except Exception as e:
        report.check("ENTSOEProvider", report.FAIL, str(e)[:200])

    # Quality Gate
    try:
        from core.quality_gate import QualityGate
        qg = QualityGate()
        # Test with a sample record
        sample = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "zone": "DK1",
            "price_eur_mwh": 45.0,
            "price_dkk_mwh": 335.0,
            "source": "test",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        qr = qg.validate(sample, data_type="spot_price")
        report.check("QualityGate", report.PASS,
                     f"Sample validation: passed={qr.passed}, gates={len(qr.gates)}")
    except Exception as e:
        report.check("QualityGate", report.FAIL, str(e)[:200])


# ═════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING PIPELINE
# ═════════════════════════════════════════════════════════════════════

def check_features():
    report.section("5 · Feature Engineering Pipeline")

    try:
        from ml.features import build_energy_features, get_energy_price_range, get_max_step_change
    except Exception as e:
        report.check("Feature imports", report.FAIL, str(e))
        return None

    # Test price range
    for zone in ["DK1", "DK2"]:
        pr = get_energy_price_range(zone)
        report.check(f"Price range ({zone})", report.PASS, f"€{pr[0]} → €{pr[1]}")

    msc = get_max_step_change("DK1")
    report.check("Max step change", report.PASS, f"{msc:.0%}")

    # Build features from database data
    import pandas as pd
    db_path = PROJECT_ROOT / "data" / "energylens.db"
    if not db_path.exists():
        report.check("Feature build test", report.SKIP, "No database")
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            """SELECT valid_time AS HourUTC, price_eur_mwh AS SpotPriceEUR,
                      price_dkk_mwh AS SpotPriceDKK
               FROM spot_prices WHERE zone = 'DK1'
               ORDER BY valid_time DESC LIMIT 800""",
            conn, parse_dates=["HourUTC"],
        )
    except Exception as e:
        report.check("Feature build test", report.FAIL, f"SQL error: {e}")
        conn.close()
        return None
    conn.close()

    if df.empty:
        report.check("Feature build test", report.SKIP, "No DK1 spot prices")
        return None

    df = df.sort_values("HourUTC").set_index("HourUTC")
    df = df.resample("h").mean().dropna()
    report.check("Raw data loaded", report.PASS, f"{len(df)} hourly rows")

    try:
        featured = build_energy_features(df)
        n_features = len(featured.columns)
        report.check("Feature build", report.PASS, f"{n_features} features from {len(featured)} rows")

        # Verify critical columns exist
        critical_cols = ["Close", "hour_sin", "hour_cos", "dow_sin", "price_lag_1",
                         "price_lag_24", "price_mean_24", "volatility_24h"]
        missing = [c for c in critical_cols if c not in featured.columns]
        if not missing:
            report.check("Critical features present", report.PASS, ", ".join(critical_cols[:5]) + "...")
        else:
            report.check("Critical features present", report.WARN, f"Missing: {missing}")

        # Check for NaN/Inf
        nan_count = featured.isna().sum().sum()
        inf_count = ((featured == float('inf')) | (featured == float('-inf'))).sum().sum()
        if nan_count == 0 and inf_count == 0:
            report.check("Feature data quality", report.PASS, "No NaN or Inf values")
        else:
            report.check("Feature data quality", report.WARN,
                         f"{nan_count} NaN, {inf_count} Inf values")

        return featured

    except Exception as e:
        report.check("Feature build", report.FAIL, traceback.format_exc()[-300:])
        return None


# ═════════════════════════════════════════════════════════════════════
# 6. MODEL LOADING & ARCHITECTURE VALIDATION
# ═════════════════════════════════════════════════════════════════════

def check_models():
    report.section("6 · Model Loading & Architecture")

    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        report.check("Models directory", report.FAIL, f"Not found: {models_dir}")
        return None, None

    # List model files
    model_files = sorted(models_dir.glob("*.pt")) + sorted(models_dir.glob("*.pth")) + sorted(models_dir.glob("*.pkl"))
    joblib_files = sorted(models_dir.glob("*.joblib"))
    all_files = model_files + joblib_files

    if not all_files:
        report.check("Model files", report.FAIL,
                     f"No .pt/.pth/.pkl/.joblib files in {models_dir}\n"
                     "Run: python -m ml.run_training")
        return None, None

    report.check("Model files found", report.PASS,
                 f"{len(all_files)} files:\n       " + "\n       ".join(
                     f"{f.name} ({f.stat().st_size / 1024:.0f} KB)" for f in all_files))

    # Try to load via the training module
    try:
        from ml.training import load_models
    except ImportError as e:
        report.check("ml.training import", report.FAIL, str(e))
        return None, None

    loaded = {}
    config = {}
    for zone in ["DK1"]:
        try:
            models, cfg = load_models(zone=zone, base_dir=str(models_dir))
            if models:
                loaded = models
                config = cfg
                report.check(f"load_models({zone})", report.PASS,
                             f"{len(models)} models: {sorted(models.keys())}")

                # Check scaler
                scaler = cfg.get("scaler")
                if scaler is not None:
                    report.check("Scaler", report.PASS,
                                 f"RobustScaler with {scaler.scale_.shape[0]} features")
                else:
                    report.check("Scaler", report.WARN, "No scaler in config — predictions may be unscaled")

                # Check used_features
                uf = cfg.get("used_features", [])
                report.check("used_features", report.PASS if uf else report.WARN,
                             f"{len(uf)} features" + (f" (first 5: {uf[:5]})" if uf else " — EMPTY"))

                # Check time_step
                ts = cfg.get("time_step", "NOT SET")
                report.check("time_step (seq_len)", report.PASS, str(ts))

                # Check price_range
                pr = cfg.get("price_range")
                if pr:
                    report.check("Training price_range", report.PASS, f"€{pr[0]:.2f} → €{pr[1]:.2f}")
                else:
                    report.check("Training price_range", report.WARN, "Not set — using fallback from constants")

            else:
                report.check(f"load_models({zone})", report.FAIL, "Returned empty dict")

        except Exception as e:
            report.check(f"load_models({zone})", report.FAIL, traceback.format_exc()[-400:])

    # Validate model architectures
    import torch
    for name, model in loaded.items():
        try:
            if hasattr(model, "parameters"):
                params = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                report.check(f"  {name}", report.PASS,
                             f"{params:,} params ({trainable:,} trainable)")

                # Check eval mode
                if hasattr(model, "training") and model.training:
                    report.check(f"  {name} eval mode", report.WARN,
                                 "Model is in TRAINING mode — should be eval() for inference")
            else:
                # sklearn / xgboost model
                report.check(f"  {name}", report.PASS, f"sklearn/xgboost model ({type(model).__name__})")
        except Exception as e:
            report.check(f"  {name}", report.WARN, str(e)[:150])

    return loaded, config


# ═════════════════════════════════════════════════════════════════════
# 7. SINGLE-STEP ENSEMBLE PREDICTION
# ═════════════════════════════════════════════════════════════════════

def check_single_prediction(models, config):
    report.section("7 · Single-Step Ensemble Prediction")

    if not models:
        report.check("Single-step test", report.SKIP, "No models loaded")
        return

    try:
        from ml.ensemble import ensemble_predict
        from ml.features import build_energy_features
    except ImportError as e:
        report.check("Ensemble import", report.FAIL, str(e))
        return

    scaler = config.get("scaler")
    time_step = config.get("time_step", 48)
    used_features = config.get("used_features", [])
    cv_weights = config.get("ensemble_weights")

    # Build a test sequence from database
    import numpy as np
    import pandas as pd

    db_path = PROJECT_ROOT / "data" / "energylens.db"
    if not db_path.exists():
        report.check("Single-step test", report.SKIP, "No database")
        return

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        f"""SELECT valid_time AS HourUTC, price_eur_mwh AS SpotPriceEUR,
                   price_dkk_mwh AS SpotPriceDKK
            FROM spot_prices WHERE zone = 'DK1'
            ORDER BY valid_time DESC LIMIT {(time_step + 200) * 4}""",
        conn, parse_dates=["HourUTC"],
    )
    conn.close()

    if df.empty or len(df) < time_step:
        report.check("Single-step test", report.SKIP, f"Insufficient data ({len(df)} rows)")
        return

    df = df.sort_values("HourUTC").set_index("HourUTC").resample("h").mean().dropna()
    current_price = float(df["SpotPriceEUR"].iloc[-1])
    report.check("Current price (DK1)", report.PASS, f"€{current_price:.2f}")

    # Build features
    featured = build_energy_features(df)

    # Align to used_features
    aligned = pd.DataFrame(0.0, index=featured.index, columns=used_features)
    for col in used_features:
        if col in featured.columns:
            aligned[col] = featured[col]

    # Scale
    if scaler is not None:
        scaled = scaler.transform(aligned.values)
    else:
        from sklearn.preprocessing import RobustScaler
        temp_scaler = RobustScaler()
        scaled = temp_scaler.fit_transform(aligned.values)

    if len(scaled) < time_step:
        report.check("Single-step test", report.SKIP, f"Scaled data too short: {len(scaled)} < {time_step}")
        return

    last_seq = scaled[-time_step:].reshape(1, time_step, -1)
    report.check("Input sequence", report.PASS,
                 f"shape={last_seq.shape}, dtype={last_seq.dtype}")

    # Run ensemble
    t0 = time.time()
    try:
        pred, per_model = ensemble_predict(
            models, last_seq, scaler, "DK1",
            cv_weights=cv_weights, price_range=None,
            current_price=current_price,
        )
        elapsed = time.time() - t0

        if pred is not None:
            report.check("Ensemble prediction", report.PASS,
                         f"€{pred:.2f} in {elapsed:.2f}s")
        else:
            report.check("Ensemble prediction", report.FAIL, "Returned None")

        # Per-model breakdown
        if per_model:
            values = list(per_model.values())
            std = np.std(values)
            mean = np.mean(values)
            report.check("Per-model predictions", report.PASS,
                         f"{len(per_model)}/{len(models)} models produced output\n"
                         f"       mean=€{mean:.2f}, σ=€{std:.2f}")
            for name, val in sorted(per_model.items(), key=lambda x: x[1], reverse=True):
                diff = val - mean
                report.check(f"  {name}", report.PASS, f"€{val:.2f}  ({'+' if diff >= 0 else ''}{diff:.2f})")

            # Check for frozen/constant predictions
            unique_vals = set(round(v, 2) for v in values)
            if len(unique_vals) < len(values) * 0.5:
                report.check("Model diversity", report.WARN,
                             f"Only {len(unique_vals)} unique predictions from {len(values)} models")
        else:
            report.check("Per-model predictions", report.FAIL, "No models returned output")

    except Exception as e:
        report.check("Ensemble prediction", report.FAIL, traceback.format_exc()[-400:])


# ═════════════════════════════════════════════════════════════════════
# 8. MULTI-STEP FORECAST (24h)
# ═════════════════════════════════════════════════════════════════════

def check_multi_step(models, config):
    report.section("8 · Multi-Step Forecast (24h)")

    if not models:
        report.check("Multi-step test", report.SKIP, "No models loaded")
        return

    try:
        from ml.ensemble import multi_step_forecast
        from ml.features import build_energy_features
    except ImportError as e:
        report.check("Imports", report.FAIL, str(e))
        return

    import numpy as np
    import pandas as pd

    scaler = config.get("scaler")
    time_step = config.get("time_step", 48)
    used_features = config.get("used_features", [])
    cv_weights = config.get("ensemble_weights")

    db_path = PROJECT_ROOT / "data" / "energylens.db"
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        f"""SELECT valid_time AS HourUTC, price_eur_mwh AS SpotPriceEUR,
                   price_dkk_mwh AS SpotPriceDKK
            FROM spot_prices WHERE zone = 'DK1'
            ORDER BY valid_time DESC LIMIT {(time_step + 200) * 4}""",
        conn, parse_dates=["HourUTC"],
    )
    conn.close()

    if df.empty:
        report.check("Multi-step test", report.SKIP, "No data")
        return

    df = df.sort_values("HourUTC").set_index("HourUTC").resample("h").mean().dropna()

    # Capture last timestamp
    last_ts = df.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_timestamp = last_ts.to_pydatetime()
    else:
        last_timestamp = pd.Timestamp(last_ts).to_pydatetime()
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

    featured = build_energy_features(df)
    aligned = pd.DataFrame(0.0, index=featured.index, columns=used_features)
    for col in used_features:
        if col in featured.columns:
            aligned[col] = featured[col]

    if scaler is not None:
        scaled = scaler.transform(aligned.values)
    else:
        from sklearn.preprocessing import RobustScaler
        scaled = RobustScaler().fit_transform(aligned.values)

    last_seq = scaled[-time_step:].reshape(1, time_step, -1)

    t0 = time.time()
    try:
        forecasts = multi_step_forecast(
            models, last_seq, scaler, steps=24, zone="DK1",
            cv_weights=cv_weights, price_range=None,
            feature_names=used_features,
            last_timestamp=last_timestamp,
        )
        elapsed = time.time() - t0

        if forecasts and len(forecasts) == 24:
            report.check("24-step forecast", report.PASS,
                         f"Completed in {elapsed:.1f}s")
        elif forecasts:
            report.check("24-step forecast", report.WARN,
                         f"Only {len(forecasts)}/24 steps completed ({elapsed:.1f}s)")
        else:
            report.check("24-step forecast", report.FAIL, "Empty forecast returned")
            return

        # Analyze forecast quality
        prices = np.array(forecasts)
        report.check("Forecast range", report.PASS,
                     f"€{prices.min():.2f} → €{prices.max():.2f}, mean=€{prices.mean():.2f}")

        # Check for constant forecasts (all same value = broken)
        if np.std(prices) < 0.01:
            report.check("Forecast variance", report.FAIL,
                         f"σ=€{np.std(prices):.4f} — forecast is essentially flat")
        elif np.std(prices) < 1.0:
            report.check("Forecast variance", report.WARN,
                         f"σ=€{np.std(prices):.2f} — very low variance, clamping may be too tight")
        else:
            report.check("Forecast variance", report.PASS, f"σ=€{np.std(prices):.2f}")

        # Check for NaN in forecasts
        nan_count = np.isnan(prices).sum()
        if nan_count > 0:
            report.check("Forecast NaN check", report.FAIL, f"{nan_count} NaN values in forecast")
        else:
            report.check("Forecast NaN check", report.PASS, "No NaN values")

        # Check if forecast has reasonable range for energy prices
        if prices.min() < -500 or prices.max() > 5000:
            report.check("Forecast bounds", report.WARN,
                         f"Extreme values detected: min=€{prices.min():.2f}, max=€{prices.max():.2f}")
        else:
            report.check("Forecast bounds", report.PASS, "Within €-500 to €5000 range")

        # Step-to-step changes
        diffs = np.diff(prices)
        max_jump = np.max(np.abs(diffs))
        report.check("Max hourly jump", report.PASS if max_jump < 200 else report.WARN,
                     f"€{max_jump:.2f} (largest step-to-step change)")

        # Print the forecast curve (ASCII sparkline)
        normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
        bars = "▁▂▃▄▅▆▇█"
        sparkline = "".join(bars[min(int(v * 7), 7)] for v in normalized)
        report.check("Forecast shape", report.PASS,
                     f"{sparkline}\n"
                     f"       h1=€{prices[0]:.2f}  h6=€{prices[5]:.2f}  "
                     f"h12=€{prices[11]:.2f}  h18=€{prices[17]:.2f}  h24=€{prices[23]:.2f}")

    except Exception as e:
        report.check("24-step forecast", report.FAIL, traceback.format_exc()[-500:])


# ═════════════════════════════════════════════════════════════════════
# 9. ENSEMBLE SAFETY RAILS
# ═════════════════════════════════════════════════════════════════════

def check_safety_rails():
    report.section("9 · Ensemble Safety Rails")

    try:
        from ml.ensemble import ensemble_predict, multi_step_forecast, calculate_confidence
        import numpy as np
    except ImportError as e:
        report.check("Safety rail imports", report.FAIL, str(e))
        return

    # Check outlier filter threshold
    import inspect
    source = inspect.getsource(ensemble_predict)

    if "max_dev = 0.35" in source:
        report.check("Outlier threshold", report.PASS, "35% (energy-appropriate)")
    elif "max_dev = 0.15" in source:
        report.check("Outlier threshold", report.WARN,
                     "Still at 15% — apply the ensemble.py patch to widen to 35%")
    else:
        # Extract the value
        for line in source.split("\n"):
            if "max_dev" in line and "=" in line:
                report.check("Outlier threshold", report.PASS, line.strip())
                break

    # Check MIN_ABS_STEP
    ms_source = inspect.getsource(multi_step_forecast)
    if "MIN_ABS_STEP = 50.0" in ms_source:
        report.check("MIN_ABS_STEP", report.PASS, "€50 (energy-appropriate)")
    elif "MIN_ABS_STEP = 15.0" in ms_source:
        report.check("MIN_ABS_STEP", report.WARN,
                     "Still at €15 — apply the ensemble.py patch to increase to €50")
    else:
        for line in ms_source.split("\n"):
            if "MIN_ABS_STEP" in line and "=" in line:
                report.check("MIN_ABS_STEP", report.PASS, line.strip())
                break

    # Check frozen detection
    if "frozen_models" in ms_source or "FROZEN" in ms_source:
        report.check("Frozen model detection", report.PASS, "Implemented")
    else:
        report.check("Frozen model detection", report.WARN,
                     "Not found — apply the ensemble.py patch to add frozen-model detection")

    # Check confidence scoring CV multiplier
    conf_source = inspect.getsource(calculate_confidence)
    if "cv * 200" in conf_source:
        report.check("Confidence CV multiplier", report.PASS, "200 (energy-tuned)")
    elif "cv * 500" in conf_source:
        report.check("Confidence CV multiplier", report.WARN,
                     "Still at 500 — too punishing for energy models, apply patch")

    # Test confidence function with realistic spread
    preds = {
        "advanced_transformer": 96.0,
        "cnn_lstm": 86.0,
        "enhanced_nbeats": 70.0,
        "lstm_gru_ensemble": 75.0,
        "enhanced_tcn": 35.0,
        "enhanced_informer": 40.0,
        "xgboost": 50.0,
    }
    conf = calculate_confidence(preds, "DK1")
    if conf >= 50:
        report.check("Confidence score (test)", report.PASS,
                     f"{conf:.1f}% — realistic spread gives {conf:.1f}% confidence")
    else:
        report.check("Confidence score (test)", report.WARN,
                     f"{conf:.1f}% — too low for 7-model ensemble. "
                     "CV multiplier may still be too aggressive")


# ═════════════════════════════════════════════════════════════════════
# 10. FORECAST SERVICE (END-TO-END)
# ═════════════════════════════════════════════════════════════════════

def check_forecast_service():
    report.section("10 · Forecast Service (End-to-End)")

    try:
        from api.forecast_service import ForecastService
    except ImportError as e:
        report.check("ForecastService import", report.FAIL, str(e))
        return

    db_path = PROJECT_ROOT / "data" / "energylens.db"
    models_dir = PROJECT_ROOT / "models"

    if not db_path.exists():
        report.check("ForecastService", report.SKIP, "No database")
        return
    if not models_dir.exists() or not any(models_dir.glob("*.pt")):
        report.check("ForecastService", report.SKIP, "No model files")
        return

    svc = ForecastService(db_path=str(db_path), model_dir=str(models_dir))

    t0 = time.time()
    try:
        result = svc.forecast(zone="DK1", hours=24)
        elapsed = time.time() - t0

        if result.get("error"):
            report.check("ForecastService.forecast()", report.FAIL, result["error"])
            return

        report.check("ForecastService.forecast()", report.PASS,
                     f"Completed in {elapsed:.1f}s")
        report.check("  zone", report.PASS, result["zone"])
        report.check("  current_price", report.PASS, f"€{result['current_price']}")
        report.check("  confidence", report.PASS, f"{result['confidence']}%")
        report.check("  models_used", report.PASS,
                     f"{result['models_used']}/{result['models_total']}")
        report.check("  forecasts", report.PASS,
                     f"{len(result['forecasts'])} hourly predictions")
        report.check("  per_model", report.PASS,
                     ", ".join(f"{k}=€{v}" for k, v in result["per_model"].items()))

        # Validate response schema
        required_keys = ["zone", "hours", "forecasts", "current_price",
                         "confidence", "models_used", "models_total",
                         "per_model", "generated_at"]
        missing = [k for k in required_keys if k not in result]
        if missing:
            report.check("  Response schema", report.WARN, f"Missing keys: {missing}")
        else:
            report.check("  Response schema", report.PASS, "All required keys present")

    except Exception as e:
        report.check("ForecastService.forecast()", report.FAIL, traceback.format_exc()[-400:])


# ═════════════════════════════════════════════════════════════════════
# 11. API ENDPOINT SMOKE TESTS
# ═════════════════════════════════════════════════════════════════════

def check_api_endpoints():
    report.section("11 · API Endpoint Smoke Tests")

    import urllib.request
    import urllib.error

    # Check if the server is running locally
    base_urls = [
        "http://localhost:8000",
        "http://localhost:8080",
    ]

    # Also check cloud URL from env
    cloud_url = os.environ.get("ENERGYLENS_URL", "")
    if cloud_url:
        base_urls.append(cloud_url.rstrip("/"))

    active_base = None
    for base in base_urls:
        try:
            req = urllib.request.Request(f"{base}/api/health", method="GET")
            urllib.request.urlopen(req, timeout=5)
            active_base = base
            break
        except Exception:
            continue

    if not active_base:
        report.check("API server", report.SKIP,
                     "No running server found on :8000/:8080.\n"
                     "       Start with: uvicorn api.main:app --port 8000\n"
                     "       Or set ENERGYLENS_URL env var for Cloud Run.")
        return

    report.check("API server", report.PASS, f"Reachable at {active_base}")

    # Test endpoints
    endpoints = [
        ("/api/health", "Health check"),
        ("/api/prices?zone=DK1&days=2", "Spot prices DK1"),
        ("/api/prices/latest", "Latest prices"),
        ("/api/prices/compare?days=2", "Zone comparison"),
        ("/api/forecast?zone=DK1&hours=24", "Forecast DK1"),
        ("/api/forecast/models", "Model status"),
    ]

    for path, desc in endpoints:
        try:
            url = f"{active_base}{path}"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=30)
            code = resp.getcode()
            body = json.loads(resp.read().decode())

            if code == 200:
                # Summarize key info from response
                summary = ""
                if "count" in body:
                    summary = f"{body['count']} records"
                elif "forecasts" in body:
                    summary = f"{len(body['forecasts'])} steps, conf={body.get('confidence', '?')}%"
                elif "status" in body:
                    summary = f"status={body['status']}"
                report.check(f"GET {path}", report.PASS, f"200 OK — {summary}")
            else:
                report.check(f"GET {path}", report.WARN, f"HTTP {code}")

        except urllib.error.HTTPError as e:
            if e.code == 503:
                report.check(f"GET {path}", report.WARN, f"503 — {desc} unavailable (models not loaded?)")
            else:
                report.check(f"GET {path}", report.FAIL, f"HTTP {e.code}: {e.reason}")
        except Exception as e:
            report.check(f"GET {path}", report.FAIL, str(e)[:150])


# ═════════════════════════════════════════════════════════════════════
# 12. FRONTEND BUILD CHECK
# ═════════════════════════════════════════════════════════════════════

def check_frontend():
    report.section("12 · Frontend Build")

    # Check for frontend source
    frontend_dirs = [
        PROJECT_ROOT / "frontend",
        PROJECT_ROOT / "client",
        PROJECT_ROOT / "ui",
    ]
    frontend_dir = None
    for d in frontend_dirs:
        if d.exists() and (d / "package.json").exists():
            frontend_dir = d
            break

    if frontend_dir is None:
        # Check if there's a built static dir
        static_dir = PROJECT_ROOT / "static"
        if static_dir.exists() and (static_dir / "index.html").exists():
            report.check("Frontend build", report.PASS,
                         f"Pre-built at {static_dir}")
            assets = list((static_dir / "assets").glob("*")) if (static_dir / "assets").exists() else []
            report.check("  Static assets", report.PASS, f"{len(assets)} files")
            return
        report.check("Frontend directory", report.SKIP,
                     "No frontend/, client/, or static/ directory found")
        return

    report.check("Frontend directory", report.PASS, str(frontend_dir))

    # Check package.json
    pkg_path = frontend_dir / "package.json"
    try:
        with open(pkg_path) as f:
            pkg = json.load(f)
        report.check("package.json", report.PASS,
                     f"name={pkg.get('name', '?')}, scripts={list(pkg.get('scripts', {}).keys())}")
    except Exception as e:
        report.check("package.json", report.FAIL, str(e))

    # Check if node_modules exist
    if (frontend_dir / "node_modules").exists():
        report.check("node_modules", report.PASS, "Installed")
    else:
        report.check("node_modules", report.WARN, "Not installed — run: npm install")

    # Check for build output
    dist_dirs = [frontend_dir / "dist", frontend_dir / "build"]
    for dd in dist_dirs:
        if dd.exists() and (dd / "index.html").exists():
            report.check("Build output", report.PASS, str(dd))
            return

    report.check("Build output", report.WARN,
                 "No dist/ or build/ found — run: npm run build")

    # Check critical source files
    src_files = ["src/App.jsx", "src/main.jsx", "src/ForecastChart.jsx"]
    for sf in src_files:
        fp = frontend_dir / sf
        if fp.exists():
            report.check(f"  {sf}", report.PASS, f"{fp.stat().st_size} bytes")
        else:
            report.check(f"  {sf}", report.WARN, "Not found")


# ═════════════════════════════════════════════════════════════════════
# 13. CROSS-MODULE SYNCHRONIZATION
# ═════════════════════════════════════════════════════════════════════

def check_synchronization(models, config):
    report.section("13 · Cross-Module Synchronization")

    if not config:
        report.check("Sync checks", report.SKIP, "No model config available")
        return

    used_features = config.get("used_features", [])

    # Check: features from build_energy_features match model expectations
    try:
        from ml.features import build_energy_features
        import pandas as pd

        db_path = PROJECT_ROOT / "data" / "energylens.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            df = pd.read_sql_query(
                """SELECT valid_time AS HourUTC, price_eur_mwh AS SpotPriceEUR,
                          price_dkk_mwh AS SpotPriceDKK
                   FROM spot_prices WHERE zone = 'DK1'
                   ORDER BY valid_time DESC LIMIT 500""",
                conn, parse_dates=["HourUTC"],
            )
            conn.close()

            if not df.empty:
                df = df.sort_values("HourUTC").set_index("HourUTC").resample("h").mean().dropna()
                featured = build_energy_features(df)

                available = set(featured.columns)
                expected = set(used_features)

                matched = available & expected
                missing_from_data = expected - available
                extra_in_data = available - expected

                report.check("Feature alignment", report.PASS if not missing_from_data else report.WARN,
                             f"{len(matched)}/{len(expected)} expected features found in data")

                if missing_from_data:
                    report.check("  Missing from data", report.WARN,
                                 f"{len(missing_from_data)} features: "
                                 f"{sorted(list(missing_from_data))[:10]}")
                    report.check("  (auto-padded with 0)", report.PASS,
                                 "forecast_service.py pads missing features")
    except Exception as e:
        report.check("Feature alignment", report.WARN, str(e)[:200])

    # Check: ensemble weights match loaded models
    try:
        from ml.ensemble import DEFAULT_WEIGHTS
        weight_names = set(DEFAULT_WEIGHTS.keys())
        model_names = set(models.keys()) if models else set()

        if model_names:
            missing_weights = model_names - weight_names
            extra_weights = weight_names - model_names

            if not missing_weights:
                report.check("Ensemble weights sync", report.PASS,
                             f"All {len(model_names)} loaded models have weights")
            else:
                report.check("Ensemble weights sync", report.WARN,
                             f"Models without explicit weights (will use 1/N): {missing_weights}")

            if extra_weights:
                report.check("  Unused weight entries", report.PASS,
                             f"{extra_weights} — harmless, weights for unloaded models")
    except Exception as e:
        report.check("Ensemble weights sync", report.WARN, str(e)[:200])

    # Check: scaler n_features matches used_features count
    scaler = config.get("scaler")
    if scaler is not None and used_features:
        scaler_n = scaler.scale_.shape[0]
        expected_n = len(used_features)
        if scaler_n == expected_n:
            report.check("Scaler/feature count sync", report.PASS,
                         f"Both have {scaler_n} features")
        else:
            report.check("Scaler/feature count sync", report.FAIL,
                         f"MISMATCH — scaler expects {scaler_n} features, "
                         f"used_features has {expected_n}")

    # Check: temporal features in TEMPORAL_FEATURES match feature list
    try:
        from ml.ensemble import TEMPORAL_FEATURES
        temporal_in_model = [f for f in used_features if f in TEMPORAL_FEATURES]
        report.check("Temporal features", report.PASS,
                     f"{len(temporal_in_model)} temporal features in model: "
                     f"{temporal_in_model[:6]}{'...' if len(temporal_in_model) > 6 else ''}")
    except Exception:
        pass

    # Check: Close is feature 0 (ensemble expects this for inverse transform)
    if used_features and used_features[0] == "Close":
        report.check("Close is feature[0]", report.PASS,
                     "Required for inverse_transform_prediction()")
    elif used_features:
        report.check("Close is feature[0]", report.WARN,
                     f"Feature[0] is '{used_features[0]}' — ensemble assumes Close is index 0")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 60)
    print("  ENERGYLENS — COMPREHENSIVE PRE-DEPLOY DIAGNOSTIC")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project root: {PROJECT_ROOT}")
    print("═" * 60)

    # 1–4: Infrastructure
    check_project_structure()
    check_configuration()
    db_counts = check_database()
    check_connectors()

    # 5: Features
    featured_df = check_features()

    # 6: Models
    models, config = check_models()

    # 7–8: Predictions
    check_single_prediction(models, config or {})
    check_multi_step(models, config or {})

    # 9: Safety rails
    check_safety_rails()

    # 10: End-to-end
    check_forecast_service()

    # 11: API
    check_api_endpoints()

    # 12: Frontend
    check_frontend()

    # 13: Sync
    check_synchronization(models, config or {})

    # Final summary
    exit_code = report.finalize()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
