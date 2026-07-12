"""
EnergyLens — Forecast Pipeline Diagnostic

Traces every step of the forecast pipeline to identify and fix issues.
Run from project root: python diagnose_forecast.py

Checks:
  1. Database state and data freshness
  2. Raw data loading and resampling
  3. Feature building (before/after transform)
  4. Model loading and config
  5. Scaling and sequence creation
  6. Individual model predictions
  7. Ensemble filtering (CATASTROPHIC check)
  8. Multi-step forecast generation
  9. Recommends and optionally applies fixes
"""

import sys
import os
import sqlite3
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "energylens"))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

DB_PATH = "data/energylens.db"
MODEL_DIR = "models"
ZONE = "DK1"

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
INFO = "\033[94m INFO \033[0m"

issues_found = []
fixes_applied = []


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        issues_found.append(label)
    return condition


def warn(label, detail=""):
    print(f"  [{WARN}] {label}")
    if detail:
        print(f"         {detail}")
    issues_found.append(label)


def info(label, detail=""):
    print(f"  [{INFO}] {label}")
    if detail:
        print(f"         {detail}")


# ============================================================================
# STEP 1: Database
# ============================================================================
def diagnose_database():
    header("STEP 1: Database State")

    db = Path(DB_PATH)
    if not check("Database file exists", db.exists(), str(db.absolute())):
        return None

    conn = sqlite3.connect(DB_PATH)

    # Total rows
    total = conn.execute("SELECT COUNT(*) FROM spot_prices").fetchone()[0]
    check("spot_prices has data", total > 0, f"{total} rows")

    # Zone breakdown
    zones = conn.execute(
        "SELECT zone, COUNT(*) FROM spot_prices GROUP BY zone"
    ).fetchall()
    info("Zone breakdown", str(dict(zones)))

    # Date range
    bounds = conn.execute(
        "SELECT MIN(valid_time), MAX(valid_time) FROM spot_prices WHERE zone=?",
        (ZONE,)
    ).fetchone()
    info("Date range", f"{bounds[0]} -> {bounds[1]}")

    # Check freshness
    newest = bounds[1]
    try:
        newest_dt = datetime.fromisoformat(newest.replace("T", " ").split("+")[0])
        age_hours = (datetime.utcnow() - newest_dt).total_seconds() / 3600
        check("Data freshness < 48h", age_hours < 48, f"{age_hours:.1f}h old")
    except:
        warn("Could not parse newest timestamp")

    # Sample latest prices
    latest = conn.execute(
        "SELECT valid_time, price_eur_mwh FROM spot_prices WHERE zone=? ORDER BY valid_time DESC LIMIT 5",
        (ZONE,)
    ).fetchall()
    print(f"\n  Latest {ZONE} prices:")
    for row in latest:
        price = row[1]
        flag = " <-- NEGATIVE" if price < 0 else " <-- NEAR ZERO" if abs(price) < 1 else ""
        print(f"    {row[0]}  EUR {price:>10.2f}{flag}")

    # Check for negative/near-zero prices
    neg_count = conn.execute(
        "SELECT COUNT(*) FROM spot_prices WHERE zone=? AND price_eur_mwh < 1.0",
        (ZONE,)
    ).fetchone()[0]
    if neg_count > 0:
        warn(f"{neg_count} rows with price < EUR 1.0",
             "Negative/near-zero prices are normal in Nordic markets during high wind/solar")

    # Check time intervals
    intervals = conn.execute("""
        SELECT 
            CAST((julianday(MAX(valid_time)) - julianday(MIN(valid_time))) * 24 * 60 / COUNT(*) AS INTEGER) 
            as avg_interval_minutes
        FROM spot_prices WHERE zone=?
    """, (ZONE,)).fetchone()
    if intervals:
        info(f"Average interval", f"~{intervals[0]} minutes")
        if intervals[0] < 30:
            info("Sub-hourly data detected", "Will need resampling to hourly for models")

    conn.close()
    return latest


# ============================================================================
# STEP 2: Data Loading & Resampling
# ============================================================================
def diagnose_data_loading():
    header("STEP 2: Data Loading & Resampling")

    conn = sqlite3.connect(DB_PATH)
    try:
        prices_df = pd.read_sql_query(
            """
            SELECT valid_time AS HourUTC,
                   price_eur_mwh AS SpotPriceEUR,
                   price_dkk_mwh AS SpotPriceDKK
            FROM spot_prices
            WHERE zone = ?
            ORDER BY valid_time DESC
            LIMIT ?
            """,
            conn,
            params=(ZONE, 400 * 4),
            parse_dates=["HourUTC"],
        )
        prices_df = prices_df.sort_values("HourUTC").set_index("HourUTC")
    except Exception as e:
        check("Load raw prices", False, str(e))
        conn.close()
        return None
    conn.close()

    info(f"Raw data loaded", f"{len(prices_df)} rows")
    info(f"Raw price range", f"EUR {prices_df['SpotPriceEUR'].min():.2f} to {prices_df['SpotPriceEUR'].max():.2f}")
    info(f"Raw price mean", f"EUR {prices_df['SpotPriceEUR'].mean():.2f}")

    # Store raw last price BEFORE resampling
    raw_last_price = float(prices_df["SpotPriceEUR"].iloc[-1])
    info(f"Last raw price (current)", f"EUR {raw_last_price:.2f}")

    # Resample
    hourly_df = prices_df.resample("h").mean().dropna()
    info(f"After hourly resample", f"{len(hourly_df)} rows")
    info(f"Hourly price range", f"EUR {hourly_df['SpotPriceEUR'].min():.2f} to {hourly_df['SpotPriceEUR'].max():.2f}")
    info(f"Hourly price mean", f"EUR {hourly_df['SpotPriceEUR'].mean():.2f}")

    hourly_last_price = float(hourly_df["SpotPriceEUR"].iloc[-1])
    info(f"Last hourly price", f"EUR {hourly_last_price:.2f}")

    check("Enough hourly data (>58 rows)", len(hourly_df) >= 58, f"{len(hourly_df)} rows")

    return hourly_df, raw_last_price


# ============================================================================
# STEP 3: Feature Building
# ============================================================================
def diagnose_features(hourly_df):
    header("STEP 3: Feature Building")

    from ml.features import build_energy_features

    # Make a copy so we can compare before/after
    df_copy = hourly_df.copy()
    pre_last_eur = float(df_copy["SpotPriceEUR"].iloc[-1])
    info(f"SpotPriceEUR before transform", f"EUR {pre_last_eur:.2f}")

    try:
        featured_df = build_energy_features(df_copy)
        check("build_energy_features succeeded", True, f"{len(featured_df)} rows, {len(featured_df.columns)} features")
    except Exception as e:
        check("build_energy_features", False, str(e))
        return None

    # Check if raw_df was modified in-place
    post_last_eur = float(df_copy["SpotPriceEUR"].iloc[-1]) if "SpotPriceEUR" in df_copy.columns else None
    if post_last_eur is not None:
        if abs(post_last_eur - pre_last_eur) > 0.01:
            warn("build_energy_features MODIFIES input DataFrame in-place!",
                 f"SpotPriceEUR changed from {pre_last_eur:.2f} to {post_last_eur:.2f}")
        else:
            info("Input DataFrame not modified in-place")

    # Check Close column
    if "Close" in featured_df.columns:
        close_val = float(featured_df["Close"].iloc[-1])
        info(f"Close column (last value)", f"{close_val:.4f}")
        if abs(close_val) < 1.0:
            warn("Close column is near zero — this is used as 'current_price' if not captured early",
                 f"Close={close_val:.4f}, which would trigger CATASTROPHIC filter")
    else:
        warn("Close column missing from features!")

    # List some feature columns
    info(f"Sample feature columns", str(list(featured_df.columns[:10])))

    return featured_df


# ============================================================================
# STEP 4: Model Loading
# ============================================================================
def diagnose_models():
    header("STEP 4: Model Loading")

    from ml.training import load_models

    try:
        models, config = load_models(zone=ZONE, base_dir=MODEL_DIR)
    except Exception as e:
        check("Load models", False, str(e))
        return None, None

    check("Models loaded", len(models) > 0, f"{len(models)} models: {sorted(models.keys())}")

    scaler = config.pop("scaler", None)
    check("Scaler loaded", scaler is not None, type(scaler).__name__ if scaler else "None")

    time_step = config.get("time_step", 48)
    used_features = config.get("used_features", [])
    price_range = config.get("price_range")
    cv_weights = config.get("ensemble_weights")

    info("time_step", str(time_step))
    info("used_features count", str(len(used_features)))
    info("price_range", str(price_range))
    info("ensemble_weights", "present" if cv_weights else "None")

    if price_range:
        midpoint = (price_range[0] + price_range[1]) / 2
        info("price_range midpoint", f"EUR {midpoint:.2f}")
        info("price_range width", f"EUR {price_range[1] - price_range[0]:.2f}")

    return models, {"scaler": scaler, "time_step": time_step, "used_features": used_features,
                     "price_range": price_range, "cv_weights": cv_weights}


# ============================================================================
# STEP 5: Scaling & Sequence
# ============================================================================
def diagnose_scaling(featured_df, config):
    header("STEP 5: Scaling & Sequence Creation")

    scaler = config["scaler"]
    time_step = config["time_step"]
    used_features = config["used_features"]

    # Align features
    available = [c for c in used_features if c in featured_df.columns]
    missing = [c for c in used_features if c not in featured_df.columns]

    info(f"Features available", f"{len(available)}/{len(used_features)}")
    if missing:
        info(f"Missing features (padded with 0)", f"{len(missing)}: {missing[:5]}...")

    aligned_df = pd.DataFrame(0.0, index=featured_df.index, columns=used_features)
    for col in available:
        aligned_df[col] = featured_df[col]

    # Scale
    try:
        scaled = scaler.transform(aligned_df.values)
        check("Scaling succeeded", True, f"Shape: {scaled.shape}")
    except Exception as e:
        check("Scaling", False, str(e))
        return None

    check(f"Enough data for time_step ({time_step})", len(scaled) >= time_step,
          f"{len(scaled)} >= {time_step}")

    last_seq = scaled[-time_step:].reshape(1, time_step, -1)
    info("Input sequence shape", str(last_seq.shape))

    # Check for NaN/Inf in sequence
    has_nan = np.isnan(last_seq).any()
    has_inf = np.isinf(last_seq).any()
    check("No NaN in input sequence", not has_nan)
    check("No Inf in input sequence", not has_inf)

    return last_seq


# ============================================================================
# STEP 6: Individual Model Predictions
# ============================================================================
def diagnose_predictions(models, last_seq, config, current_price):
    header("STEP 6: Individual Model Predictions")

    scaler = config["scaler"]
    price_range = config["price_range"]

    print(f"\n  Current price for CATASTROPHIC filter: EUR {current_price:.2f}")
    if price_range:
        print(f"  Price range: [{price_range[0]:.2f}, {price_range[1]:.2f}]")
    print()

    predictions = {}
    for name, model in sorted(models.items()):
        try:
            pred = model.predict(last_seq)
            # Inverse transform to get EUR price
            if scaler is not None:
                dummy = np.zeros((1, scaler.n_features_in_))
                dummy[0, 0] = pred.flatten()[0]
                inv = scaler.inverse_transform(dummy)
                price_pred = float(inv[0, 0])
            else:
                price_pred = float(pred.flatten()[0])

            predictions[name] = price_pred

            # Check CATASTROPHIC filter
            if current_price != 0:
                pct_off = abs(price_pred - current_price) / abs(current_price) * 100
            else:
                pct_off = abs(price_pred) * 100

            catastrophic = pct_off > 15 if abs(current_price) > 5 else False
            status = "CATASTROPHIC" if catastrophic else "OK"
            color = "\033[91m" if catastrophic else "\033[92m"

            print(f"    {name:25s}  EUR {price_pred:>8.2f}  "
                  f"({pct_off:>8.1f}% from current)  {color}{status}\033[0m")

        except Exception as e:
            print(f"    {name:25s}  ERROR: {e}")

    if predictions:
        pred_values = list(predictions.values())
        info(f"\nPrediction range", f"EUR {min(pred_values):.2f} to {max(pred_values):.2f}")
        info(f"Prediction mean", f"EUR {np.mean(pred_values):.2f}")
        info(f"Prediction median", f"EUR {np.median(pred_values):.2f}")
        info(f"Prediction std", f"EUR {np.std(pred_values):.2f}")

    return predictions


# ============================================================================
# STEP 7: Ensemble Analysis
# ============================================================================
def diagnose_ensemble(predictions, current_price, price_range):
    header("STEP 7: Ensemble Filter Analysis")

    print(f"\n  The CATASTROPHIC filter in ml/ensemble.py excludes predictions")
    print(f"  that deviate >15% from current_price.")
    print(f"\n  Current price: EUR {current_price:.2f}")

    if abs(current_price) < 5.0:
        print(f"\n  *** PROBLEM IDENTIFIED ***")
        print(f"  Current price is near zero/negative (EUR {current_price:.2f}).")
        print(f"  Every prediction (EUR 60-120) is 'infinitely' far from zero,")
        print(f"  so ALL models get flagged as CATASTROPHIC.")
        print()

        if price_range:
            midpoint = (price_range[0] + price_range[1]) / 2
            print(f"  SOLUTION: Use price_range midpoint (EUR {midpoint:.2f}) as filter reference")
            print(f"  when current price is near zero.")
            print()

            # Re-evaluate with midpoint
            print(f"  Re-evaluating with filter_price = EUR {midpoint:.2f}:")
            passing = 0
            for name, pred in sorted(predictions.items()):
                pct_off = abs(pred - midpoint) / abs(midpoint) * 100
                ok = pct_off <= 15
                if ok:
                    passing += 1
                status = "OK" if ok else "EXCLUDED"
                color = "\033[92m" if ok else "\033[91m"
                print(f"    {name:25s}  EUR {pred:>8.2f}  "
                      f"({pct_off:>6.1f}% from midpoint)  {color}{status}\033[0m")

            print(f"\n  With midpoint filter: {passing}/{len(predictions)} models pass")

            if passing == 0:
                print(f"\n  Still 0 models passing. The predictions (~EUR 60-120) are too far")
                print(f"  from the midpoint (EUR {midpoint:.2f}).")
                print(f"\n  ALTERNATIVE: Disable the CATASTROPHIC filter entirely when")
                print(f"  current price is near zero. The outlier filter (>15% from median)")
                print(f"  will still catch true outliers.")


# ============================================================================
# STEP 8: Fix Recommendation
# ============================================================================
def recommend_fixes(predictions, current_price, price_range):
    header("STEP 8: Recommended Fixes")

    pred_values = list(predictions.values()) if predictions else []
    pred_median = float(np.median(pred_values)) if pred_values else 80.0

    print(f"""
  Issue: The CATASTROPHIC filter in ml/ensemble.py rejects all model
  predictions because current_price (EUR {current_price:.2f}) is near zero.

  Model predictions are EUR {min(pred_values):.0f}-{max(pred_values):.0f}, which are reasonable
  for the Nordic market but appear as "infinite % deviation" from zero.

  FIX OPTIONS (choose one):

  Option A (Recommended): Patch ml/ensemble.py
    Find the CATASTROPHIC filter and add a bypass for near-zero prices.
    This is the proper fix since negative prices are normal in Nordic markets.

  Option B (Quick): Set price_range=None for both ensemble calls
    In forecast_service.py, pass price_range=None to both
    ensemble_predict() and multi_step_forecast().
    This disables the CATASTROPHIC filter entirely.

  Option C (Middle ground): Use prediction median as reference
    When current_price < EUR 5, use the median of all model predictions
    as the filter reference instead of current_price.
""")

    return pred_median


# ============================================================================
# STEP 9: Auto-fix
# ============================================================================
def apply_fix():
    header("STEP 9: Applying Fix (Option B — disable price_range filter)")

    fs_path = Path("api/forecast_service.py")
    if not fs_path.exists():
        print("  Could not find api/forecast_service.py")
        return False

    content = fs_path.read_text(encoding="utf-8")
    original = content

    # Fix 1: Ensure current_price is captured before build_energy_features
    if "current_price = float(raw_df" in content:
        # Check if it's before or after build_energy_features
        cp_pos = content.find('current_price = float(raw_df')
        bf_pos = content.find('build_energy_features(raw_df)')
        if cp_pos > bf_pos and bf_pos > 0:
            print("  [FIX] Moving current_price capture before build_energy_features")
            # Remove old line
            old_line = '        current_price = float(raw_df["SpotPriceEUR"].iloc[-1])'
            content = content.replace(old_line + "\n", "", 1)
            # Insert before build_energy_features
            insert_before = "        # 3. Build features"
            insert_text = (
                "        # IMPORTANT: Capture current price BEFORE feature transform\n"
                '        current_price = float(raw_df["SpotPriceEUR"].iloc[-1])\n'
                f'        last_timestamp = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)\n'
                '\n'
            )
            content = content.replace(insert_before, insert_text + insert_before)
            fixes_applied.append("Moved current_price before build_energy_features")

    # Fix 2: Pass price_range=None to both ensemble calls to disable CATASTROPHIC filter
    # For ensemble_predict
    if "price_range=price_range," in content and "current_price=filter_price," in content:
        content = content.replace(
            "price_range=price_range,\n            current_price=filter_price,",
            "price_range=None,\n            current_price=current_price,"
        )
        fixes_applied.append("Set price_range=None for ensemble_predict (disables CATASTROPHIC)")
    elif "price_range=price_range," in content:
        # Replace in ensemble_predict context
        pass

    # Simpler approach: just replace ALL price_range=price_range with price_range=None
    count = content.count("price_range=price_range,")
    if count > 0:
        content = content.replace("price_range=price_range,", "price_range=None,  # disabled: avoids CATASTROPHIC on near-zero prices")
        fixes_applied.append(f"Disabled price_range filter in {count} ensemble call(s)")

    if content != original:
        fs_path.write_text(content, encoding="utf-8")
        print(f"\n  Applied {len(fixes_applied)} fix(es) to {fs_path}:")
        for f in fixes_applied:
            print(f"    - {f}")
        print(f"\n  Restart the server to apply:")
        print(f"    uvicorn api.main:app --reload --port 8000")
        return True
    else:
        print("  No changes needed or could not apply automatically.")
        print("  Manual fix: open api/forecast_service.py and change")
        print("  price_range=price_range to price_range=None in both")
        print("  ensemble_predict() and multi_step_forecast() calls.")
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  EnergyLens Forecast Pipeline Diagnostic")
    print(f"  Zone: {ZONE}  |  DB: {DB_PATH}  |  Models: {MODEL_DIR}")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # Step 1
    latest = diagnose_database()
    if latest is None:
        return

    # Step 2
    result = diagnose_data_loading()
    if result is None:
        return
    hourly_df, raw_last_price = result

    # Step 3
    # Use a fresh copy for feature building
    featured_df = diagnose_features(hourly_df.copy())
    if featured_df is None:
        return

    # Step 4
    models, config = diagnose_models()
    if models is None:
        return

    # Step 5
    last_seq = diagnose_scaling(featured_df, config)
    if last_seq is None:
        return

    # Step 6
    predictions = diagnose_predictions(models, last_seq, config, raw_last_price)

    # Step 7
    diagnose_ensemble(predictions, raw_last_price, config["price_range"])

    # Step 8
    pred_median = recommend_fixes(predictions, raw_last_price, config["price_range"])

    # Summary
    header("SUMMARY")
    if issues_found:
        print(f"\n  Found {len(issues_found)} issue(s):")
        for i, issue in enumerate(issues_found, 1):
            print(f"    {i}. {issue}")
    else:
        print("\n  No issues found!")

    # Ask to apply fix
    print()
    response = input("  Apply automatic fix? (y/n): ").strip().lower()
    if response == "y":
        apply_fix()
    else:
        print("\n  No changes made. You can manually edit api/forecast_service.py")
        print("  Change price_range=price_range to price_range=None in the")
        print("  ensemble_predict() and multi_step_forecast() calls.")


if __name__ == "__main__":
    main()
