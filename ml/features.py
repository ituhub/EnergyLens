"""
EnergyLens Phase 2 — Energy-specific feature engineering.

Replaces MarketLens financial indicators (RSI, MACD, Bollinger Bands, etc.)
with features relevant to Nordic power markets:
  • Temporal patterns (hour-of-day, day-of-week, month — cyclical encoding)
  • Weather covariates (temperature, wind speed, solar radiation)
  • Lag features (1h, 24h, 168h for weekly seasonality)
  • Rolling statistics (mean, std, min, max over multiple windows)
  • Price-derived features (returns, volatility, momentum)
  • Demand/supply indicators (load forecast, renewable generation)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── Cyclical encoding ────────────────────────────────────────────────────────

def cyclical_encode(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a periodic variable as sin/cos pair."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


# ── Core feature builder ─────────────────────────────────────────────────────

def build_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive feature set from raw energy data.

    Expected columns (from Phase 1 pipeline):
        SpotPriceDKK, SpotPriceEUR   – Nord Pool spot prices
        temperature_2m, wind_speed_10m, shortwave_radiation  – weather
        load_forecast, renewable_generation  – ENTSO-E (optional)

    The output is a DataFrame with all original columns plus engineered features.
    Close is an alias for SpotPriceEUR (the prediction target).
    """
    feat = df.copy()

    # ── Determine the price target column ────────────────────────────
    price_col = None
    for candidate in ["SpotPriceEUR", "SpotPriceDKK", "price_eur", "Close"]:
        if candidate in feat.columns:
            price_col = candidate
            break

    if price_col is None:
        logger.error("No price column found in data")
        return feat

    # Alias to Close for model compatibility
    if price_col != "Close":
        feat["Close"] = feat[price_col]

    prices = feat["Close"]

    # ── 1. Temporal features ─────────────────────────────────────────
    if hasattr(feat.index, "hour"):
        idx = feat.index
    else:
        try:
            idx = pd.to_datetime(feat.index)
            feat.index = idx
        except Exception:
            logger.warning("Could not parse index as datetime; skipping temporal features")
            idx = None

    if idx is not None:
        feat["hour"] = idx.hour
        feat["hour_sin"], feat["hour_cos"] = cyclical_encode(feat["hour"], 24)
        feat["dow"] = idx.dayofweek
        feat["dow_sin"], feat["dow_cos"] = cyclical_encode(feat["dow"], 7)
        feat["month"] = idx.month
        feat["month_sin"], feat["month_cos"] = cyclical_encode(feat["month"], 12)
        feat["quarter"] = idx.quarter
        feat["is_weekend"] = (feat["dow"] >= 5).astype(int)

        # Business hour flag (07:00-19:00 CET is peak in Nordics)
        feat["is_peak_hour"] = ((feat["hour"] >= 7) & (feat["hour"] <= 19)).astype(int)

    # ── 2. Price lag features ────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        feat[f"price_lag_{lag}"] = prices.shift(lag)

    # ── 3. Price returns & momentum ──────────────────────────────────
    feat["return_1h"] = prices.pct_change(1)
    feat["return_24h"] = prices.pct_change(24)
    feat["return_168h"] = prices.pct_change(168)

    feat["momentum_6h"] = prices.diff(6)
    feat["momentum_24h"] = prices.diff(24)

    # ── 4. Rolling statistics ────────────────────────────────────────
    for window in [6, 12, 24, 48, 168]:
        roll = prices.rolling(window, min_periods=1)
        feat[f"price_mean_{window}"] = roll.mean()
        feat[f"price_std_{window}"] = roll.std()
        feat[f"price_min_{window}"] = roll.min()
        feat[f"price_max_{window}"] = roll.max()

        # Relative position within rolling range (like Bollinger %B)
        price_range = feat[f"price_max_{window}"] - feat[f"price_min_{window}"]
        feat[f"price_position_{window}"] = (
            (prices - feat[f"price_min_{window}"]) / (price_range + 1e-8)
        )

        # Price distance from rolling mean (z-score)
        feat[f"price_zscore_{window}"] = (
            (prices - feat[f"price_mean_{window}"]) / (feat[f"price_std_{window}"] + 1e-8)
        )

    # ── 5. Volatility features ───────────────────────────────────────
    returns = prices.pct_change().fillna(0)
    feat["volatility_6h"] = returns.rolling(6, min_periods=1).std()
    feat["volatility_24h"] = returns.rolling(24, min_periods=1).std()
    feat["volatility_168h"] = returns.rolling(168, min_periods=1).std()

    # Realized volatility ratio (short vs long)
    feat["vol_ratio_6_24"] = feat["volatility_6h"] / (feat["volatility_24h"] + 1e-8)
    feat["vol_ratio_24_168"] = feat["volatility_24h"] / (feat["volatility_168h"] + 1e-8)

    # Return skewness and kurtosis
    feat["return_skew_24"] = returns.rolling(24, min_periods=6).skew()
    feat["return_kurt_24"] = returns.rolling(24, min_periods=6).kurt()

    # ── 6. Weather features (if available) ───────────────────────────
    weather_cols = {
        "temperature_2m": "temp",
        "wind_speed_10m": "wind",
        "shortwave_radiation": "solar",
    }

    for raw_col, short in weather_cols.items():
        if raw_col not in feat.columns:
            continue

        w = feat[raw_col]
        # Lags
        feat[f"{short}_lag_1"] = w.shift(1)
        feat[f"{short}_lag_24"] = w.shift(24)

        # Rolling stats
        for win in [6, 24]:
            feat[f"{short}_mean_{win}"] = w.rolling(win, min_periods=1).mean()
            feat[f"{short}_std_{win}"] = w.rolling(win, min_periods=1).std()

        # Change features
        feat[f"{short}_change_1h"] = w.diff(1)
        feat[f"{short}_change_24h"] = w.diff(24)

    # Wind power proxy (wind^3 approximates power output)
    if "wind_speed_10m" in feat.columns:
        feat["wind_power_proxy"] = feat["wind_speed_10m"] ** 3

    # Heating degree hours (below 18°C → more demand)
    if "temperature_2m" in feat.columns:
        feat["heating_degree"] = np.maximum(0, 18.0 - feat["temperature_2m"])
        feat["cooling_degree"] = np.maximum(0, feat["temperature_2m"] - 24.0)

    # ── 7. Demand / supply features (if available) ───────────────────
    if "load_forecast" in feat.columns:
        load = feat["load_forecast"]
        feat["load_lag_24"] = load.shift(24)
        feat["load_change_24h"] = load.diff(24)
        feat["load_mean_24"] = load.rolling(24, min_periods=1).mean()

        # Price per unit load (merit-order proxy)
        feat["price_per_load"] = prices / (load + 1e-8)

    if "renewable_generation" in feat.columns:
        ren = feat["renewable_generation"]
        feat["renewable_share"] = ren / (feat.get("load_forecast", ren) + 1e-8)
        feat["renewable_lag_24"] = ren.shift(24)

    # ── 8. Price spike indicator ─────────────────────────────────────
    # Useful for the model to learn extreme events
    rolling_mean_24 = prices.rolling(24, min_periods=1).mean()
    rolling_std_24 = prices.rolling(24, min_periods=1).std()
    feat["spike_indicator"] = (
        (prices > rolling_mean_24 + 2 * rolling_std_24) |
        (prices < rolling_mean_24 - 2 * rolling_std_24)
    ).astype(int)

    # ── 9. Cross-feature interactions ────────────────────────────────
    if "temperature_2m" in feat.columns and "is_peak_hour" in feat.columns:
        feat["temp_peak_interaction"] = feat["temperature_2m"] * feat["is_peak_hour"]

    if "wind_speed_10m" in feat.columns:
        feat["wind_price_interaction"] = feat["wind_speed_10m"] * feat["return_1h"].fillna(0)

    # ── Cleanup ──────────────────────────────────────────────────────
    # Forward-fill, backward-fill, then zero for any remaining NaN
    feat = feat.ffill().bfill().fillna(0)

    # Keep only numeric columns
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    feat = feat[numeric_cols]

    # Replace infinities
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Built {len(feat.columns)} features from {len(feat)} rows")
    return feat


def get_energy_price_range(zone: str = "DK1") -> tuple[float, float]:
    """
    Return reasonable EUR/MWh price range for a Nordic bidding zone.
    These are wide fallback ranges — dynamic ranges from training data are preferred.
    """
    ranges = {
        "DK1": (-50, 500),    # Denmark West (can go negative with wind oversupply)
        "DK2": (-50, 500),    # Denmark East
        "NO1": (-20, 400),    # Norway South
        "NO2": (-20, 400),
        "SE1": (-30, 300),    # Sweden North
        "SE3": (-30, 450),    # Sweden South
        "FI":  (-30, 400),    # Finland
    }
    return ranges.get(zone, (-50, 500))


def get_max_step_change(zone: str = "DK1") -> float:
    """
    Maximum reasonable per-step forecast change as a fraction.
    Energy prices are more volatile than most financial assets
    (can go from 30 to 300 EUR/MWh in hours during supply crises).
    """
    # Energy prices can swing wildly, but consecutive hourly steps
    # rarely exceed 50% change from base under normal conditions.
    return 0.50
