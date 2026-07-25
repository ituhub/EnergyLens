"""
EnergyLens — Energy-specific feature engineering.

Produces exactly 97 features matching the Kaggle-trained models.
Close is an alias for SpotPriceEUR (prediction target, index 0).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def build_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 97 features from raw energy + weather data.

    Expected input columns:
        SpotPriceEUR  — Nord Pool spot price
        temperature_2m, wind_speed_10m, shortwave_radiation  — weather (optional)

    Returns DataFrame with Close at index 0 plus 96 other features.
    """
    feat = df.copy()

    # ── Price target: Close = SpotPriceEUR ──
    price_col = None
    for c in ["SpotPriceEUR", "SpotPriceDKK", "price_eur", "Close"]:
        if c in feat.columns:
            price_col = c
            break
    if price_col is None:
        logger.error("No price column found")
        return feat
    if price_col != "Close":
        feat["Close"] = feat[price_col]
    prices = feat["Close"]

    # ── Temporal features ──
    if hasattr(feat.index, 'hour'):
        h  = feat.index.hour
        dw = feat.index.dayofweek
        mo = feat.index.month

        feat['hour']     = h
        feat['hour_sin'] = np.sin(2 * np.pi * h / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * h / 24)
        feat['dow']      = dw
        feat['dow_sin']  = np.sin(2 * np.pi * dw / 7)
        feat['dow_cos']  = np.cos(2 * np.pi * dw / 7)
        feat['month']    = mo
        feat['month_sin'] = np.sin(2 * np.pi * mo / 12)
        feat['month_cos'] = np.cos(2 * np.pi * mo / 12)
        feat['quarter']    = feat.index.quarter
        feat['is_weekend'] = (dw >= 5).astype(float)
        feat['is_peak_hour'] = ((h >= 8) & (h <= 20)).astype(float)

    # ── Price lags ──
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        feat[f'price_lag_{lag}'] = prices.shift(lag)

    # ── Returns ──
    feat['return_1h']   = prices.pct_change(1)
    feat['return_24h']  = prices.pct_change(24)
    feat['return_168h'] = prices.pct_change(168)

    # ── Momentum ──
    feat['momentum_6h']  = prices - prices.shift(6)
    feat['momentum_24h'] = prices - prices.shift(24)

    # ── Rolling statistics ──
    for w in [6, 12, 24, 48, 168]:
        rm = prices.rolling(w).mean()
        rs = prices.rolling(w).std()
        rmin = prices.rolling(w).min()
        rmax = prices.rolling(w).max()

        feat[f'price_mean_{w}']     = rm
        feat[f'price_std_{w}']      = rs
        feat[f'price_min_{w}']      = rmin
        feat[f'price_max_{w}']      = rmax
        feat[f'price_position_{w}'] = (prices - rmin) / (rmax - rmin + 1e-8)
        feat[f'price_zscore_{w}']   = (prices - rm) / (rs + 1e-8)

    # ── Volatility ──
    ret = prices.pct_change()
    feat['volatility_6h']   = ret.rolling(6).std()
    feat['volatility_24h']  = ret.rolling(24).std()
    feat['volatility_168h'] = ret.rolling(168).std()
    feat['vol_ratio_6_24']  = feat['volatility_6h'] / (feat['volatility_24h'] + 1e-8)
    feat['vol_ratio_24_168'] = feat['volatility_24h'] / (feat['volatility_168h'] + 1e-8)

    # ── Return distribution ──
    feat['return_skew_24'] = ret.rolling(24).skew()
    feat['return_kurt_24'] = ret.rolling(24).kurt()

    # ── Weather features: temperature ──
    if 'temperature_2m' in feat.columns:
        t = feat['temperature_2m']
        feat['temp_lag_1']      = t.shift(1)
        feat['temp_lag_24']     = t.shift(24)
        feat['temp_mean_6']     = t.rolling(6).mean()
        feat['temp_std_6']      = t.rolling(6).std()
        feat['temp_mean_24']    = t.rolling(24).mean()
        feat['temp_std_24']     = t.rolling(24).std()
        feat['temp_change_1h']  = t.diff(1)
        feat['temp_change_24h'] = t.diff(24)
    else:
        for c in ['temp_lag_1','temp_lag_24','temp_mean_6','temp_std_6',
                   'temp_mean_24','temp_std_24','temp_change_1h','temp_change_24h']:
            feat[c] = 0.0

    # ── Weather features: wind ──
    if 'wind_speed_10m' in feat.columns:
        w = feat['wind_speed_10m']
        feat['wind_lag_1']      = w.shift(1)
        feat['wind_lag_24']     = w.shift(24)
        feat['wind_mean_6']     = w.rolling(6).mean()
        feat['wind_std_6']      = w.rolling(6).std()
        feat['wind_mean_24']    = w.rolling(24).mean()
        feat['wind_std_24']     = w.rolling(24).std()
        feat['wind_change_1h']  = w.diff(1)
        feat['wind_change_24h'] = w.diff(24)
    else:
        for c in ['wind_lag_1','wind_lag_24','wind_mean_6','wind_std_6',
                   'wind_mean_24','wind_std_24','wind_change_1h','wind_change_24h']:
            feat[c] = 0.0

    # ── Weather features: solar ──
    if 'shortwave_radiation' in feat.columns:
        s = feat['shortwave_radiation']
        feat['solar_lag_1']      = s.shift(1)
        feat['solar_lag_24']     = s.shift(24)
        feat['solar_mean_6']     = s.rolling(6).mean()
        feat['solar_std_6']      = s.rolling(6).std()
        feat['solar_mean_24']    = s.rolling(24).mean()
        feat['solar_std_24']     = s.rolling(24).std()
        feat['solar_change_1h']  = s.diff(1)
        feat['solar_change_24h'] = s.diff(24)
    else:
        for c in ['solar_lag_1','solar_lag_24','solar_mean_6','solar_std_6',
                   'solar_mean_24','solar_std_24','solar_change_1h','solar_change_24h']:
            feat[c] = 0.0

    # ── Weather interactions ──
    wind = feat.get('wind_speed_10m', pd.Series(0, index=feat.index))
    temp = feat.get('temperature_2m', pd.Series(0, index=feat.index))
    feat['wind_power_proxy'] = wind ** 3  # wind energy ~ v^3
    feat['heating_degree']   = np.maximum(18 - temp, 0)
    feat['cooling_degree']   = np.maximum(temp - 24, 0)

    # ── Price interactions ──

    # ── 7b. ENTSO-E generation features ─────────────────────────────
    if "wind_generation_mw" in feat.columns:
        wind_gen = feat["wind_generation_mw"]
        feat["wind_gen_lag_1"] = wind_gen.shift(1)
        feat["wind_gen_lag_24"] = wind_gen.shift(24)
        feat["wind_gen_change_1h"] = wind_gen.diff(1)
        feat["wind_gen_change_24h"] = wind_gen.diff(24)
        feat["wind_gen_mean_6"] = wind_gen.rolling(6, min_periods=1).mean()
        feat["wind_gen_mean_24"] = wind_gen.rolling(24, min_periods=1).mean()
        feat["wind_gen_std_24"] = wind_gen.rolling(24, min_periods=1).std()
        feat["price_wind_gen_ratio"] = prices / (wind_gen + 1e-8)

    if "solar_generation_mw" in feat.columns:
        solar_gen = feat["solar_generation_mw"]
        feat["solar_gen_lag_1"] = solar_gen.shift(1)
        feat["solar_gen_lag_24"] = solar_gen.shift(24)
        feat["solar_gen_mean_6"] = solar_gen.rolling(6, min_periods=1).mean()

    if "total_generation_mw" in feat.columns:
        total_gen = feat["total_generation_mw"]
        feat["total_gen_lag_1"] = total_gen.shift(1)
        feat["total_gen_lag_24"] = total_gen.shift(24)
        feat["total_gen_mean_24"] = total_gen.rolling(24, min_periods=1).mean()
        feat["total_gen_change_24h"] = total_gen.diff(24)
        feat["price_per_gen_mw"] = prices / (total_gen + 1e-8)

    feat['spike_indicator']       = (prices > prices.rolling(24).mean() + 2 * prices.rolling(24).std()).astype(float)
    feat['temp_peak_interaction'] = temp * feat.get('is_peak_hour', 0)
    feat['wind_price_interaction'] = wind * prices

    # ── Clean up ──
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    feat = feat.select_dtypes(include=[np.number])

    logger.info(f"Built {len(feat.columns)} features from {len(feat)} rows")
    return feat


def get_energy_price_range(zone: str = "DK1") -> tuple[float, float]:
    """Reasonable price range for Nordic bidding zones."""
    ranges = {
        "DK1": (-500, 1000), "DK2": (-500, 1000),
        "NO1": (-200, 800),  "NO2": (-200, 800),
        "SE1": (-200, 800),  "SE3": (-200, 800),
        "FI":  (-200, 1000),
    }
    return ranges.get(zone, (-500, 1000))


def get_max_step_change(zone: str = "DK1") -> float:
    """Max allowed per-step price change (EUR/MWh)."""
    return 0.50
