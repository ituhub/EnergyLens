"""
EnergyLens ML — 8-model neural ensemble for Nordic energy price forecasting.

Adapted from MarketLens with three layers of prediction safety:
  1. Catastrophic exclusion (>50% deviation from current price)
  2. Outlier filtering (>15% from ensemble median)
  3. Per-step clamping (max cumulative change from base price)

Usage:
    from energylens.ml import train_all_models, ensemble_predict, multi_step_forecast
    from energylens.ml.features import build_energy_features

    # 1. Build features from raw data
    featured_df = build_energy_features(raw_df)

    # 2. Train all 8 models
    models, scaler, config = train_all_models(featured_df, featured_df.columns.tolist())

    # 3. Run ensemble prediction
    price, per_model = ensemble_predict(models, last_sequence, scaler)

    # 4. Multi-step forecast
    forecast_24h = multi_step_forecast(models, last_sequence, scaler, steps=24)
"""

from .training import train_all_models, save_models, load_models
from .ensemble import ensemble_predict, multi_step_forecast, calculate_confidence
from .features import build_energy_features, get_energy_price_range

__all__ = [
    "train_all_models",
    "save_models",
    "load_models",
    "ensemble_predict",
    "multi_step_forecast",
    "calculate_confidence",
    "build_energy_features",
    "get_energy_price_range",
]
