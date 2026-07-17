"""
EnergyLens — SHAP Explainability Engine

Post-hoc feature importance using already-trained models.
No retraining required.

Strategy:
- XGBoost → shap.TreeExplainer (fast, exact SHAP values)
- Sklearn Ensemble → shap.TreeExplainer (fast)
- Fallback → model's built-in feature_importances_

Produces:
- Global feature importance rankings
- Per-forecast SHAP breakdown (which features push price up/down)
- Feature group summaries (weather vs time vs price lags)
"""

import os
import sys
import pickle
import sqlite3
import numpy as np
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Feature Group Definitions ────────────────────────────────────

FEATURE_GROUPS = {
    "Price Lags": [
        "price_lag_1h", "price_lag_2h", "price_lag_3h", "price_lag_6h",
        "price_lag_12h", "price_lag_24h", "price_lag_48h",
        "price_rolling_mean_6h", "price_rolling_mean_12h",
        "price_rolling_mean_24h", "price_rolling_std_6h",
        "price_rolling_std_12h", "price_rolling_std_24h",
        "price_diff_1h", "price_diff_24h", "price_pct_change_1h",
        "price_pct_change_24h", "price_momentum_6h", "price_momentum_12h",
        # Names that build_energy_features actually produces:
        "Close", "SpotPriceEUR", "return_1h", "return_24h", "return_168h",
        "momentum_6h", "momentum_24h",
        "price_mean_6", "price_std_6", "price_min_6", "price_max_6",
        "price_position_6", "price_zscore_6",
        "price_mean_12", "price_std_12", "price_min_12", "price_max_12",
        "price_position_12", "price_zscore_12",
        "price_mean_24", "price_std_24", "price_min_24", "price_max_24",
        "price_position_24", "price_zscore_24",
        "price_mean_48", "price_std_48", "price_min_48", "price_max_48",
        "price_position_48", "price_zscore_48",
        "price_mean_168", "price_std_168", "price_min_168", "price_max_168",
        "price_position_168", "price_zscore_168",
        "volatility_6h", "volatility_24h", "volatility_168h",
        "is_price_spike",
    ],
    "Weather": [
        "temperature_2m", "wind_speed_10m", "wind_speed_100m",
        "wind_direction_10m", "wind_direction", "relative_humidity_2m",
        "surface_pressure", "cloud_cover", "direct_radiation",
        "diffuse_radiation", "precipitation",
        "temp_rolling_mean_6h", "temp_rolling_mean_24h",
        "wind_rolling_mean_6h", "wind_rolling_mean_24h",
        "radiation_rolling_mean_6h", "radiation_rolling_mean_24h",
        "wind_x_component", "wind_y_component",
        "wind_price_interaction",
    ],
    "Calendar": [
        "hour", "day_of_week", "month", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "quarter",
        "is_business_hour", "is_peak_hour",
    ],
    "Demand Patterns": [
        "is_morning_ramp", "is_evening_ramp",
        "demand_proxy", "heating_degree_hours", "cooling_degree_hours",
        "temp_price_interaction",
    ],
    "Cross-Features": [
        "wind_x_radiation", "temp_x_hour", "wind_x_demand",
        "price_lag_x_volatility",
    ]
}


def _classify_feature(name: str) -> str:
    """Map a feature name to its group."""
    for group, features in FEATURE_GROUPS.items():
        if name in features:
            return group
    # Fallback heuristics
    if any(k in name for k in ["price", "lag", "return", "momentum", "Close", "zscore", "volatil"]):
        return "Price Lags"
    if any(k in name for k in ["wind", "temp", "radiation", "cloud", "precip", "humid", "pressure"]):
        return "Weather"
    if any(k in name for k in ["hour", "day", "month", "weekend", "peak", "business", "quarter"]):
        return "Calendar"
    if any(k in name for k in ["ramp", "demand", "heating", "cooling"]):
        return "Demand Patterns"
    return "Other"


class ShapEngine:
    """Post-hoc SHAP explainability for EnergyLens forecasts."""

    def __init__(self, model_dir: str = "models", db_path: str = "data/energylens.db"):
        self.model_dir = Path(model_dir)
        self.db_path = db_path
        self._xgb_model = None
        self._sklearn_model = None
        self._feature_names = None
        self._scaler = None
        self._shap_available = False

        try:
            import shap
            self._shap_available = True
        except ImportError:
            print("[SHAP] shap not installed — falling back to built-in feature importance")

    # ── Model Loading ────────────────────────────────────────────

    def _load_models(self, zone: str = "DK1"):
        """Load tree-based models for SHAP analysis."""
        zone_lower = zone.lower()

        # Custom unpickler to handle models trained on Kaggle (classes in __main__)
        try:
            import ml.models as _models_module
        except ImportError:
            _models_module = None

        class _ModelUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__" and _models_module and hasattr(_models_module, name):
                    return getattr(_models_module, name)
                return super().find_class(module, name)

        # Try loading XGBoost model
        xgb_paths = [
            self.model_dir / f"{zone.upper()}_xgboost.pkl",
            self.model_dir / f"xgboost_{zone_lower}.pkl",
            self.model_dir / f"{zone_lower}_xgboost.pkl",
        ]
        for path in xgb_paths:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        data = _ModelUnpickler(f).load()
                    if isinstance(data, dict):
                        self._xgb_model = data.get("model", data)
                        self._feature_names = data.get("feature_names")
                        self._scaler = data.get("scaler")
                    else:
                        self._xgb_model = data
                    print(f"[SHAP] Loaded XGBoost from {path}")
                    break
                except Exception as e:
                    print(f"[SHAP] Failed to load {path}: {e}")

        # Try loading sklearn ensemble
        sklearn_paths = [
            self.model_dir / f"{zone.upper()}_sklearn_ensemble.pkl",
            self.model_dir / f"sklearn_ensemble_{zone_lower}.pkl",
            self.model_dir / f"{zone_lower}_sklearn_ensemble.pkl",
        ]
        for path in sklearn_paths:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        data = _ModelUnpickler(f).load()
                    if isinstance(data, dict):
                        self._sklearn_model = data.get("model", data)
                        if not self._feature_names:
                            self._feature_names = data.get("feature_names")
                        if not self._scaler:
                            self._scaler = data.get("scaler")
                    else:
                        self._sklearn_model = data
                    print(f"[SHAP] Loaded sklearn ensemble from {path}")
                    break
                except Exception as e:
                    print(f"[SHAP] Failed to load {path}: {e}")

        # Load scaler separately if not bundled with model
        if not self._scaler:
            scaler_path = self.model_dir / f"{zone.upper()}_scaler.pkl"
            if scaler_path.exists():
                try:
                    with open(scaler_path, "rb") as f:
                        self._scaler = pickle.load(f)
                    print(f"[SHAP] Loaded scaler from {scaler_path}")
                except Exception as e:
                    print(f"[SHAP] Scaler load failed: {e}")

    # ── Feature Data ─────────────────────────────────────────────

    def _get_feature_data(self, zone: str = "DK1", hours: int = 200) -> Optional[np.ndarray]:
        """
        Get recent feature data for SHAP computation.
        Uses the same feature engineering as the forecast pipeline.
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from ml.features import build_energy_features
            import pandas as pd

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Fetch recent spot prices
            prices = conn.execute("""
                SELECT valid_time, price_eur_mwh, zone
                FROM spot_prices
                WHERE zone = ?
                ORDER BY valid_time DESC
                LIMIT ?
            """, (zone, hours * 3)).fetchall()
            conn.close()

            if not prices:
                print("[SHAP] No price data found")
                return None

            # Build price dataframe
            price_df = pd.DataFrame([dict(r) for r in prices])
            price_df = price_df.sort_values("valid_time").reset_index(drop=True)

            # Rename to match what build_energy_features expects
            price_df = price_df.rename(columns={"price_eur_mwh": "SpotPriceEUR"})

            # Parse valid_time to datetime so temporal features work
            price_df["valid_time"] = pd.to_datetime(price_df["valid_time"])

            # Build features
            features_df = build_energy_features(price_df)

            if features_df is None or len(features_df) == 0:
                print("[SHAP] Feature builder returned empty")
                return None

            # Keep only numeric columns (drop strings, datetimes, etc.)
            numeric_df = features_df.select_dtypes(include=[np.number])

            # Drop columns that are all NaN
            numeric_df = numeric_df.dropna(axis=1, how='all')

            # Fill remaining NaN with 0, replace inf
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            numeric_df = numeric_df.fillna(0)

            if numeric_df.empty:
                print("[SHAP] No numeric features after cleaning")
                return None

            self._feature_names = list(numeric_df.columns)
            result = numeric_df.values.astype(np.float64)

            print(f"[SHAP] Feature data ready: {result.shape} ({len(self._feature_names)} features)")
            return result

        except Exception as e:
            print(f"[SHAP] Feature data error: {e}")
            import traceback
            traceback.print_exc()

        return None

    # ── SHAP Computation ─────────────────────────────────────────

    def compute_shap(self, zone: str = "DK1") -> dict:
        """
        Compute SHAP values for the current forecast.
        Returns feature importance rankings and per-feature contributions.
        """
        self._load_models(zone)

        if not self._xgb_model and not self._sklearn_model:
            return self._fallback_importance(zone)

        feature_data = self._get_feature_data(zone)

        if feature_data is None or len(feature_data) == 0:
            return self._fallback_importance(zone)

        model = self._xgb_model or self._sklearn_model
        model_name = "XGBoost" if self._xgb_model else "Sklearn Ensemble"

        # Unwrap custom wrapper classes to get raw estimator
        raw_model = getattr(model, 'model', model)

        # Check expected feature count from the model
        expected_features = None
        try:
            if hasattr(raw_model, 'n_features_in_'):
                expected_features = raw_model.n_features_in_
        except Exception:
            pass

        n_actual = feature_data.shape[1] if len(feature_data.shape) == 2 else 0
        print(f"[SHAP] Features: {n_actual} actual, {expected_features} expected by model")

        # Use the last row (most recent) for current explanation
        X = feature_data[-1:] if len(feature_data.shape) == 2 else feature_data.reshape(1, -1)

        # If feature count doesn't match, skip TreeSHAP (it will crash)
        if expected_features and n_actual != expected_features:
            print(f"[SHAP] Feature mismatch ({n_actual} vs {expected_features}) — using built-in importance")
            return self._compute_builtin_importance(raw_model, model_name, X)

        # Background data for SHAP (use last N rows)
        n_background = min(50, len(feature_data))
        background = feature_data[-n_background:]

        if self._shap_available:
            return self._compute_shap_values(raw_model, model_name, X, background)
        else:
            return self._compute_builtin_importance(raw_model, model_name, X)

    def _compute_shap_values(self, raw_model, model_name: str,
                             X: np.ndarray, background: np.ndarray) -> dict:
        """Compute SHAP values using the shap library."""
        import shap

        try:
            explainer = shap.TreeExplainer(raw_model, background)
            shap_values = explainer.shap_values(X, check_additivity=False)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            # Build feature importance list
            feature_names = self._feature_names or [f"feature_{i}" for i in range(len(shap_row))]
            features = []
            for i, (name, sv) in enumerate(zip(feature_names, shap_row)):
                sv_float = float(sv)
                features.append({
                    "feature": name,
                    "shap_value": round(sv_float, 4),
                    "abs_shap": round(abs(sv_float), 4),
                    "direction": "up" if sv_float > 0 else "down" if sv_float < 0 else "neutral",
                    "group": _classify_feature(name),
                    "input_value": round(float(X[0][i]), 4) if i < X.shape[1] else None
                })

            # Sort by absolute SHAP value
            features.sort(key=lambda f: f["abs_shap"], reverse=True)

            # Group summaries
            groups = self._build_group_summary(features)

            base_value = float(explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])

            return {
                "zone": "DK1",
                "model_used": model_name,
                "method": "TreeSHAP",
                "base_value": round(base_value, 2),
                "predicted_value": round(base_value + sum(f["shap_value"] for f in features), 2),
                "top_features": features[:15],
                "all_features": features,
                "group_summary": groups,
                "n_features": len(features),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            print(f"[SHAP] TreeExplainer failed: {e}, falling back to builtin")
            return self._compute_builtin_importance(raw_model, model_name, X)

    def _compute_builtin_importance(self, raw_model, model_name: str,
                                    X: np.ndarray) -> dict:
        """
        Fallback: use model's built-in feature importance.
        Works without shap library.
        """
        importance = None

        try:
            if hasattr(raw_model, 'feature_importances_'):
                importance = raw_model.feature_importances_
            elif hasattr(raw_model, 'get_booster'):
                booster = raw_model.get_booster()
                scores = booster.get_score(importance_type='gain')
                n_features = X.shape[1] if X is not None else 0
                feature_names = self._feature_names or [f"f{i}" for i in range(n_features)]
                importance = np.zeros(len(feature_names))
                for fname, score in scores.items():
                    idx = int(fname.replace("f", "")) if fname.startswith("f") else None
                    if idx is not None and idx < len(importance):
                        importance[idx] = score
        except Exception as e:
            print(f"[SHAP] Built-in importance failed: {e}")

        if importance is None or len(importance) == 0:
            return {
                "zone": "DK1",
                "model_used": model_name,
                "method": "unavailable",
                "error": "Could not extract feature importance",
                "top_features": [],
                "group_summary": [],
                "n_features": 0,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        feature_names = self._feature_names or [f"feature_{i}" for i in range(len(importance))]

        # Handle length mismatch
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

        # Normalize
        total = float(np.sum(importance))
        if total == 0:
            total = 1.0

        features = []
        for name, imp in zip(feature_names, importance):
            norm_imp = float(imp / total)
            features.append({
                "feature": name,
                "importance": round(norm_imp, 4),
                "shap_value": round(norm_imp, 4),
                "abs_shap": round(norm_imp, 4),
                "direction": "unknown",
                "group": _classify_feature(name)
            })

        features.sort(key=lambda f: f["abs_shap"], reverse=True)

        # Group summaries
        groups = self._build_group_summary(features)

        return {
            "zone": "DK1",
            "model_used": model_name,
            "method": "feature_importance",
            "top_features": features[:15],
            "all_features": features,
            "group_summary": groups,
            "n_features": len(features),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def _build_group_summary(self, features: list) -> list:
        """Build group impact summary from feature list."""
        group_impact = {}
        for f in features:
            g = f["group"]
            if g not in group_impact:
                group_impact[g] = {"total_abs": 0, "net": 0, "count": 0}
            group_impact[g]["total_abs"] += f.get("abs_shap", 0)
            group_impact[g]["net"] += f.get("shap_value", 0)
            group_impact[g]["count"] += 1

        return [
            {
                "group": name,
                "total_impact": round(data["total_abs"], 4),
                "net_direction": "up" if data["net"] > 0 else "down" if data["net"] < 0 else "unknown",
                "net_shap": round(data["net"], 4),
                "feature_count": data["count"]
            }
            for name, data in sorted(group_impact.items(),
                                     key=lambda x: x[1]["total_abs"],
                                     reverse=True)
        ]

    def _fallback_importance(self, zone: str) -> dict:
        """
        Last resort: return domain-knowledge estimates.
        """
        default_groups = [
            {"group": "Price Lags", "total_impact": 0.42, "net_direction": "up",
             "feature_count": 18,
             "description": "Historical prices and momentum drive ~42% of predictions"},
            {"group": "Weather", "total_impact": 0.28, "net_direction": "down",
             "feature_count": 18,
             "description": "Wind, solar radiation, temperature affect supply-side ~28%"},
            {"group": "Calendar", "total_impact": 0.18, "net_direction": "up",
             "feature_count": 12,
             "description": "Hour-of-day and weekday patterns drive demand ~18%"},
            {"group": "Demand Patterns", "total_impact": 0.08, "net_direction": "up",
             "feature_count": 6,
             "description": "Peak hours and heating/cooling demand ~8%"},
            {"group": "Cross-Features", "total_impact": 0.04, "net_direction": "neutral",
             "feature_count": 4,
             "description": "Interaction effects ~4%"}
        ]

        return {
            "zone": zone,
            "model_used": "domain_knowledge",
            "method": "domain_default",
            "note": "SHAP computation unavailable — showing domain-knowledge estimates. "
                    "Install shap (pip install shap) for exact values.",
            "top_features": [],
            "group_summary": default_groups,
            "n_features": 0,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def get_feature_groups(self) -> dict:
        """Return feature group definitions for the frontend."""
        return {
            name: {
                "features": features,
                "count": len(features),
                "description": {
                    "Price Lags": "Historical spot prices, rolling statistics, and momentum indicators",
                    "Weather": "Temperature, wind speed, solar radiation, precipitation, cloud cover",
                    "Calendar": "Hour of day, day of week, month, weekend/holiday indicators",
                    "Demand Patterns": "Peak hours, morning/evening ramps, heating/cooling degree hours",
                    "Cross-Features": "Interaction terms between weather, demand, and price features"
                }.get(name, "")
            }
            for name, features in FEATURE_GROUPS.items()
        }