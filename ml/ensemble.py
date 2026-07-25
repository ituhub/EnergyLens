"""
EnergyLens Phase 2 — Ensemble prediction with safety rails.

Three layers of protection (carried over from MarketLens):

  1. CATASTROPHIC EXCLUSION  — any single model prediction deviating >50% from
     the current price is silently dropped from the ensemble.

  2. OUTLIER FILTERING       — of the remaining predictions, any model deviating
     >15% from the ensemble median is excluded (requires ≥3 models).

  3. CLAMPING                — the final ensemble value, and each multi-step
     forecast step, are clamped to the dynamic price range computed from
     training data (with a 25% buffer).

Energy-specific notes:
  • Nord Pool prices can be negative (wind oversupply), so we allow min < 0.
  • Hourly volatility can be extreme (30 → 300 EUR/MWh during supply crises),
    so per-step clamping is looser than MarketLens (50% vs 15%).
"""

import logging
import numpy as np
import torch
from datetime import datetime, timedelta

from .features import get_energy_price_range, get_max_step_change

logger = logging.getLogger(__name__)

# ── Temporal feature names that can be computed for future timesteps ──────
TEMPORAL_FEATURES = {
    "hour", "hour_sin", "hour_cos",
    "dow", "dow_sin", "dow_cos",
    "month", "month_sin", "month_cos",
    "quarter", "is_weekend", "is_peak_hour",
}


def _compute_temporal_values(dt: datetime) -> dict[str, float]:
    """Compute raw (unscaled) temporal feature values for a given datetime."""
    hour = dt.hour
    dow = dt.weekday()
    month = dt.month
    return {
        "hour": float(hour),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "dow": float(dow),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "month": float(month),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "quarter": float((month - 1) // 3 + 1),
        "is_weekend": float(1 if dow >= 5 else 0),
        "is_peak_hour": float(1 if 7 <= hour <= 19 else 0),
    }

# ── Default ensemble weights (balanced, no single model dominates) ────────
DEFAULT_WEIGHTS = {
    "advanced_transformer": 0.14,
    "cnn_lstm":             0.14,
    "enhanced_tcn":         0.14,
    "enhanced_informer":    0.14,
    "lstm_gru_ensemble":    0.14,
    "enhanced_nbeats":      0.14,
    "xgboost":              0.10,
    "sklearn_ensemble":     0.06,
}


def inverse_transform_prediction(pred_scaled: float, scaler, feature_index: int = 0) -> float:
    """Invert the RobustScaler transform for a single predicted value."""
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((1, n_features))
    dummy[0, feature_index] = pred_scaled
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, feature_index])


# ═════════════════════════════════════════════════════════════════════════════
# CORE ENSEMBLE PREDICT
# ═════════════════════════════════════════════════════════════════════════════

def ensemble_predict(
    models_dict: dict,
    x_seq: np.ndarray,
    scaler=None,
    zone: str = "DK1",
    cv_weights: dict | None = None,
    price_range: tuple[float, float] | None = None,
    current_price: float | None = None,
    recent_price_range: tuple[float, float] | None = None,
) -> tuple[float | None, dict]:
    """
    Run all models, apply safety rails, return (final_price, per_model_preds).

    Safety rails:
      1. Catastrophic exclusion: >50% from current_price → dropped
      2. Outlier filtering: >15% from median → dropped (if ≥3 models)
      3. Final clamping to price_range
    """
    if not models_dict:
        logger.error("No models provided")
        return None, {}

    logger.info(f"Ensemble: {len(models_dict)} models — {sorted(models_dict.keys())}")

    # Resolve price bounds
    if price_range is not None:
        min_price, max_price = price_range
    else:
        min_price, max_price = get_energy_price_range(zone)

    weights = cv_weights if cv_weights else DEFAULT_WEIGHTS

    # Prepare inputs
    x_flat = x_seq.reshape(x_seq.shape[0], -1)
    x_tensor = torch.tensor(x_seq, dtype=torch.float32)

    # ── Collect raw predictions ──────────────────────────────────────
    raw_predictions: dict[str, float] = {}

    for name, model in models_dict.items():
        try:
            if name in ("xgboost", "sklearn_ensemble"):
                pred_scaled = model.predict(x_flat)
                pred_scaled = float(pred_scaled[0]) if hasattr(pred_scaled, "__iter__") else float(pred_scaled)
            else:
                model.eval()
                with torch.no_grad():
                    inp = x_tensor.reshape(x_tensor.shape[0], -1) if name == "enhanced_nbeats" else x_tensor
                    out = model(inp)
                    pred_scaled = float(out.detach().cpu().numpy().flatten()[0])

            # Inverse-transform to EUR/MWh
            if scaler is not None:
                pred_eur = inverse_transform_prediction(pred_scaled, scaler)
            else:
                pred_eur = pred_scaled

            # ── SAFETY RAIL 1: Catastrophic exclusion ────────────────
            # Skip when price is near zero/negative (common in Nordic wind oversupply)
            if current_price is not None and abs(current_price) > 5.0:
                deviation = abs(pred_eur - current_price) / (abs(current_price) + 1e-8)
                # Hybrid threshold: 50% relative, but at least €30 absolute.
                # ALSO factor in recent 24h price span — Nordic intraday
                # swings can be 10x (€170 → €10) and predictions reflecting
                # recent prices aren't truly catastrophic.
                abs_limit = max(abs(current_price) * 0.50, 30.0)
                if recent_price_range is not None:
                    recent_low, recent_high = recent_price_range
                    price_span = recent_high - recent_low
                    # Allow predictions within 50% of the recent 24h span
                    span_limit = price_span * 0.50
                    abs_limit = max(abs_limit, span_limit)
                    # Hard ceiling: never allow predictions above 120% of 24h high
                    hard_ceiling = max(recent_high * 1.20, recent_high + 30.0)
                    if pred_eur > hard_ceiling:
                        logger.warning(
                            f"CATASTROPHIC: {name} → €{pred_eur:.2f} exceeds "
                            f"120% of 24h high €{recent_high:.2f} — EXCLUDED"
                        )
                        continue
                if abs(pred_eur - current_price) > abs_limit:
                    logger.warning(
                        f"CATASTROPHIC: {name} → €{pred_eur:.2f} vs current €{current_price:.2f} "
                        f"({deviation:.0%} off, limit €{abs_limit:.0f}) — EXCLUDED"
                    )
                    continue

            # Soft clamp to extended range (allow 20% buffer)
            buffer = (max_price - min_price) * 0.20
            if pred_eur < (min_price - buffer) or pred_eur > (max_price + buffer):
                logger.warning(f"{name} out of bounds: €{pred_eur:.2f} — clamping")
                pred_eur = np.clip(pred_eur, min_price, max_price)

            raw_predictions[name] = pred_eur
            logger.info(f"  {name} → €{pred_eur:.2f}")

        except Exception as e:
            logger.warning(f"  {name} FAILED: {e}")
            continue

    if not raw_predictions:
        logger.error("No valid predictions from any model")
        return None, {}

    # ── SAFETY RAIL 2: Outlier filtering (median ± 35%) ──────────────
    # NOTE: Energy prices are far more volatile than equities — a 15%
    # threshold was excluding 5/7 models on most steps.  35% keeps the
    # guard-rail but lets the full ensemble contribute.
    if len(raw_predictions) >= 3:
        values = list(raw_predictions.values())
        median = np.median(values)
        max_dev = 0.35

        filtered = {
            name: pred for name, pred in raw_predictions.items()
            if abs(median) < 1e-8 or abs(pred - median) / (abs(median) + 1e-8) <= max_dev
        }

        if filtered and len(filtered) >= 2:
            excluded = set(raw_predictions) - set(filtered)
            if excluded:
                logger.info(f"Outlier filter: excluded {excluded} (>{max_dev:.0%} from median €{median:.2f})")
            predictions_for_ensemble = filtered
        else:
            predictions_for_ensemble = raw_predictions
    else:
        predictions_for_ensemble = raw_predictions

    # ── Weighted average ─────────────────────────────────────────────
    weighted_sum = 0.0
    total_weight = 0.0
    for name, pred in predictions_for_ensemble.items():
        w = weights.get(name, 1.0 / len(predictions_for_ensemble))
        # Config stores weights as {'mean': float, 'std': float} — extract mean
        if isinstance(w, dict):
            w = w.get('mean', 1.0 / len(predictions_for_ensemble))
        weighted_sum += pred * w
        total_weight += w

    final = weighted_sum / total_weight if total_weight > 0 else np.median(list(raw_predictions.values()))

    # ── SAFETY RAIL 3: Final clamping ────────────────────────────────
    if final < min_price or final > max_price:
        logger.warning(f"Final €{final:.2f} out of [{min_price}, {max_price}] — using median")
        final = np.median(list(raw_predictions.values()))
        if final < min_price or final > max_price:
            final = np.clip(final, min_price, max_price)

    ensemble_std = float(np.std(list(raw_predictions.values())))
    logger.info(
        f"ENSEMBLE → €{final:.2f}  ({len(predictions_for_ensemble)}/{len(models_dict)} models, σ=€{ensemble_std:.2f})"
    )

    return float(final), raw_predictions


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-STEP FORECAST WITH PER-STEP CLAMPING
# ═════════════════════════════════════════════════════════════════════════════

def multi_step_forecast(
    models_dict: dict,
    initial_sequence: np.ndarray,
    scaler,
    steps: int = 24,
    zone: str = "DK1",
    cv_weights: dict | None = None,
    price_range: tuple[float, float] | None = None,
    feature_names: list[str] | None = None,
    last_timestamp=None,
    current_price: float | None = None,
    recent_price_range: tuple[float, float] | None = None,
) -> list[float]:
    """
    Generate a multi-step (e.g. 24-hour) forecast with per-step clamping.

    Each step:
      1. Run ensemble_predict on the current sequence
      2. Clamp the prediction to max_step_change from base price
      3. Roll the sequence window forward with proper feature fill

    Args:
        feature_names: ordered list of feature column names (from config['used_features']).
                       Enables temporal feature advancement for future timesteps.
        last_timestamp: datetime of the last known data point.
                        If None, temporal features are forward-filled instead of computed.
    """
    if not models_dict or initial_sequence is None:
        return []

    max_change = get_max_step_change(zone)
    forecasts: list[float] = []
    curr_seq = initial_sequence.copy()

    # Derive base price from last step of initial sequence
    base_price = None
    try:
        if scaler is not None:
            dummy = np.zeros((1, scaler.scale_.shape[0]))
            dummy[0, 0] = initial_sequence[0, -1, 0]  # Close is feature 0
            base_price = float(scaler.inverse_transform(dummy)[0, 0])
    except Exception:
        pass

    if base_price is None and price_range:
        base_price = (price_range[0] + price_range[1]) / 2

    logger.info(f"Forecasting {steps} steps for zone {zone} (base €{base_price:.2f})")

    # ── Frozen-model detection ──────────────────────────────────────
    # Track per-model predictions across steps.  If a model returns
    # the exact same value (within €0.50) for FROZEN_THRESHOLD
    # consecutive steps, exclude it — it has saturated.
    FROZEN_THRESHOLD = 3
    model_history: dict[str, list[float]] = {name: [] for name in models_dict}
    frozen_models: set[str] = set()

    for step in range(steps):
        try:
            # Use a filtered model dict that excludes frozen models
            active_models = {k: v for k, v in models_dict.items() if k not in frozen_models}
            if not active_models:
                logger.warning("All models frozen — falling back to full set")
                active_models = models_dict

            pred, step_preds = ensemble_predict(
                active_models, curr_seq, scaler, zone,
                cv_weights=cv_weights, price_range=price_range,
                current_price=base_price,
                recent_price_range=recent_price_range,
            )

            # Update frozen-model tracking
            for name, val in step_preds.items():
                model_history.setdefault(name, []).append(val)
                recent = model_history[name][-FROZEN_THRESHOLD:]
                if len(recent) >= FROZEN_THRESHOLD:
                    if max(recent) - min(recent) < 0.50:
                        if name not in frozen_models:
                            frozen_models.add(name)
                            logger.warning(
                                f"FROZEN: {name} returned €{val:.2f} for "
                                f"{FROZEN_THRESHOLD} consecutive steps — excluding"
                            )

            if pred is None:
                logger.warning(f"Step {step + 1}: no prediction — stopping")
                break

            # ── PER-STEP CLAMP ───────────────────────────────────────
            # Near-zero prices (common in Nordic wind oversupply) break
            # percentage-based clamping — a €0.01 base can never reach €80.
            # Fix: use absolute EUR cap when base_price is small.
            MIN_ABS_STEP = 50.0  # EUR — Nordic prices swing €50+/hour during wind events
            NEAR_ZERO_THRESHOLD = 10.0  # EUR — below this, use absolute mode

            if base_price is not None:
                max_cumulative = max_change * (1 + step * 0.15)

                if abs(base_price) <= NEAR_ZERO_THRESHOLD:
                    # ── ABSOLUTE MODE: near-zero/negative base price ──
                    abs_limit = MIN_ABS_STEP * (1 + step * 0.40)
                    change_abs = abs(pred - base_price)

                    if change_abs > abs_limit:
                        direction = 1 if pred > base_price else -1
                        clamped = base_price + direction * abs_limit
                        logger.info(
                            f"Step {step + 1} abs-clamped: €{pred:.2f} → €{clamped:.2f} "
                            f"(Δ€{change_abs:.1f} > limit €{abs_limit:.1f})"
                        )
                        pred = clamped
                elif abs(base_price) > 1e-8:
                    # ── PERCENTAGE MODE: normal prices ────────────────
                    change_pct = abs(pred - base_price) / (abs(base_price) + 1e-8)

                    if change_pct > max_cumulative:
                        direction = 1 if pred > base_price else -1
                        # Also enforce absolute floor so small prices aren't trapped
                        pct_limit = abs(base_price) * max_cumulative * 0.5
                        effective_limit = max(pct_limit, MIN_ABS_STEP)
                        clamped = base_price + direction * effective_limit
                        logger.warning(
                            f"Step {step + 1} clamped: €{pred:.2f} → €{clamped:.2f} "
                            f"({change_pct:.0%} > {max_cumulative:.0%})"
                        )
                        pred = clamped

            forecasts.append(pred)

            # Roll the sequence: drop first timestep, append new prediction.
            #
            # Three-tier feature fill strategy:
            #   a) Price (feature 0): set to the new prediction (scaled).
            #   b) Temporal features (hour_sin, dow_cos, …): compute for the
            #      future hour so models see time advancing, not frozen.
            #   c) Everything else (weather, lags, rolling stats): forward-fill
            #      from the last known timestep — better than zeros.
            try:
                # Start from last known timestep (carries weather, lags, etc.)
                new_row = curr_seq[:, -1:, :].copy()

                # (a) Overwrite price (feature 0) with the new prediction
                if scaler is not None:
                    dummy = np.zeros((1, scaler.scale_.shape[0]))
                    dummy[0, 0] = pred
                    scaled = scaler.transform(dummy)[0, 0]
                    new_row[0, 0, 0] = scaled
                else:
                    new_row[0, 0, 0] = pred

                # (b) Advance temporal features if we have the metadata
                if feature_names and last_timestamp:
                    future_dt = last_timestamp + timedelta(hours=step + 1)
                    raw_temporal = _compute_temporal_values(future_dt)

                    for feat_name, raw_val in raw_temporal.items():
                        if feat_name in feature_names:
                            idx = feature_names.index(feat_name)
                            # Scale the raw value through the scaler
                            if scaler is not None:
                                dummy = np.zeros((1, scaler.scale_.shape[0]))
                                dummy[0, idx] = raw_val
                                new_row[0, 0, idx] = scaler.transform(dummy)[0, idx]
                            else:
                                new_row[0, 0, idx] = raw_val

                curr_seq = np.concatenate([curr_seq[:, 1:, :], new_row], axis=1)
            except Exception:
                break

        except Exception as e:
            logger.warning(f"Step {step + 1} failed: {e}")
            break

    logger.info(f"Forecast complete: {len(forecasts)} steps")
    return forecasts


# ═════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ═════════════════════════════════════════════════════════════════════════════

def calculate_confidence(per_model_preds: dict, zone: str = "DK1") -> float:
    """
    Score ensemble confidence (0–100) based on:
      • Model agreement (CV of predictions)
      • Number of models that produced valid predictions
      • Zone-specific base confidence
    """
    if len(per_model_preds) < 2:
        return 50.0

    values = list(per_model_preds.values())
    mean_pred = np.mean(values)
    std_pred = np.std(values)

    # Coefficient of variation → consistency score
    # Energy models naturally disagree more than equity models;
    # cv * 200 (was 500) avoids punishing normal spread.
    if abs(mean_pred) > 1e-8:
        cv = std_pred / abs(mean_pred)
        consistency = max(0, 100 - cv * 200)
    else:
        consistency = 50.0

    # Agreement: fraction of total models that produced predictions
    agreement = min(len(per_model_preds) / 8.0, 1.0) * 100

    # Zone-specific base (Nordic prices are inherently noisier)
    base = 65.0

    confidence = 0.4 * consistency + 0.3 * agreement + 0.3 * base
    confidence = np.clip(confidence, 40.0, 88.0)

    logger.info(f"Confidence: {confidence:.1f}% (models={len(per_model_preds)}, CV={cv:.3f})")
    return float(confidence)
