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

from .features import get_energy_price_range, get_max_step_change

logger = logging.getLogger(__name__)

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
            if current_price is not None and current_price != 0:
                deviation = abs(pred_eur - current_price) / (abs(current_price) + 1e-8)
                if deviation > 0.50:
                    logger.warning(
                        f"CATASTROPHIC: {name} → €{pred_eur:.2f} vs current €{current_price:.2f} "
                        f"({deviation:.0%} off) — EXCLUDED"
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

    # ── SAFETY RAIL 2: Outlier filtering (median ± 15%) ──────────────
    if len(raw_predictions) >= 3:
        values = list(raw_predictions.values())
        median = np.median(values)
        max_dev = 0.15

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
) -> list[float]:
    """
    Generate a multi-step (e.g. 24-hour) forecast with per-step clamping.

    Each step:
      1. Run ensemble_predict on the current sequence
      2. Clamp the prediction to max_step_change from base price
      3. Roll the sequence window forward
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

    for step in range(steps):
        try:
            pred, _ = ensemble_predict(
                models_dict, curr_seq, scaler, zone,
                cv_weights=cv_weights, price_range=price_range,
                current_price=base_price,
            )

            if pred is None:
                logger.warning(f"Step {step + 1}: no prediction — stopping")
                break

            # ── PER-STEP CLAMP ───────────────────────────────────────
            if base_price is not None and abs(base_price) > 1e-8:
                # Allow cumulative change to grow per step (+15% slack)
                max_cumulative = max_change * (1 + step * 0.15)
                change_pct = abs(pred - base_price) / (abs(base_price) + 1e-8)

                if change_pct > max_cumulative:
                    direction = 1 if pred > base_price else -1
                    clamped = base_price * (1 + direction * max_cumulative * 0.5)
                    logger.warning(
                        f"Step {step + 1} clamped: €{pred:.2f} → €{clamped:.2f} "
                        f"({change_pct:.0%} > {max_cumulative:.0%})"
                    )
                    pred = clamped

            forecasts.append(pred)

            # Roll the sequence: drop first timestep, append new prediction
            try:
                new_row = np.zeros((1, 1, curr_seq.shape[2]))
                if scaler is not None:
                    dummy = np.zeros((1, scaler.scale_.shape[0]))
                    dummy[0, 0] = pred
                    scaled = scaler.transform(dummy)[0, 0]
                    new_row[0, 0, 0] = scaled
                else:
                    new_row[0, 0, 0] = pred

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
    if abs(mean_pred) > 1e-8:
        cv = std_pred / abs(mean_pred)
        consistency = max(0, 100 - cv * 500)
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
