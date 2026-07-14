"""
EnergyLens — Signal Quality Gate.

5-gate forecast reliability scoring adapted from MarketLens.
Each gate returns PASS / FAIL / WARN with a reason string.

Gates:
  1. Confidence Band  — Is ensemble confidence in the optimal range?
  2. Model Consensus  — Do models agree on price direction & magnitude?
  3. Data Freshness   — Is input data current enough for reliable forecasting?
  4. Forecast Stability — Are consecutive steps stable (not oscillating)?
  5. Volatility Check  — Is market volatility within normal bounds?
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


def evaluate_quality_gate(
    confidence: float,
    per_model_preds: dict,
    forecast_prices: list[float],
    current_price: float,
    last_data_timestamp: datetime | None = None,
    zone: str = "DK1",
) -> dict:
    """
    Run all 5 quality gates and return structured results.

    Returns:
        {
            "overall": "PASS" | "FAIL",
            "passed": int,
            "total": 5,
            "gates": [
                {"name": str, "status": "PASS"|"FAIL"|"WARN", "reason": str, "icon": str},
                ...
            ],
            "summary": str,
        }
    """
    gates = [
        _gate_confidence_band(confidence),
        _gate_model_consensus(per_model_preds, current_price),
        _gate_data_freshness(last_data_timestamp),
        _gate_forecast_stability(forecast_prices),
        _gate_volatility(per_model_preds, current_price, zone),
    ]

    passed = sum(1 for g in gates if g["status"] == "PASS")
    total = len(gates)

    # Overall: PASS if ≥3 gates pass, no critical failures
    critical_fails = [g for g in gates if g["status"] == "FAIL" and g.get("critical")]
    if critical_fails:
        overall = "FAIL"
        summary = f"Critical issue: {critical_fails[0]['reason']}"
    elif passed >= 3:
        overall = "PASS"
        summary = _get_pass_summary(passed, total, confidence)
    else:
        overall = "FAIL"
        summary = f"Only {passed}/{total} gates passed — forecast reliability is low"

    result = {
        "overall": overall,
        "passed": passed,
        "total": total,
        "gates": gates,
        "summary": summary,
    }

    logger.info(f"Quality Gate: {overall} ({passed}/{total}) — {summary}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL GATES
# ═════════════════════════════════════════════════════════════════════════════

def _gate_confidence_band(confidence: float) -> dict:
    """
    Gate 1: Confidence Band
    Sweet spot is 55-80%. Too low = unreliable. Too high = possibly overfitting.
    """
    gate = {"name": "Confidence Band", "icon": "confidence"}

    if 55.0 <= confidence <= 80.0:
        gate["status"] = "PASS"
        gate["reason"] = f"Confidence {confidence:.1f}% is in the 55-80% optimal range"
    elif 45.0 <= confidence < 55.0:
        gate["status"] = "WARN"
        gate["reason"] = f"Confidence {confidence:.1f}% is below optimal — moderate reliability"
    elif confidence > 80.0:
        gate["status"] = "WARN"
        gate["reason"] = f"Confidence {confidence:.1f}% is unusually high — verify model diversity"
    else:
        gate["status"] = "FAIL"
        gate["reason"] = f"Confidence {confidence:.1f}% is too low — models disagree significantly"

    return gate


def _gate_model_consensus(per_model_preds: dict, current_price: float) -> dict:
    """
    Gate 2: Model Consensus
    Check if models agree on direction and magnitude relative to current price.
    """
    gate = {"name": "Model Consensus", "icon": "consensus"}

    if len(per_model_preds) < 3:
        gate["status"] = "FAIL"
        gate["reason"] = f"Only {len(per_model_preds)} models produced predictions — need ≥3"
        gate["critical"] = True
        return gate

    values = list(per_model_preds.values())
    mean_pred = np.mean(values)
    std_pred = np.std(values)

    # Direction consensus: how many agree on up vs down from current?
    if abs(current_price) > 1.0:
        directions = [1 if p > current_price else -1 for p in values]
        bullish = sum(1 for d in directions if d > 0)
        bearish = len(directions) - bullish
        consensus_pct = max(bullish, bearish) / len(directions) * 100
    else:
        # Near-zero price — check agreement on magnitude instead
        consensus_pct = 100 - min(100, (std_pred / (abs(mean_pred) + 1e-8)) * 100)

    # CV-based agreement
    if abs(mean_pred) > 1e-8:
        cv = std_pred / abs(mean_pred)
    else:
        cv = 1.0

    if cv < 0.20 and consensus_pct >= 70:
        gate["status"] = "PASS"
        gate["reason"] = (
            f"{len(per_model_preds)} models agree — "
            f"CV={cv:.2f}, {consensus_pct:.0f}% directional consensus"
        )
    elif cv < 0.35 or consensus_pct >= 60:
        gate["status"] = "WARN"
        gate["reason"] = (
            f"Moderate agreement — CV={cv:.2f}, "
            f"{consensus_pct:.0f}% directional consensus"
        )
    else:
        gate["status"] = "FAIL"
        gate["reason"] = (
            f"Models disagree — CV={cv:.2f}, "
            f"spread €{std_pred:.1f} across {len(per_model_preds)} models"
        )

    return gate


def _gate_data_freshness(last_data_timestamp: datetime | None) -> dict:
    """
    Gate 3: Data Freshness
    Check how stale the input data is. Energy markets move fast.
    """
    gate = {"name": "Data Freshness", "icon": "freshness"}

    if last_data_timestamp is None:
        gate["status"] = "WARN"
        gate["reason"] = "Cannot determine data timestamp"
        return gate

    now = datetime.now(timezone.utc)
    # Ensure both are timezone-aware
    if last_data_timestamp.tzinfo is None:
        last_data_timestamp = last_data_timestamp.replace(tzinfo=timezone.utc)

    staleness = now - last_data_timestamp
    hours_stale = staleness.total_seconds() / 3600

    if hours_stale <= 2.0:
        gate["status"] = "PASS"
        gate["reason"] = f"Data is {hours_stale:.1f}h old — current"
    elif hours_stale <= 6.0:
        gate["status"] = "WARN"
        gate["reason"] = f"Data is {hours_stale:.1f}h old — slightly stale"
    else:
        gate["status"] = "FAIL"
        gate["reason"] = f"Data is {hours_stale:.1f}h old — too stale for reliable forecasting"
        gate["critical"] = True

    return gate


def _gate_forecast_stability(forecast_prices: list[float]) -> dict:
    """
    Gate 4: Forecast Stability
    Check if consecutive forecast steps are stable (not oscillating wildly).
    Some variation is expected (hourly price patterns), but extreme oscillation
    indicates model instability.
    """
    gate = {"name": "Forecast Stability", "icon": "stability"}

    if len(forecast_prices) < 4:
        gate["status"] = "WARN"
        gate["reason"] = f"Only {len(forecast_prices)} forecast steps — insufficient to assess"
        return gate

    # Calculate step-to-step changes
    changes = [abs(forecast_prices[i+1] - forecast_prices[i])
               for i in range(len(forecast_prices) - 1)]
    mean_change = np.mean(changes)
    max_change = max(changes)

    # Count direction reversals (sign changes in consecutive diffs)
    diffs = [forecast_prices[i+1] - forecast_prices[i]
             for i in range(len(forecast_prices) - 1)]
    reversals = sum(1 for i in range(len(diffs) - 1)
                    if diffs[i] * diffs[i+1] < 0)
    reversal_rate = reversals / (len(diffs) - 1) if len(diffs) > 1 else 0

    # Forecast range
    forecast_range = max(forecast_prices) - min(forecast_prices)

    if mean_change < 15.0 and reversal_rate < 0.6 and max_change < 40.0:
        gate["status"] = "PASS"
        gate["reason"] = (
            f"Stable trajectory — avg step Δ€{mean_change:.1f}, "
            f"range €{forecast_range:.0f}"
        )
    elif mean_change < 25.0 and reversal_rate < 0.75:
        gate["status"] = "WARN"
        gate["reason"] = (
            f"Some oscillation — avg step Δ€{mean_change:.1f}, "
            f"{reversals} reversals in {len(diffs)} steps"
        )
    else:
        gate["status"] = "FAIL"
        gate["reason"] = (
            f"Unstable forecast — avg step Δ€{mean_change:.1f}, "
            f"{reversals} reversals, max jump €{max_change:.1f}"
        )

    return gate


def _gate_volatility(
    per_model_preds: dict, current_price: float, zone: str
) -> dict:
    """
    Gate 5: Volatility Check
    Is the ensemble's predicted range within normal bounds for this zone?
    """
    gate = {"name": "Volatility", "icon": "volatility"}

    if len(per_model_preds) < 2:
        gate["status"] = "WARN"
        gate["reason"] = "Not enough models to assess volatility"
        return gate

    values = list(per_model_preds.values())
    pred_range = max(values) - min(values)
    std_pred = np.std(values)

    # Normal volatility thresholds per zone (EUR/MWh)
    # Nordic prices can be very volatile, but within bounds
    normal_std = {
        "DK1": 35.0, "DK2": 35.0,
        "NO1": 25.0, "NO2": 25.0,
        "SE1": 20.0, "SE3": 30.0,
        "FI": 30.0,
    }
    threshold = normal_std.get(zone, 35.0)

    if std_pred <= threshold * 0.8:
        gate["status"] = "PASS"
        gate["reason"] = (
            f"Normal volatility (σ=€{std_pred:.1f}) — "
            f"standard position sizing OK"
        )
    elif std_pred <= threshold * 1.2:
        gate["status"] = "WARN"
        gate["reason"] = (
            f"Elevated volatility (σ=€{std_pred:.1f}) — "
            f"wider confidence bands recommended"
        )
    else:
        gate["status"] = "FAIL"
        gate["reason"] = (
            f"High volatility (σ=€{std_pred:.1f}, range €{pred_range:.0f}) — "
            f"forecasts less reliable"
        )

    return gate


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _get_pass_summary(passed: int, total: int, confidence: float) -> str:
    """Generate a contextual summary for passing results."""
    if passed == total:
        return "All gates passed — high forecast reliability"
    elif passed >= 4:
        return f"Strong forecast quality with {confidence:.1f}% confidence"
    else:
        return f"Acceptable forecast quality — {passed}/{total} gates passed"
