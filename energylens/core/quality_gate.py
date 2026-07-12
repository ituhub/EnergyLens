"""
EnergyLens — Data Quality Gate.
Adapted from MarketLens Signal Quality Gate (5-check validation).

Every data record passes through 5 gates before entering the
standardized data layer. Failed records are quarantined with
reason codes, not dropped.

Gates:
1. Completeness — required fields present
2. Range — values within physical bounds
3. Freshness — data not stale beyond threshold
4. Consistency — cross-source agreement checks
5. Anomaly — statistical outlier detection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

from config.constants import QUALITY_GATES

logger = logging.getLogger("energylens.core.quality")


class GateResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class GateCheck:
    """Result of a single quality gate check."""
    gate: str
    result: GateResult
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    """Aggregate quality report for a data record."""
    record_id: str
    source: str
    timestamp: str
    gates: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.result != GateResult.FAIL for g in self.gates)

    @property
    def has_warnings(self) -> bool:
        return any(g.result == GateResult.WARN for g in self.gates)

    @property
    def failed_gates(self) -> list[str]:
        return [g.gate for g in self.gates if g.result == GateResult.FAIL]

    @property
    def summary(self) -> dict:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "passed": self.passed,
            "warnings": self.has_warnings,
            "failed_gates": self.failed_gates,
            "gate_count": len(self.gates),
            "pass_count": sum(1 for g in self.gates if g.result == GateResult.PASS),
        }


class QualityGate:
    """
    Five-gate data quality validator.

    Usage:
        gate = QualityGate()
        report = gate.validate(record, data_type="spot_price")
        if report.passed:
            # proceed to standardized layer
        else:
            # quarantine with reason codes
    """

    def __init__(self):
        self._rolling_stats: dict[str, list[float]] = {}
        self._rolling_window = 168  # 1 week of hourly data

    def validate(self, record: dict, data_type: str = "spot_price") -> QualityReport:
        """
        Run all 5 gates on a data record.

        Args:
            record: Data record to validate
            data_type: Type of data ("spot_price", "weather", "generation")

        Returns:
            QualityReport with individual gate results
        """
        record_id = f"{record.get('source', 'unknown')}_{record.get('timestamp_utc', 'no_ts')}"
        report = QualityReport(
            record_id=record_id,
            source=record.get("source", "unknown"),
            timestamp=record.get("timestamp_utc", ""),
        )

        # Run each gate
        report.gates.append(self._gate_completeness(record, data_type))
        report.gates.append(self._gate_range(record, data_type))
        report.gates.append(self._gate_freshness(record, data_type))
        report.gates.append(self._gate_consistency(record, data_type))
        report.gates.append(self._gate_anomaly(record, data_type))

        if not report.passed:
            logger.warning(
                f"Quality gate FAILED for {record_id}: {report.failed_gates}"
            )
        elif report.has_warnings:
            logger.info(f"Quality gate PASSED with warnings: {record_id}")

        return report

    # ─── Gate 1: Completeness ───

    def _gate_completeness(self, record: dict, data_type: str) -> GateCheck:
        """Check that required fields are present and non-null."""
        required_fields = {
            "spot_price": ["timestamp_utc", "zone", "price_eur_mwh"],
            "weather": ["timestamp_utc", "temperature_2m", "wind_speed_10m"],
            "generation": ["timestamp_utc", "area", "value_mw"],
        }

        fields = required_fields.get(data_type, ["timestamp_utc"])
        present = sum(1 for f in fields if record.get(f) is not None)
        ratio = present / len(fields) if fields else 1.0
        threshold = QUALITY_GATES["completeness"]["min_fields_present"]

        if ratio >= threshold:
            return GateCheck("completeness", GateResult.PASS, f"{present}/{len(fields)} fields present")
        else:
            missing = [f for f in fields if record.get(f) is None]
            return GateCheck(
                "completeness", GateResult.FAIL,
                f"Missing fields: {missing}",
                {"missing": missing, "ratio": ratio}
            )

    # ─── Gate 2: Range ───

    def _gate_range(self, record: dict, data_type: str) -> GateCheck:
        """Check values are within physical bounds."""
        ranges = QUALITY_GATES["range"]
        violations = []

        if data_type == "spot_price":
            price = record.get("price_eur_mwh")
            if price is not None:
                if price < ranges["price_min_eur_mwh"]:
                    violations.append(f"price {price} < min {ranges['price_min_eur_mwh']}")
                if price > ranges["price_max_eur_mwh"]:
                    violations.append(f"price {price} > max {ranges['price_max_eur_mwh']}")

        elif data_type == "weather":
            wind = record.get("wind_speed_10m")
            if wind is not None and wind > ranges["wind_speed_max_ms"]:
                violations.append(f"wind_speed {wind} > max {ranges['wind_speed_max_ms']}")

            temp = record.get("temperature_2m")
            if temp is not None:
                if temp < ranges["temperature_min_c"]:
                    violations.append(f"temperature {temp} < min {ranges['temperature_min_c']}")
                if temp > ranges["temperature_max_c"]:
                    violations.append(f"temperature {temp} > max {ranges['temperature_max_c']}")

        if violations:
            return GateCheck("range", GateResult.FAIL, "; ".join(violations))
        return GateCheck("range", GateResult.PASS, "All values within bounds")

    # ─── Gate 3: Freshness ───

    def _gate_freshness(self, record: dict, data_type: str) -> GateCheck:
        """Check data is not stale beyond threshold."""
        fetched_at = record.get("fetched_at")
        if not fetched_at:
            return GateCheck("freshness", GateResult.WARN, "No fetched_at timestamp")

        try:
            fetch_time = datetime.fromisoformat(fetched_at)
            if fetch_time.tzinfo is None:
                fetch_time = fetch_time.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - fetch_time).total_seconds()
        except (ValueError, TypeError):
            return GateCheck("freshness", GateResult.WARN, f"Unparseable fetched_at: {fetched_at}")

        thresholds = QUALITY_GATES["freshness"]
        max_age_key = f"{data_type}_max_age_seconds"
        # Default to 1 hour if data type not in thresholds
        max_age = thresholds.get(max_age_key, 3600)

        if age > max_age:
            return GateCheck(
                "freshness", GateResult.FAIL,
                f"Data age {age:.0f}s exceeds max {max_age}s",
                {"age_seconds": age, "max_seconds": max_age}
            )
        elif age > max_age * 0.8:
            return GateCheck(
                "freshness", GateResult.WARN,
                f"Data age {age:.0f}s approaching max {max_age}s"
            )
        return GateCheck("freshness", GateResult.PASS, f"Data age {age:.0f}s OK")

    # ─── Gate 4: Consistency ───

    def _gate_consistency(self, record: dict, data_type: str) -> GateCheck:
        """Cross-source consistency checks."""
        # For spot prices: EUR and DKK prices should be consistent
        if data_type == "spot_price":
            eur = record.get("price_eur_mwh")
            dkk = record.get("price_dkk_mwh")
            if eur is not None and dkk is not None and eur != 0:
                implied_rate = dkk / eur
                # EUR/DKK rate should be roughly 7.4–7.5 (pegged)
                if not (7.0 <= implied_rate <= 8.0):
                    return GateCheck(
                        "consistency", GateResult.WARN,
                        f"Implied EUR/DKK rate {implied_rate:.2f} outside expected range",
                        {"implied_rate": implied_rate}
                    )

        return GateCheck("consistency", GateResult.PASS, "Consistency check passed")

    # ─── Gate 5: Anomaly Detection ───

    def _gate_anomaly(self, record: dict, data_type: str) -> GateCheck:
        """Statistical outlier detection using rolling z-score."""
        if data_type == "spot_price":
            value = record.get("price_eur_mwh")
            stat_key = f"price_{record.get('zone', 'unknown')}"
        elif data_type == "generation":
            value = record.get("value_mw")
            stat_key = f"gen_{record.get('area', 'unknown')}"
        else:
            return GateCheck("anomaly", GateResult.PASS, "No anomaly check for this type")

        if value is None:
            return GateCheck("anomaly", GateResult.PASS, "No value to check")

        # Update rolling stats
        if stat_key not in self._rolling_stats:
            self._rolling_stats[stat_key] = []

        history = self._rolling_stats[stat_key]
        history.append(value)

        # Keep only the rolling window
        if len(history) > self._rolling_window:
            self._rolling_stats[stat_key] = history[-self._rolling_window:]
            history = self._rolling_stats[stat_key]

        # Need at least 24 hours of data for meaningful stats
        if len(history) < 24:
            return GateCheck("anomaly", GateResult.PASS, f"Insufficient history ({len(history)} points)")

        mean = np.mean(history[:-1])  # Exclude current value
        std = np.std(history[:-1])

        if std == 0:
            return GateCheck("anomaly", GateResult.PASS, "Zero variance — cannot compute z-score")

        z_score = abs(value - mean) / std
        threshold = QUALITY_GATES["anomaly"]["price_zscore_threshold"]

        if z_score > threshold:
            return GateCheck(
                "anomaly", GateResult.WARN,
                f"Z-score {z_score:.2f} exceeds threshold {threshold} "
                f"(value={value:.2f}, mean={mean:.2f}, std={std:.2f})",
                {"z_score": z_score, "value": value, "mean": mean, "std": std}
            )
        return GateCheck("anomaly", GateResult.PASS, f"Z-score {z_score:.2f} OK")

    def get_stats(self) -> dict:
        """Return current rolling statistics for monitoring."""
        return {
            key: {
                "count": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None,
            }
            for key, vals in self._rolling_stats.items()
        }
