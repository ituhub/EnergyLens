"""
EnergyLens — Pipeline Tracer.

Collects stage-by-stage timing and metadata during a forecast run.
Powers the animated stepper in the React frontend.

6 pipeline stages (same as MarketLens transparency system):
    1. Data       — Load spot prices, weather, generation from SQLite
    2. Features   — Build ~125 energy features (lags, rolling, generation ratios)
    3. Models     — Load/cache 7-model ensemble (Transformer, CNN-LSTM, etc.)
    4. Ensemble   — Run multi-step forecast with outlier filtering
    5. Gate       — Signal Quality Gate (5 checks)
    6. Result     — Format response, log to accuracy engine

Usage:
    from api.pipeline_tracer import PipelineTrace

    trace = PipelineTrace(zone="DK1")
    with trace.stage("data"):
        data = load_data()
    trace.set_meta("rows_loaded", len(data))

    # Include in API response:
    return {**forecast_result, "pipeline_trace": trace.to_dict()}
"""

import time
from datetime import datetime, timezone
from typing import Any


PIPELINE_STAGES = [
    {"id": "data",     "label": "Data Loading",      "icon": "📡", "description": "Spot prices · Weather · Generation"},
    {"id": "features", "label": "Feature Engineering", "icon": "⚙️", "description": "125+ energy market features"},
    {"id": "models",   "label": "Model Loading",      "icon": "🧠", "description": "7-model neural ensemble"},
    {"id": "ensemble", "label": "Ensemble Forecast",   "icon": "📊", "description": "Multi-step price prediction"},
    {"id": "gate",     "label": "Quality Gate",        "icon": "🛡️", "description": "5-check signal validation"},
    {"id": "result",   "label": "Result",              "icon": "✅", "description": "Forecast packaged & logged"},
]


class PipelineTrace:
    """
    Accumulates stage timings and metadata for a single forecast run.
    Thread-safe for single-request use (one trace per API call).
    """

    def __init__(self, zone: str = "DK1"):
        self.zone = zone
        self.started_at = datetime.now(timezone.utc)
        self._stages: dict[str, dict] = {}
        self._meta: dict[str, Any] = {}
        self._total_start = time.perf_counter()

    def stage(self, stage_id: str) -> "_StageContext":
        """Context manager for timing a pipeline stage."""
        return _StageContext(self, stage_id)

    def set_meta(self, key: str, value: Any):
        """Attach metadata to the trace (model names, row counts, etc.)."""
        self._meta[key] = value

    def to_dict(self) -> dict:
        """
        Serialize the trace for the API response.
        The frontend stepper replays this data.
        """
        total_ms = round((time.perf_counter() - self._total_start) * 1000, 1)

        stages_list = []
        for s in PIPELINE_STAGES:
            stage_data = self._stages.get(s["id"], {})
            stages_list.append({
                "id": s["id"],
                "label": s["label"],
                "icon": s["icon"],
                "description": s["description"],
                "status": stage_data.get("status", "pending"),
                "duration_ms": stage_data.get("duration_ms", 0),
                "detail": stage_data.get("detail"),
            })

        return {
            "zone": self.zone,
            "started_at": self.started_at.isoformat(),
            "total_duration_ms": total_ms,
            "stages": stages_list,
            "meta": self._meta,
        }


class _StageContext:
    """Context manager for a single pipeline stage."""

    def __init__(self, trace: PipelineTrace, stage_id: str):
        self.trace = trace
        self.stage_id = stage_id
        self._start = None
        self._detail = None

    def set_detail(self, detail: str):
        """Add a detail string shown in the trace replay (e.g. '2,474 rows')."""
        self._detail = detail

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        status = "ok" if exc_type is None else "error"

        entry = {
            "status": status,
            "duration_ms": elapsed_ms,
        }
        if self._detail:
            entry["detail"] = self._detail
        if exc_type:
            entry["detail"] = f"Error: {exc_val}"

        self.trace._stages[self.stage_id] = entry
        return False
