"""
EnergyLens — Structured Logging Configuration.

JSON-formatted logs for Cloud Run (searchable in Cloud Logging).
Adds request tracing, user context, and pipeline stage timing.

Usage:
    from api.logging_config import setup_logging, get_logger
    setup_logging()
    logger = get_logger("energylens.api")
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone

# Context vars for request-scoped metadata
_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_user_id: ContextVar[str] = ContextVar("user_id", default="anonymous")


def set_request_context(request_id: str, user_id: str = "anonymous"):
    """Set per-request context (called by auth middleware)."""
    _request_id.set(request_id)
    _user_id.set(user_id)


class CloudRunJsonFormatter(logging.Formatter):
    """
    Structured JSON formatter for Cloud Run / Cloud Logging.
    Maps Python log levels to Cloud Logging severity.
    """

    SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self.SEVERITY_MAP.get(record.levelname, "DEFAULT"),
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id.get("-"),
            "user_id": _user_id.get("anonymous"),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields (pipeline stage, duration, etc.)
        for key in ("stage", "duration_ms", "zone", "models_loaded",
                     "confidence", "event", "email", "role", "endpoint"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO"):
    """
    Configure structured logging for the entire application.
    Call once at app startup (before any loggers are used).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove default handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudRunJsonFormatter())
    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger (always uses the configured formatter)."""
    return logging.getLogger(name)


class PipelineTimer:
    """
    Context manager for timing pipeline stages.
    Logs start and end with duration, and collects into a trace dict.

    Usage:
        trace = {}
        with PipelineTimer("data_loading", trace, logger):
            load_data()
        # trace == {"data_loading": {"status": "ok", "duration_ms": 142}}
    """

    def __init__(self, stage_name: str, trace: dict, logger: logging.Logger):
        self.stage = stage_name
        self.trace = trace
        self.logger = logger
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        self.logger.info(
            f"Pipeline stage started: {self.stage}",
            extra={"stage": self.stage, "event": "stage_start"},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)
        status = "ok" if exc_type is None else "error"

        self.trace[self.stage] = {
            "status": status,
            "duration_ms": elapsed_ms,
        }

        if exc_type:
            self.trace[self.stage]["error"] = str(exc_val)
            self.logger.error(
                f"Pipeline stage failed: {self.stage} ({elapsed_ms}ms) — {exc_val}",
                extra={"stage": self.stage, "duration_ms": elapsed_ms, "event": "stage_error"},
            )
        else:
            self.logger.info(
                f"Pipeline stage complete: {self.stage} ({elapsed_ms}ms)",
                extra={"stage": self.stage, "duration_ms": elapsed_ms, "event": "stage_complete"},
            )

        return False  # Don't suppress exceptions
