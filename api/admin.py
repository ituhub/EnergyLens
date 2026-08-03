"""
EnergyLens — Admin Panel API Routes.

Provides admin-only endpoints for:
  - User management (list, activate/deactivate)
  - System health & statistics
  - Forecast log inspection
  - Pipeline activity overview

All endpoints require admin role (set via ENERGYLENS_ADMIN_EMAILS env var).

Usage:
    from api.admin import register_admin_routes
    register_admin_routes(app)
"""

import sqlite3
import logging
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Depends, Query

from api.auth import require_admin, get_all_users, toggle_user_active
from api.logging_config import get_logger

logger = get_logger("energylens.admin")

DB_PATH = "data/energylens.db"


def register_admin_routes(app: FastAPI):
    """Register all /api/admin/* routes on the FastAPI app."""

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    @app.get("/api/admin/users")
    async def admin_list_users(admin=Depends(require_admin)):
        """List all registered users with stats."""
        users = get_all_users()
        return {
            "total_users": len(users),
            "active_users": sum(1 for u in users if u.get("is_active", True)),
            "admin_users": sum(1 for u in users if u.get("role") == "admin"),
            "users": users,
        }

    @app.post("/api/admin/users/{uid}/toggle")
    async def admin_toggle_user(
        uid: str,
        active: bool = Query(..., description="true to activate, false to deactivate"),
        admin=Depends(require_admin),
    ):
        """Activate or deactivate a user account."""
        success = toggle_user_active(uid, active)
        if not success:
            return {"error": f"User {uid} not found"}
        return {"uid": uid, "is_active": active, "status": "updated"}

    # ------------------------------------------------------------------
    # System Stats
    # ------------------------------------------------------------------

    @app.get("/api/admin/stats")
    async def admin_system_stats(admin=Depends(require_admin)):
        """Comprehensive system statistics for the admin dashboard."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        stats = {}

        try:
            # Table counts
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = {}
            for (table,) in cursor:
                if table == "predictions":
                    continue
                count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
                tables[table] = count
            stats["tables"] = tables

            # Data freshness
            row = conn.execute(
                "SELECT MIN(valid_time) as oldest, MAX(valid_time) as newest FROM spot_prices"
            ).fetchone()
            stats["data_range"] = {
                "oldest": row["oldest"],
                "newest": row["newest"],
            }

            # Forecast log stats (last 24h, 7d, 30d)
            for label, hours in [("24h", 24), ("7d", 168), ("30d", 720)]:
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
                try:
                    row = conn.execute(
                        """
                        SELECT COUNT(*) as count,
                               AVG(confidence) as avg_confidence,
                               MIN(confidence) as min_confidence,
                               MAX(confidence) as max_confidence
                        FROM forecast_log
                        WHERE created_at >= ?
                        """,
                        (cutoff,),
                    ).fetchone()
                    stats[f"forecasts_{label}"] = {
                        "count": row["count"],
                        "avg_confidence": round(row["avg_confidence"] or 0, 1),
                        "min_confidence": round(row["min_confidence"] or 0, 1),
                        "max_confidence": round(row["max_confidence"] or 0, 1),
                    }
                except Exception:
                    stats[f"forecasts_{label}"] = {"count": 0}

        finally:
            conn.close()

        # User stats
        users = get_all_users()
        stats["users"] = {
            "total": len(users),
            "active": sum(1 for u in users if u.get("is_active", True)),
            "total_predictions": sum(u.get("prediction_count", 0) for u in users),
        }

        stats["timestamp"] = datetime.now(timezone.utc).isoformat()
        return stats

    # ------------------------------------------------------------------
    # Forecast Logs
    # ------------------------------------------------------------------

    @app.get("/api/admin/forecast-logs")
    async def admin_forecast_logs(
        hours: int = Query(24, ge=1, le=720),
        limit: int = Query(50, ge=1, le=200),
        admin=Depends(require_admin),
    ):
        """Recent forecast log entries with model details."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                """
                SELECT *
                FROM forecast_log
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
            return {
                "count": len(rows),
                "hours": hours,
                "logs": [dict(r) for r in rows],
            }
        except Exception as e:
            return {"error": str(e), "logs": []}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Pipeline Activity
    # ------------------------------------------------------------------

    @app.get("/api/admin/activity")
    async def admin_activity(
        hours: int = Query(24, ge=1, le=168),
        admin=Depends(require_admin),
    ):
        """Recent pipeline activity — refreshes, forecasts, errors."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        activity = []

        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

            # Forecast runs
            try:
                rows = conn.execute(
                    """
                    SELECT forecast_hour, zone, confidence, models_used, created_at
                    FROM forecast_log
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT 100
                    """,
                    (cutoff,),
                ).fetchall()
                for r in rows:
                    activity.append({
                        "type": "forecast",
                        "timestamp": r["created_at"],
                        "zone": r["zone"],
                        "confidence": r["confidence"],
                        "models_used": r["models_used"],
                    })
            except Exception:
                pass

            # Sort by timestamp
            activity.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        finally:
            conn.close()

        return {
            "hours": hours,
            "count": len(activity),
            "activity": activity[:100],
        }

    logger.info("Admin routes registered", extra={"event": "admin_init"})
