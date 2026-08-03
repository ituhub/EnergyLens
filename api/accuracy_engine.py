"""
EnergyLens — Forecast Accuracy Engine

Compares logged predictions against actual spot prices.
Computes MAE, MAPE, RMSE, directional accuracy, bias.
Supports per-model breakdown and rolling historical windows.

No model training required — works purely from stored data.
"""

import sqlite3
import math
from datetime import datetime, timedelta, timezone
from typing import Optional


class AccuracyEngine:
    def __init__(self, db_path: str = "data/energylens.db"):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Schema Setup ─────────────────────────────────────────────

    def initialize(self):
        """Create forecast_log table if it doesn't exist."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS forecast_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                zone            TEXT NOT NULL,
                forecast_hour   TEXT NOT NULL,
                generated_at    TEXT NOT NULL,
                model_name      TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                confidence      REAL,
                lookback_hours  INTEGER DEFAULT 48,
                UNIQUE(zone, forecast_hour, generated_at, model_name)
            );

            CREATE INDEX IF NOT EXISTS idx_fl_zone_hour
                ON forecast_log(zone, forecast_hour);
            CREATE INDEX IF NOT EXISTS idx_fl_generated
                ON forecast_log(generated_at);
            CREATE INDEX IF NOT EXISTS idx_fl_model
                ON forecast_log(model_name);
        """)
        conn.commit()
        conn.close()

    # ── Logging ──────────────────────────────────────────────────

    def log_forecast(self, zone: str, forecasts: list, per_model: dict,
                     confidence: float = 0.0):
        """
        Persist predictions for accuracy tracking.

        Args:
            zone: Bidding zone (DK1, DK2, etc.)
            forecasts: List of dicts with 'timestamp' and 'price' (ensemble)
            per_model: Dict of model_name -> list of {'timestamp', 'price'}
            confidence: Ensemble confidence score
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = []

        # Log ensemble predictions
        for f in forecasts:
            ts = f.get("timestamp") or f.get("time") or f.get("timestamp_utc")
            price = f.get("price") or f.get("predicted_price") or f.get("price_eur")
            if ts:
                ts = ts.replace("+00:00", "").replace("Z", "")
            if ts and price is not None:
                rows.append((zone, ts, now, "ensemble", float(price),
                             confidence, 48))

        # Log per-model predictions
        for model_name, predictions in per_model.items():
            if isinstance(predictions, (int, float)):
                # Single value per model — log against each forecast hour
                for f in forecasts:
                    ts = f.get("timestamp") or f.get("time") or f.get("timestamp_utc")
                    if ts:
                        ts = ts.replace("+00:00", "").replace("Z", "")
                        rows.append((zone, ts, now, model_name,
                                     float(predictions), None, 48))
            elif isinstance(predictions, list):
                for p in predictions:
                    ts = p.get("timestamp") or p.get("time") or p.get("timestamp_utc")
                    price = p.get("price") or p.get("predicted_price") or p.get("price_eur")
                    if ts and price is not None:
                        rows.append((zone, ts, now, model_name,
                                     float(price), None, 48))

        if not rows:
            return 0

        conn = self._connect()
        conn.executemany(
            """INSERT OR IGNORE INTO forecast_log
               (zone, forecast_hour, generated_at, model_name,
                predicted_price, confidence, lookback_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        conn.commit()
        inserted = conn.total_changes
        conn.close()
        return inserted

    # ── Core Accuracy Computation ────────────────────────────────

    def _get_pairs(self, zone: str, hours_back: int = 24,
                   model_name: str = "ensemble"):
        """
        Join latest forecasts with actual spot prices.
        Returns list of (forecast_hour, predicted, actual) tuples.
        """
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

        # Get the most recent forecast for each hour (latest generated_at)
        # Join rounds spot prices to the hour to handle 15-min data
        query = """
            WITH latest_forecast AS (
                SELECT
                    zone,
                    forecast_hour,
                    predicted_price,
                    generated_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY zone, forecast_hour
                        ORDER BY generated_at DESC
                    ) AS rn
                FROM forecast_log
                WHERE zone = ?
                  AND model_name = ?
                  AND forecast_hour >= ?
            ),
            hourly_prices AS (
                SELECT
                    zone,
                    SUBSTR(valid_time, 1, 14) || '00:00' AS hour_ts,
                    AVG(price_eur_mwh) AS price_eur_mwh
                FROM spot_prices
                GROUP BY zone, SUBSTR(valid_time, 1, 14) || '00:00'
            )
            SELECT
                lf.forecast_hour,
                lf.predicted_price,
                hp.price_eur_mwh AS actual_price
            FROM latest_forecast lf
            JOIN hourly_prices hp
                ON lf.forecast_hour = hp.hour_ts
                AND lf.zone = hp.zone
            WHERE lf.rn = 1
            ORDER BY lf.forecast_hour
        """
        rows = conn.execute(query, (zone, model_name, cutoff)).fetchall()
        conn.close()
        return [(r["forecast_hour"], r["predicted_price"], r["actual_price"])
                for r in rows]

    def _compute_metrics(self, pairs: list) -> dict:
        """Compute accuracy metrics from (hour, predicted, actual) pairs."""
        if not pairs:
            return {
                "mae": None, "mape": None, "rmse": None,
                "directional_accuracy": None, "max_error": None,
                "bias": None, "n_predictions": 0
            }

        errors = []
        pct_errors = []
        sq_errors = []
        direction_correct = 0
        direction_total = 0

        for i, (hour, pred, actual) in enumerate(pairs):
            err = pred - actual
            abs_err = abs(err)
            errors.append(err)
            sq_errors.append(err ** 2)

            if actual != 0:
                pct_errors.append(abs_err / abs(actual) * 100)

            # Directional accuracy: compare to previous hour
            if i > 0:
                prev_actual = pairs[i - 1][2]
                actual_dir = actual - prev_actual
                pred_dir = pred - pairs[i - 1][1]
                if (actual_dir > 0 and pred_dir > 0) or \
                   (actual_dir < 0 and pred_dir < 0) or \
                   (actual_dir == 0 and pred_dir == 0):
                    direction_correct += 1
                direction_total += 1

        n = len(pairs)
        abs_errors = [abs(e) for e in errors]

        return {
            "mae": round(sum(abs_errors) / n, 2),
            "mape": round(sum(pct_errors) / len(pct_errors), 2) if pct_errors else None,
            "rmse": round(math.sqrt(sum(sq_errors) / n), 2),
            "directional_accuracy": round(direction_correct / direction_total, 3) if direction_total > 0 else None,
            "max_error": round(max(abs_errors), 2),
            "bias": round(sum(errors) / n, 2),
            "n_predictions": n
        }

    # ── Public API Methods ───────────────────────────────────────

    def get_latest_accuracy(self, zone: str = "DK1",
                            hours: int = 24) -> dict:
        """
        Accuracy metrics for the most recent N hours.
        Used by AccuracyTracker component.
        """
        # Ensemble accuracy
        ensemble_pairs = self._get_pairs(zone, hours, "ensemble")
        ensemble_metrics = self._compute_metrics(ensemble_pairs)

        # Build prediction/actual pairs for charting
        chart_pairs = []
        for hour, pred, actual in ensemble_pairs:
            err = pred - actual
            chart_pairs.append({
                "hour": hour,
                "predicted": round(pred, 2),
                "actual": round(actual, 2),
                "error": round(err, 2),
                "abs_error": round(abs(err), 2),
                "pct_error": round(abs(err) / abs(actual) * 100, 2) if actual != 0 else None,
                "direction_correct": None  # filled below
            })

        # Directional correctness per pair
        for i in range(1, len(chart_pairs)):
            prev_actual = chart_pairs[i - 1]["actual"]
            actual_dir = chart_pairs[i]["actual"] - prev_actual
            pred_dir = chart_pairs[i]["predicted"] - chart_pairs[i - 1]["predicted"]
            chart_pairs[i]["direction_correct"] = (
                (actual_dir >= 0 and pred_dir >= 0) or
                (actual_dir < 0 and pred_dir < 0)
            )

        # Per-model accuracy
        per_model = self._get_per_model_accuracy(zone, hours)

        return {
            "zone": zone,
            "period_hours": hours,
            "metrics": ensemble_metrics,
            "pairs": chart_pairs,
            "per_model": per_model,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def _get_per_model_accuracy(self, zone: str, hours: int) -> dict:
        """Get accuracy metrics for each individual model."""
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        models = conn.execute(
            """SELECT DISTINCT model_name FROM forecast_log
               WHERE zone = ? AND forecast_hour >= ? AND model_name != 'ensemble'""",
            (zone, cutoff)
        ).fetchall()
        conn.close()

        result = {}
        for row in models:
            model = row["model_name"]
            pairs = self._get_pairs(zone, hours, model)
            metrics = self._compute_metrics(pairs)

            # Detect frozen models (all predictions identical)
            # No matched pairs = unknown, not frozen
            if len(pairs) >= 2:
                prices = [p[1] for p in pairs]
                is_frozen = len(set(round(p, 2) for p in prices)) <= 1
            else:
                is_frozen = False

            result[model] = {**metrics, "frozen": is_frozen}

        return result

    def get_accuracy_history(self, zone: str = "DK1",
                             days: int = 30,
                             window: int = 1) -> dict:
        """
        Rolling daily accuracy over a time window.
        Used by BacktestDashboard component.
        """
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Get all ensemble forecast/actual pairs in the window
        query = """
            WITH latest_forecast AS (
                SELECT
                    zone, forecast_hour, predicted_price, generated_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY zone, forecast_hour
                        ORDER BY generated_at DESC
                    ) AS rn
                FROM forecast_log
                WHERE zone = ? AND model_name = 'ensemble'
                  AND forecast_hour >= ?
            )
            SELECT
                DATE(lf.forecast_hour) AS forecast_date,
                lf.forecast_hour,
                lf.predicted_price,
                sp.price_eur_mwh AS actual_price
            FROM latest_forecast lf
            JOIN spot_prices sp
                ON REPLACE(lf.forecast_hour, 'T', ' ') = REPLACE(sp.valid_time, 'T', ' ')
                AND lf.zone = sp.zone
            WHERE lf.rn = 1
            ORDER BY lf.forecast_hour
        """
        rows = conn.execute(query, (zone, cutoff)).fetchall()
        conn.close()

        if not rows:
            return {
                "zone": zone,
                "window_days": days,
                "daily_metrics": [],
                "per_model_daily": {},
                "summary": {},
                "error_distribution": []
            }

        # Group by date
        daily_groups = {}
        for r in rows:
            date = r["forecast_date"]
            if date not in daily_groups:
                daily_groups[date] = []
            daily_groups[date].append(
                (r["forecast_hour"], r["predicted_price"], r["actual_price"])
            )

        # Compute daily metrics
        daily_metrics = []
        all_abs_errors = []
        for date in sorted(daily_groups.keys()):
            pairs = daily_groups[date]
            metrics = self._compute_metrics(pairs)
            actuals = [p[2] for p in pairs]
            avg_price = sum(actuals) / len(actuals) if actuals else 0
            volatility = (max(actuals) - min(actuals)) if len(actuals) > 1 else 0

            daily_metrics.append({
                "date": date,
                **metrics,
                "avg_price": round(avg_price, 2),
                "price_volatility": round(volatility, 2)
            })

            all_abs_errors.extend([abs(p[1] - p[2]) for p in pairs])

        # Per-model daily metrics
        per_model_daily = self._get_per_model_daily(zone, days)

        # Summary
        all_maes = [d["mae"] for d in daily_metrics if d["mae"] is not None]
        target_mae = 5.0
        best_day = min(daily_metrics, key=lambda d: d["mae"] or 999) if daily_metrics else None
        worst_day = max(daily_metrics, key=lambda d: d["mae"] or 0) if daily_metrics else None

        summary = {
            "overall_mae": round(sum(all_maes) / len(all_maes), 2) if all_maes else None,
            "overall_mape": None,
            "best_day": {"date": best_day["date"], "mae": best_day["mae"]} if best_day else None,
            "worst_day": {"date": worst_day["date"], "mae": worst_day["mae"]} if worst_day else None,
            "days_below_target": sum(1 for m in all_maes if m <= target_mae),
            "total_days": len(daily_metrics),
            "target_mae": target_mae
        }

        # Error distribution (€1 buckets)
        error_dist = {}
        for err in all_abs_errors:
            bucket = f"{int(err)}-{int(err) + 1}"
            error_dist[bucket] = error_dist.get(bucket, 0) + 1

        error_distribution = [
            {"bucket": k, "count": v}
            for k, v in sorted(error_dist.items(),
                               key=lambda x: int(x[0].split("-")[0]))
        ]

        return {
            "zone": zone,
            "window_days": days,
            "daily_metrics": daily_metrics,
            "per_model_daily": per_model_daily,
            "summary": summary,
            "error_distribution": error_distribution
        }

    def _get_per_model_daily(self, zone: str, days: int) -> dict:
        """Get daily MAE per model for trend charts."""
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        models = conn.execute(
            """SELECT DISTINCT model_name FROM forecast_log
               WHERE zone = ? AND forecast_hour >= ? AND model_name != 'ensemble'""",
            (zone, cutoff)
        ).fetchall()

        result = {}
        for row in models:
            model = row["model_name"]
            query = """
                WITH latest_forecast AS (
                    SELECT
                        zone, forecast_hour, predicted_price,
                        ROW_NUMBER() OVER (
                            PARTITION BY zone, forecast_hour
                            ORDER BY generated_at DESC
                        ) AS rn
                    FROM forecast_log
                    WHERE zone = ? AND model_name = ?
                      AND forecast_hour >= ?
                )
                SELECT
                    DATE(lf.forecast_hour) AS forecast_date,
                    AVG(ABS(lf.predicted_price - sp.price_eur_mwh)) AS daily_mae
                FROM latest_forecast lf
                JOIN spot_prices sp
                    ON REPLACE(lf.forecast_hour, 'T', ' ') = REPLACE(sp.valid_time, 'T', ' ')
                    AND lf.zone = sp.zone
                WHERE lf.rn = 1
                GROUP BY forecast_date
                ORDER BY forecast_date
            """
            rows = conn.execute(query, (zone, model, cutoff)).fetchall()
            result[model] = [
                {"date": r["forecast_date"], "mae": round(r["daily_mae"], 2)}
                for r in rows
            ]

        conn.close()
        return result

    def get_model_leaderboard(self, zone: str = "DK1",
                              days: int = 30) -> list:
        """
        Rank all models by cumulative accuracy.
        Used by ModelLeaderboard component.
        """
        conn = self._connect()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        models = conn.execute(
            """SELECT DISTINCT model_name FROM forecast_log
               WHERE zone = ? AND forecast_hour >= ?""",
            (zone, cutoff)
        ).fetchall()
        conn.close()

        leaderboard = []
        for row in models:
            model = row["model_name"]
            pairs = self._get_pairs(zone, days * 24, model)
            metrics = self._compute_metrics(pairs)

            # Detect frozen — need at least 2 pairs to judge
            if len(pairs) >= 2:
                prices = [p[1] for p in pairs]
                is_frozen = len(set(round(p, 2) for p in prices)) <= 1
            else:
                is_frozen = False

            leaderboard.append({
                "model_name": model,
                **metrics,
                "frozen": is_frozen
            })

        # Sort by MAE ascending (best first), frozen models last
        leaderboard.sort(
            key=lambda m: (m["frozen"], m["mae"] if m["mae"] is not None else 999)
        )

        # Add rank
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    def get_log_stats(self) -> dict:
        """Quick stats on how much forecast data we have logged."""
        conn = self._connect()
        stats = conn.execute("""
            SELECT
                COUNT(*) AS total_logs,
                COUNT(DISTINCT forecast_hour) AS unique_hours,
                COUNT(DISTINCT model_name) AS models,
                MIN(forecast_hour) AS earliest,
                MAX(forecast_hour) AS latest,
                COUNT(DISTINCT zone) AS zones
            FROM forecast_log
        """).fetchone()
        conn.close()

        return {
            "total_logs": stats["total_logs"],
            "unique_hours": stats["unique_hours"],
            "models": stats["models"],
            "earliest": stats["earliest"],
            "latest": stats["latest"],
            "zones": stats["zones"]
        }
