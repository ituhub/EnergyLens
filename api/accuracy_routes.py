"""
EnergyLens — Accuracy & Explainability API Routes

Add these to your existing FastAPI app in main.py:
    from api.accuracy_routes import register_accuracy_routes
    register_accuracy_routes(app)
"""

from fastapi import FastAPI, Query, HTTPException
from api.accuracy_engine import AccuracyEngine
from api.shap_engine import ShapEngine


def register_accuracy_routes(app: FastAPI):
    """Register all accuracy and SHAP endpoints on the FastAPI app."""

    accuracy = AccuracyEngine()
    accuracy.initialize()  # Create forecast_log table

    shap_engine = ShapEngine()

    # ═══════════════════════════════════════════════════════════════
    # ACCURACY ENDPOINTS
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/accuracy/latest")
    async def accuracy_latest(
        zone: str = Query("DK1", description="Bidding zone"),
        hours: int = Query(24, ge=1, le=168, description="Hours to look back")
    ):
        """
        Latest accuracy metrics — predicted vs actual for recent hours.
        Powers the AccuracyTracker component.
        """
        try:
            result = accuracy.get_latest_accuracy(zone, hours)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/accuracy/history")
    async def accuracy_history(
        zone: str = Query("DK1"),
        days: int = Query(30, ge=1, le=365, description="Days of history"),
        window: int = Query(1, ge=1, le=30, description="Rolling window size")
    ):
        """
        Historical accuracy over time — daily MAE trends, best/worst days.
        Powers the BacktestDashboard component.
        """
        try:
            result = accuracy.get_accuracy_history(zone, days, window)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/accuracy/models")
    async def accuracy_models(
        zone: str = Query("DK1"),
        days: int = Query(30, ge=1, le=365)
    ):
        """
        Model leaderboard — rank all models by accuracy.
        """
        try:
            result = accuracy.get_model_leaderboard(zone, days)
            return {"zone": zone, "days": days, "leaderboard": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/accuracy/stats")
    async def accuracy_stats():
        """Quick stats on forecast logging coverage."""
        try:
            return accuracy.get_log_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ═══════════════════════════════════════════════════════════════
    # SHAP / EXPLAINABILITY ENDPOINTS
    # ═══════════════════════════════════════════════════════════════

    @app.get("/api/explain")
    async def explain_forecast(
        zone: str = Query("DK1", description="Bidding zone")
    ):
        """
        SHAP explainability for the current forecast.
        Returns feature importance rankings and which features
        are pushing the price up vs down.
        """
        try:
            result = shap_engine.compute_shap(zone)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/explain/groups")
    async def explain_groups():
        """
        Feature group definitions and descriptions.
        Used by the frontend to render group labels and tooltips.
        """
        return shap_engine.get_feature_groups()

    # ═══════════════════════════════════════════════════════════════
    # FORECAST LOGGING HOOK
    # ═══════════════════════════════════════════════════════════════

    # To enable automatic forecast logging, add this to your
    # existing /api/forecast endpoint or /api/refresh handler:
    #
    #   from api.accuracy_engine import AccuracyEngine
    #   accuracy_engine = AccuracyEngine()
    #   accuracy_engine.initialize()
    #
    #   # After generating forecast:
    #   accuracy_engine.log_forecast(
    #       zone=zone,
    #       forecasts=forecast_result["forecasts"],
    #       per_model=forecast_result["per_model"],
    #       confidence=forecast_result["confidence"]
    #   )

    return accuracy, shap_engine
