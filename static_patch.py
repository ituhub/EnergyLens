# ────────────────────────────────────────────────────────
# STATIC FILES PATCH — Add to energylens/api/main.py
#
# This serves the React build from the same service.
# No CORS needed — frontend and backend share one origin.
# ────────────────────────────────────────────────────────

# Add these imports at the top:
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ──────────────────────────────────────────────
# Add this AFTER all your @app.get/post routes:
# ──────────────────────────────────────────────

# Serve React static build
STATIC_DIR = Path("/app/static")

if STATIC_DIR.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Catch-all: serve index.html for any non-API route (React Router)
    @app.get("/{path:path}")
    async def serve_react(path: str):
        # If file exists in static dir, serve it (favicon, manifest, etc.)
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html (React handles routing)
        return FileResponse(STATIC_DIR / "index.html")
