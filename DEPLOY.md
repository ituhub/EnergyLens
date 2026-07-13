# EnergyLens Deployment Guide
## Single GCP Cloud Run Service (Backend + Frontend)

---

## Architecture

```
  Client (Enkel Energi)
         │
         ▼
┌───────────────────────────────────┐
│  GCP Cloud Run (FREE TIER)        │
│  europe-north1 (Finland)          │
│                                   │
│  ┌─────────────────────────────┐  │
│  │ FastAPI                     │  │
│  │  /api/*  → backend routes   │  │
│  │  /*      → React static     │  │
│  └─────────────────────────────┘  │
│                                   │
│  SQLite + 8 ML models + data      │
└───────────────────────────────────┘
         ▲
         │ every 2h
┌────────┴────────┐
│ Cloud Scheduler  │
└─────────────────┘
```

One URL. One service. No CORS. No Vercel.

---

## Prerequisites

1. GCP account with billing enabled (free tier covers everything)
2. Google Cloud CLI installed:
   - Windows: `winget install Google.CloudSDK`
   - Or: https://cloud.google.com/sdk/docs/install
3. Node.js installed (for local testing — Docker handles the build)
4. Repo pushed to GitHub (`ituhub/EnergyLens`)

---

## Step 1: Patch Your Code (2 changes)

### 1a. Update API base URL in `dashboard/src/App.jsx`

Since frontend and backend share the same origin, set API_BASE to empty string
in production:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || '';
```

Then use `${API_BASE}/api/...` in all fetch calls:

```javascript
// Before
fetch('http://localhost:8000/api/status')

// After
fetch(`${API_BASE}/api/status`)
```

For local dev, create `dashboard/.env.development`:
```
VITE_API_URL=http://localhost:8000
```

### 1b. Mount static files in `energylens/api/main.py`

Add these imports at the top:

```python
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
```

Add this AFTER all your `@app.get`/`@app.post` API routes (order matters —
API routes must be registered before the catch-all):

```python
# Serve React static build
STATIC_DIR = Path("/app/static")

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    async def serve_react(path: str):
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
```

No CORS middleware needed — same origin.

### 1c. Copy deployment files to repo root

```
energylens/                   ← repo root
├── Dockerfile                ← NEW (multi-stage: Node + Python)
├── .dockerignore             ← NEW
├── requirements.txt          ← NEW (or update existing)
├── energylens/
│   ├── api/main.py           ← PATCHED (static file mount)
│   ├── ml/
│   ├── models/
│   └── data/energylens.db
├── dashboard/
│   ├── .env.development      ← NEW (local dev only)
│   └── src/App.jsx           ← PATCHED (API_BASE)
└── auto_refresh.py
```

### 1d. Push

```bash
cd C:\Users\Nicuma\Downloads\energylens
git add .
git commit -m "Unified Cloud Run deployment — single service"
git push origin main
```

---

## Step 2: One-Time GCP Setup

```bash
gcloud auth login

# Create project (or use existing)
gcloud projects create energylens-prod --name="EnergyLens"
gcloud config set project energylens-prod

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

---

## Step 3: Deploy

```bash
cd C:\Users\Nicuma\Downloads\energylens

gcloud run deploy energylens \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300
```

First build: ~10-12 min (Node + Python + PyTorch).
Subsequent deploys: ~4 min.

Output:

```
Service URL: https://energylens-XXXXXXXXXX-lz.a.run.app
```

Test:
- Dashboard: https://energylens-XXXXXXXXXX-lz.a.run.app
- API: https://energylens-XXXXXXXXXX-lz.a.run.app/api/status

---

## Step 4: Auto-Refresh with Cloud Scheduler

Add a refresh endpoint to `energylens/api/main.py`:

```python
import subprocess

@app.post("/api/refresh")
async def trigger_refresh():
    result = subprocess.run(
        ["python", "auto_refresh.py"],
        capture_output=True, text=True, timeout=120
    )
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "stdout": result.stdout[-500:] if result.stdout else "",
    }
```

Then set up the cron:

```bash
# Enable Scheduler API
gcloud services enable cloudscheduler.googleapis.com

# Create service account
gcloud iam service-accounts create scheduler-sa \
  --display-name="Cloud Scheduler SA"

# Grant invoke permission
gcloud run services add-iam-policy-binding energylens \
  --region europe-north1 \
  --member="serviceAccount:scheduler-sa@energylens-prod.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

# Create cron job (every 2 hours)
gcloud scheduler jobs create http energylens-refresh \
  --location europe-north1 \
  --schedule "0 */2 * * *" \
  --uri "https://energylens-XXXXXXXXXX-lz.a.run.app/api/refresh" \
  --http-method POST \
  --oidc-service-account-email scheduler-sa@energylens-prod.iam.gserviceaccount.com
```

---

## Step 5: Share With Client

Send this link:

```
https://energylens-XXXXXXXXXX-lz.a.run.app
```

Done. One URL, everything included.

---

## Custom Domain (Optional)

```bash
# Map your domain
gcloud run domain-mappings create \
  --service energylens \
  --domain app.energylens.dk \
  --region europe-north1
```

Then add the DNS records shown in the output to your domain registrar.
Client sees: `https://app.energylens.dk`

---

## Cost Summary

| Service            | Tier       | Monthly Cost | Limits                           |
|--------------------|------------|--------------|----------------------------------|
| Cloud Run          | Free tier  | $0           | 2M requests, 360K GB-sec         |
| Cloud Build        | Free tier  | $0           | 120 build-min/day                |
| Artifact Registry  | Free tier  | $0           | 500 MB storage                   |
| Cloud Scheduler    | Free tier  | $0           | 3 jobs free                      |
| **Total**          |            | **$0/mo**    |                                  |

---

## Redeploy After Changes

```bash
cd C:\Users\Nicuma\Downloads\energylens
gcloud run deploy energylens --source . --region europe-north1
```

That's it. One command.

---

## Troubleshooting

**Build times out**
PyTorch is large. If Cloud Build exceeds the default timeout:
```bash
gcloud run deploy energylens --source . --region europe-north1 --timeout 600
```

**React page shows blank**
Check that the static mount in main.py comes AFTER all API routes.
The catch-all `/{path:path}` must be the last registered route.

**API returns 404 for /api/status**
Make sure your API routes use the `/api/` prefix consistently.

**"Permission denied" on deploy**
```bash
gcloud auth login
gcloud config set project energylens-prod
```

**Models return stale predictions**
Retrain on Kaggle with recent data, update models/, redeploy:
```bash
gcloud run deploy energylens --source . --region europe-north1
```
