# EnergyLens Phase 3 — ML Integration Guide

## What's New

Three new files integrate the ML ensemble with the existing FastAPI backend and React dashboard:

```
energylens/
├── api/
│   ├── __init__.py              ← NEW
│   ├── main.py                  ← UPDATED (adds /api/forecast endpoints)
│   └── forecast_service.py      ← NEW (bridges ML module ↔ API)
├── dashboard/
│   └── src/
│       └── ForecastChart.jsx    ← NEW (actual vs forecast chart)
├── ml/                          ← Phase 2 (unchanged)
└── models/                      ← Training artifacts land here
```

## New API Endpoints

| Endpoint               | Method | Description                                    |
|------------------------|--------|------------------------------------------------|
| `GET /api/forecast`    | GET    | Generate price forecast (zone, hours params)   |
| `GET /api/forecast/models` | GET | Show loaded model status                    |
| `POST /api/forecast/reload` | POST | Force-reload models after retraining       |

### Example

```bash
# 24-hour DK1 forecast
curl http://localhost:8000/api/forecast?zone=DK1&hours=24

# Check model status
curl http://localhost:8000/api/forecast/models

# Reload after retraining
curl -X POST http://localhost:8000/api/forecast/reload
```

## Setup Steps

### Step 1: Copy the new files

Copy these files into your project at `C:\Users\Nicuma\Downloads\energylens\energylens\`:

- `api/__init__.py`
- `api/main.py` (replaces existing)
- `api/forecast_service.py`
- `dashboard/src/ForecastChart.jsx`

### Step 2: Train the models (one-time)

Before the forecast endpoint works, you need trained models:

```bash
cd C:\Users\Nicuma\Downloads\energylens\energylens

# First, backfill at least 365 days of data
python -m pipeline.ingest --backfill --days 365

# Then train (takes 5-15 minutes depending on data)
python -m ml.run_training --zone DK1

# Optional: also train DK2
python -m ml.run_training --zone DK2
```

After training you'll see files in `models/`:
```
models/
├── DK1_advanced_transformer.pt
├── DK1_cnn_lstm.pt
├── DK1_enhanced_tcn.pt
├── DK1_enhanced_informer.pt
├── DK1_enhanced_nbeats.pt
├── DK1_lstm_gru_ensemble.pt
├── DK1_xgboost.pkl
├── DK1_sklearn_ensemble.pkl
├── DK1_scaler.pkl
└── DK1_config.pkl
```

### Step 3: Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Test the forecast endpoint:
```
http://localhost:8000/api/forecast?zone=DK1&hours=24
```

### Step 4: Add ForecastChart to the dashboard

In your existing `App.jsx`, import and add the component:

```jsx
import ForecastChart from './ForecastChart';

// Inside your App return, add a new section:
<section style={{ padding: 24 }}>
  <ForecastChart zone={selectedZone} hours={24} actualDays={2} />
</section>
```

The component auto-fetches from the API and refreshes every 5 minutes. If no trained models exist yet, it shows a helpful banner instead of erroring.

### Step 5: Verify everything

1. API health: `http://localhost:8000/api/health` should now show `ml_models` section
2. Forecast: `http://localhost:8000/api/forecast?zone=DK1&hours=24` should return predictions
3. Dashboard: `http://localhost:5173` should show the forecast chart with actual + predicted lines

## How It Works

```
User request → /api/forecast?zone=DK1&hours=24
                    │
          ┌─────────▼──────────┐
          │  ForecastService   │
          │                    │
          │  1. Load models    │ ← Lazy-loaded, cached
          │     (from models/) │
          │                    │
          │  2. Pull recent    │ ← SQLite: spot_prices + weather
          │     data           │
          │                    │
          │  3. Build features │ ← features.py: 80+ features
          │                    │
          │  4. Scale + window │ ← RobustScaler, 48h lookback
          │                    │
          │  5. Ensemble       │ ← 8 models → safety rails
          │     predict        │    → weighted average
          │                    │
          │  6. Multi-step     │ ← 24h with per-step clamping
          │     forecast       │
          └────────┬───────────┘
                   │
                   ▼
          JSON response with:
          - 24 hourly price predictions
          - Confidence score (40-88%)
          - Per-model breakdown
          - Price range bounds
```

## After Retraining

When you retrain models (e.g. with more data):

```bash
python -m ml.run_training --zone DK1
```

Then reload without restarting the API:

```bash
curl -X POST http://localhost:8000/api/forecast/reload
```
