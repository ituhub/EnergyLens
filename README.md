# EnergyLens

AI-powered energy market forecasting platform for Nordic power markets.

## Project Structure

```
energylens/
├── config/              # Configuration, environment, constants
│   ├── settings.py      # Central config (DB, API keys, feature flags)
│   └── constants.py     # Market zones, data source URLs, thresholds
├── connectors/          # Data source connectors (one per source)
│   ├── base.py          # Base DataProvider class (cache, retry, health)
│   ├── nordpool.py      # Nord Pool day-ahead & intraday prices
│   ├── entsoe.py        # ENTSO-E transparency platform
│   ├── weather.py       # ECMWF / Open-Meteo weather forecasts
│   ├── remit.py         # REMIT urgent market messages
│   └── energidata.py    # Danish Energi Data Service
├── core/                # Core data infrastructure
│   ├── database.py      # PostgreSQL connection, bitemporal queries
│   ├── raw_archive.py   # Immutable raw storage (local files / GCS)
│   ├── quality_gate.py  # 5-gate data validation (from MarketLens)
│   └── schemas.py       # Data models / Pydantic schemas
├── models/              # ML pipeline
│   ├── features.py      # Energy-specific feature engineering
│   ├── ensemble.py      # 8-model ensemble manager
│   ├── registry.py      # Model versioning and tracking
│   └── explainer.py     # SHAP explainability
├── pipeline/            # Orchestration
│   ├── ingest.py        # Ingestion orchestrator
│   ├── transform.py     # Standardization factory
│   └── predict.py       # Prediction pipeline
├── dashboard/           # Streamlit frontend
│   └── app.py           # Main Streamlit dashboard
├── tests/               # Test suite
│   ├── test_connectors.py
│   ├── test_quality.py
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml   # Local dev (PostgreSQL + app)
└── .env.example         # Environment variables template
```

## Local Development

```bash
# 1. Clone and setup
git clone <repo>
cd energylens
cp .env.example .env  # Add your API keys

# 2. Start PostgreSQL locally
docker-compose up -d db

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run data ingestion
python -m pipeline.ingest

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Deployment Path

1. **Local** → Build and test everything
2. **Streamlit Cloud** → Deploy dashboard for demo/review
3. **GCP** → Full production deployment (Cloud Run + Cloud SQL)
