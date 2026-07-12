"""
EnergyLens — Market constants, zone definitions, and data source URLs.
"""

# ─── Nordic Bidding Zones ───
# Denmark has two price zones
BIDDING_ZONES = {
    "DK1": {"name": "Denmark West (Jutland)", "entsoe_code": "10YDK-1--------W"},
    "DK2": {"name": "Denmark East (Zealand)", "entsoe_code": "10YDK-2--------M"},
    "NO1": {"name": "Norway South-East", "entsoe_code": "10YNO-1--------2"},
    "NO2": {"name": "Norway South-West", "entsoe_code": "10YNO-2--------T"},
    "SE3": {"name": "Sweden Stockholm", "entsoe_code": "10Y1001A1001A46L"},
    "SE4": {"name": "Sweden Malmö", "entsoe_code": "10Y1001A1001A47J"},
    "FI":  {"name": "Finland", "entsoe_code": "10YFI-1--------U"},
}

# Start with Danish zones, expand later
ACTIVE_ZONES = ["DK1", "DK2"]

# ─── Data Source URLs ───
NORDPOOL_BASE_URL = "https://dataportal-api.nordpoolgroup.com/api"
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
ENERGI_DATA_SERVICE_URL = "https://api.energidataservice.dk"
OPEN_METEO_URL = "https://api.open-meteo.com/v1"

# ─── Weather Stations (for Danish wind/solar forecasts) ───
# Coordinates for key Danish weather reference points
WEATHER_LOCATIONS = {
    "DK1_wind": {"lat": 55.86, "lon": 8.62, "name": "Esbjerg (West Jutland)"},
    "DK1_solar": {"lat": 56.16, "lon": 10.21, "name": "Aarhus"},
    "DK2_wind": {"lat": 55.63, "lon": 12.08, "name": "Roskilde (Zealand)"},
    "DK2_solar": {"lat": 55.68, "lon": 12.57, "name": "Copenhagen"},
}

# ─── Market Timing (CET/CEST) ───
# Nord Pool day-ahead auction results published at ~12:42 CET
DAY_AHEAD_PUBLISH_HOUR = 12
DAY_AHEAD_PUBLISH_MINUTE = 42

# Intraday continuous trading: 15-minute products
INTRADAY_RESOLUTION_MINUTES = 15

# ─── Quality Gate Thresholds ───
QUALITY_GATES = {
    "completeness": {
        "min_fields_present": 0.95,  # 95% of expected fields must be present
    },
    "range": {
        "price_min_eur_mwh": -500,   # Negative prices happen in energy markets
        "price_max_eur_mwh": 5000,   # Circuit breaker above this
        "wind_speed_max_ms": 50,     # m/s — above this is measurement error
        "temperature_min_c": -40,
        "temperature_max_c": 50,
    },
    "freshness": {
        "spot_price_max_age_seconds": 120,     # 2 minutes
        "weather_max_age_seconds": 7200,       # 2 hours
        "generation_max_age_seconds": 3600,    # 1 hour
    },
    "anomaly": {
        "price_zscore_threshold": 4.0,   # Flag prices > 4 std devs from rolling mean
        "volume_zscore_threshold": 3.5,
    },
}

# ─── Feature Engineering ───
ROLLING_WINDOWS = [6, 12, 24, 48, 168]  # hours: 6h, 12h, 1d, 2d, 1 week
PRICE_LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]  # lag features

# ─── Model Configuration ───
FORECAST_HORIZONS = {
    "day_ahead": 24,       # 24 hourly predictions
    "intraday": 4,         # 4 hours ahead in 15-min blocks
}

ENSEMBLE_MODELS = [
    "transformer",
    "cnn_lstm",
    "tcn",
    "lstm_attention",
    "xgboost",
    "lightgbm",
    "gru",
    "bilstm",
]

# ─── Energy Market Regimes ───
MARKET_REGIMES = [
    "high_wind",           # Wind generation > 60% of capacity
    "low_wind",            # Wind generation < 15% of capacity
    "solar_peak",          # Solar generation > 40% of capacity (summer midday)
    "demand_spike",        # Demand > 95th percentile
    "negative_price",      # Spot price < 0
    "normal",              # None of the above
]
