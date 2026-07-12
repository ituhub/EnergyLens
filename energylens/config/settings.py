"""
EnergyLens — Central configuration.
Reads from environment variables with sensible defaults for local dev.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ARCHIVE_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for d in [DATA_DIR, RAW_ARCHIVE_DIR, PROCESSED_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Database ───
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "energylens"),
    "user": os.getenv("DB_USER", "energylens"),
    "password": os.getenv("DB_PASSWORD", "energylens_dev"),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ─── API Keys ───
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "")

# ─── Cache Settings ───
CACHE_TTL_MARKET_OPEN = int(os.getenv("CACHE_TTL_MARKET_OPEN", 30))    # seconds
CACHE_TTL_MARKET_CLOSED = int(os.getenv("CACHE_TTL_MARKET_CLOSED", 300))

# ─── Application ───
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
