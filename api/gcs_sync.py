"""
EnergyLens — GCS Sync (models + database).

Model sync:  Kaggle trains → uploads to GCS → hit /api/forecast/reload → done.
DB sync:     download on cold start, upload after each refresh.
             Integrity verified with SHA-256 checksums to prevent corrupt restores.
"""
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get(
    "MODEL_BUCKET",
    "energylens-models-project-91e8fbfb-13be-4995-831"
)
GCS_PREFIX = "models/"
GCS_DB_PREFIX = "db/"
GCS_DB_BLOB = f"{GCS_DB_PREFIX}energylens.db"
GCS_DB_CHECKSUM_BLOB = f"{GCS_DB_PREFIX}energylens.db.sha256"


def _get_storage_client():
    """Lazy import + client creation."""
    from google.cloud import storage
    return storage.Client()


def _sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# DATABASE SYNC
# ==============================================================================

def upload_db_to_gcs(db_path: str) -> dict:
    """Upload SQLite DB to GCS with SHA-256 checksum.

    Writes both the .db file and a .sha256 sidecar so that
    download_db_from_gcs can verify integrity on restore.
    """
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_BUCKET)

        if not os.path.exists(db_path):
            return {"status": "skipped", "reason": "db file not found"}

        size_kb = os.path.getsize(db_path) / 1024
        checksum = _sha256(db_path)

        # Upload the database
        blob = bucket.blob(GCS_DB_BLOB)
        blob.upload_from_filename(db_path)

        # Upload the checksum sidecar
        checksum_blob = bucket.blob(GCS_DB_CHECKSUM_BLOB)
        checksum_blob.upload_from_string(checksum)

        logger.info(f"DB backup: {size_kb:.0f} KB, sha256={checksum[:16]}...")
        return {"status": "ok", "size_kb": round(size_kb, 1), "sha256": checksum}

    except Exception as e:
        logger.error(f"DB backup failed: {e}")
        return {"status": "error", "reason": str(e)}


def download_db_from_gcs(db_path: str) -> dict:
    """Download SQLite DB from GCS and verify SHA-256 checksum.

    If checksum verification fails, the corrupt download is removed
    and the app starts with a fresh database instead.
    """
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_BUCKET)

        blob = bucket.blob(GCS_DB_BLOB)
        if not blob.exists():
            logger.info("No DB backup in GCS — starting fresh")
            return {"status": "fresh", "reason": "no backup found"}

        # Ensure target directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(db_path)
        size_kb = os.path.getsize(db_path) / 1024

        # Verify checksum if sidecar exists
        checksum_blob = bucket.blob(GCS_DB_CHECKSUM_BLOB)
        if checksum_blob.exists():
            expected = checksum_blob.download_as_text().strip()
            actual = _sha256(db_path)
            if actual != expected:
                logger.error(
                    f"DB CHECKSUM MISMATCH — expected {expected[:16]}... "
                    f"got {actual[:16]}... — removing corrupt download"
                )
                os.remove(db_path)
                return {"status": "error", "reason": "checksum mismatch"}
            logger.info(f"DB restore: {size_kb:.0f} KB, checksum verified ✓")
        else:
            logger.info(f"DB restore: {size_kb:.0f} KB (no checksum sidecar — legacy backup)")

        return {"status": "ok", "size_kb": round(size_kb, 1)}

    except Exception as e:
        logger.error(f"DB restore failed: {e}")
        return {"status": "error", "reason": str(e)}


# ==============================================================================
# MODEL SYNC
# ==============================================================================

def sync_models_from_gcs(local_dir: str = "models", zone: str | None = None) -> dict:
    try:
        client = _get_storage_client()
    except ImportError:
        logger.warning("google-cloud-storage not installed — skipping GCS sync")
        return {"status": "skipped", "reason": "google-cloud-storage not installed"}

    try:
        bucket = client.bucket(GCS_BUCKET)
        blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
        if not blobs:
            logger.warning(f"No models in gs://{GCS_BUCKET}/{GCS_PREFIX}")
            return {"status": "empty"}

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        downloaded = []

        for blob in blobs:
            fname = blob.name.replace(GCS_PREFIX, "")
            if not fname:
                continue
            if zone and not fname.startswith(f"{zone}_"):
                continue
            dest = local_path / fname
            blob.download_to_filename(str(dest))
            size_kb = dest.stat().st_size / 1024
            logger.info(f"  Downloaded {fname} ({size_kb:.1f} KB)")
            downloaded.append(fname)

        logger.info(f"GCS sync: {len(downloaded)} files from gs://{GCS_BUCKET}/{GCS_PREFIX}")
        return {"status": "ok", "downloaded": downloaded, "count": len(downloaded)}

    except Exception as e:
        logger.error(f"GCS sync failed: {e}")
        return {"status": "error", "reason": str(e)}
