"""
EnergyLens — GCS Model Sync
Downloads trained model artifacts from GCS bucket to local models/ directory.
Kaggle trains → uploads to GCS → hit /api/forecast/reload → done.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET = os.environ.get(
    "MODEL_BUCKET",
    "energylens-models-project-91e8fbfb-13be-4995-831"
)
GCS_PREFIX = "models/"


def sync_models_from_gcs(local_dir: str = "models", zone: str | None = None) -> dict:
    try:
        from google.cloud import storage
    except ImportError:
        logger.warning("google-cloud-storage not installed — skipping GCS sync")
        return {"status": "skipped", "reason": "google-cloud-storage not installed"}

    try:
        client = storage.Client()
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
