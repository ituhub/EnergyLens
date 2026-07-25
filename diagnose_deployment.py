#!/usr/bin/env python3
"""
EnergyLens Deployment Diagnostic v3 — Jul 2026
================================================
Loads model files from GCS (streaming into memory) or local models/ dir.
Inspects the 3 blocking deployment issues:
  1. EnhancedInformer state_dict key mismatch
  2. XGBoost pickle class name mismatch
  3. Catastrophic filter threshold
  + Config/feature alignment & file inventory

Run from repo root:
    python diagnose_deployment.py
    python diagnose_deployment.py --issue 1
    python diagnose_deployment.py --local        # use local models/ dir
"""

import argparse
import sys
import pickle
import re
import tempfile
from pathlib import Path
from collections import OrderedDict

# ─────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────
OK   = "\033[92m✓\033[0m"
WARN = "\033[93m⚠\033[0m"
FAIL = "\033[91m✗\033[0m"
BOLD = "\033[1m"
RST  = "\033[0m"

def header(title):
    print(f"\n{BOLD}{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}{RST}\n")

# ─────────────────────────────────────────────
# Naming convention: GCS uses DK1_<model>.ext
# ─────────────────────────────────────────────
MODEL_FILES = {
    "advanced_transformer": "DK1_advanced_transformer.pt",
    "cnn_lstm":             "DK1_cnn_lstm.pt",
    "enhanced_informer":    "DK1_enhanced_informer.pt",
    "enhanced_nbeats":      "DK1_enhanced_nbeats.pt",
    "enhanced_tcn":         "DK1_enhanced_tcn.pt",
    "lstm_gru_ensemble":    "DK1_lstm_gru_ensemble.pt",
    "sklearn_ensemble":     "DK1_sklearn_ensemble.pkl",
    "xgboost":              "DK1_xgboost.pkl",
}

CONFIG_FILES = {
    "config_pkl":     "DK1_config.pkl",
    "config_json":    "DK1_config.json",
    "scaler":         "DK1_scaler.pkl",
    "target_scaler":  "DK1_target_scaler.pkl",
}

# ─────────────────────────────────────────────
# Model loader — GCS or local
# ─────────────────────────────────────────────
DEFAULT_BUCKET = "energylens-models-project-91e8fbfb-13be-4995-831"
GCS_PREFIX     = "models/"

_gcs_client = None

def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


class ModelLoader:
    """Unified loader: GCS-streaming or local filesystem."""

    def __init__(self, local_dir=None, bucket=None):
        self.local_dir = Path(local_dir) if local_dir else None
        self.bucket_name = bucket

    @property
    def source_label(self):
        if self.local_dir:
            return f"local: {self.local_dir}"
        return f"gs://{self.bucket_name}/{GCS_PREFIX}"

    def _get_bytes(self, filename: str) -> bytes | None:
        if self.local_dir:
            path = self.local_dir / filename
            if not path.exists():
                return None
            return path.read_bytes()
        else:
            client = get_gcs_client()
            blob = client.bucket(self.bucket_name).blob(f"{GCS_PREFIX}{filename}")
            if not blob.exists():
                return None
            return blob.download_as_bytes()

    def load_pickle(self, filename: str):
        data = self._get_bytes(filename)
        if data is None:
            return None
        return pickle.loads(data)

    def load_torch(self, filename: str):
        import torch
        data = self._get_bytes(filename)
        if data is None:
            return None
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            return torch.load(tmp.name, map_location="cpu", weights_only=False)

    def get_raw_bytes(self, filename: str) -> bytes | None:
        return self._get_bytes(filename)

    def list_files(self) -> list[dict]:
        if self.local_dir:
            results = []
            if self.local_dir.exists():
                for p in sorted(self.local_dir.iterdir()):
                    if p.is_file():
                        results.append({"name": p.name, "size": p.stat().st_size})
            return results
        else:
            client = get_gcs_client()
            results = []
            for blob in client.bucket(self.bucket_name).list_blobs(prefix=GCS_PREFIX):
                name = blob.name.replace(GCS_PREFIX, "")
                if name:
                    results.append({"name": name, "size": blob.size or 0})
            return results


# ═══════════════════════════════════════════════
# ISSUE 1 — EnhancedInformer state_dict keys
# ═══════════════════════════════════════════════
def diagnose_informer(loader: ModelLoader):
    header("ISSUE 1: EnhancedInformer state_dict key naming")

    filename = MODEL_FILES["enhanced_informer"]
    print(f"  Source: {loader.source_label}{filename}")

    try:
        import torch
    except ImportError:
        print(f"  {FAIL} torch not installed — pip install torch --break-system-packages")
        return

    print(f"  Loading checkpoint...")
    checkpoint = loader.load_torch(filename)
    if checkpoint is None:
        print(f"  {FAIL} File not found!")
        return

    # Extract state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
        layout = "dict → 'model_state_dict'"
    elif isinstance(checkpoint, OrderedDict) or (isinstance(checkpoint, dict) and any("weight" in k for k in checkpoint)):
        sd = checkpoint
        layout = "bare state_dict"
    else:
        layout = f"dict with keys: {list(checkpoint.keys())[:10]}"
        sd = checkpoint.get("state_dict", checkpoint)

    print(f"  Layout: {layout}")

    all_keys = list(sd.keys()) if isinstance(sd, dict) else []
    print(f"  Total keys: {len(all_keys)}")

    # ── Short (notebook) vs Long (production) ──
    SHORT_TO_LONG = {
        "proj":  "input_projection",
        "enc":   "encoder",
        "out":   "output_projection",
        "norm":  "input_norm",
        "pos":   "pos_encoding",
    }

    short_found = set()
    long_found = set()
    all_prefixes = set()

    for key in all_keys:
        top = key.split(".")[0]
        all_prefixes.add(top)
        if top in SHORT_TO_LONG:
            short_found.add(top)
        if top in SHORT_TO_LONG.values():
            long_found.add(top)

    print(f"  Top-level prefixes: {sorted(all_prefixes)}")
    print(f"\n  First 20 keys:")
    for k in all_keys[:20]:
        print(f"    • {k}")
    if len(all_keys) > 20:
        print(f"    ... ({len(all_keys)-20} more)")

    print()
    if short_found and not long_found:
        print(f"  {FAIL} CONFIRMED: checkpoint uses SHORT names (notebook)")
        print(f"     Found:    {sorted(short_found)}")
        print(f"     Expected: {sorted(SHORT_TO_LONG.values())}")
        print()
        print(f"  {BOLD}FIX — remap keys at load time in ml/models.py or ml/training.py:{RST}")
        print(f"     KEY_REMAP = {dict(sorted(SHORT_TO_LONG.items()))}")
        print(f"     def remap_state_dict(sd):")
        print(f"         return {{")
        print(f"             '.'.join([KEY_REMAP.get(parts[0], parts[0])] + parts[1:]): v")
        print(f"             for k, v in sd.items()")
        print(f"             for parts in [k.split('.')]")
        print(f"         }}")
    elif long_found and not short_found:
        print(f"  {OK} checkpoint uses LONG names — matches production ml/models.py")
    elif short_found and long_found:
        print(f"  {WARN} MIXED — short: {sorted(short_found)}, long: {sorted(long_found)}")
    else:
        print(f"  {WARN} Neither short nor long prefixes detected")
        print(f"     Prefixes found: {sorted(all_prefixes)}")
        print(f"     The model architecture may differ from what's in ml/models.py")


# ═══════════════════════════════════════════════
# ISSUE 2 — XGBoost pickle class name
# ═══════════════════════════════════════════════
def diagnose_xgboost(loader: ModelLoader):
    header("ISSUE 2: XGBoost pickle class name mismatch")

    filename = MODEL_FILES["xgboost"]
    print(f"  Source: {loader.source_label}{filename}")
    print(f"  Loading raw bytes...")

    raw = loader.get_raw_bytes(filename)
    if raw is None:
        print(f"  {FAIL} File not found!")
        return

    print(f"  Size: {len(raw):,} bytes")

    # ── Scan pickle stream for class references ──
    class_refs = []
    idx = 0
    while idx < len(raw):
        byte = raw[idx:idx+1]
        if byte == b'c':  # GLOBAL opcode
            try:
                end_mod = raw.index(b'\n', idx + 1)
                end_cls = raw.index(b'\n', end_mod + 1)
                module = raw[idx+1:end_mod].decode('utf-8', errors='replace')
                cls = raw[end_mod+1:end_cls].decode('utf-8', errors='replace')
                class_refs.append(f"{module}.{cls}")
                idx = end_cls + 1
                continue
            except ValueError:
                pass
        elif byte == b'\x8c':  # SHORT_BINUNICODE
            length = raw[idx+1]
            s = raw[idx+2:idx+2+length].decode('utf-8', errors='replace')
            if any(kw in s for kw in ['Model', 'XGBoost', 'xgboost', 'Scaler', 'Booster', 'Pipeline']):
                class_refs.append(s)
            idx += 2 + length
            continue
        idx += 1

    class_refs = list(dict.fromkeys(class_refs))

    print(f"\n  Class references found in pickle:")
    for c in class_refs:
        print(f"    • {c}")

    has_short = any("XGBoostModel" in c and "TimeSeries" not in c for c in class_refs)
    has_long  = any("XGBoostTimeSeriesModel" in c for c in class_refs)

    print()
    if has_short and not has_long:
        print(f"  {FAIL} CONFIRMED: pickle expects 'XGBoostModel' (notebook class)")
        print(f"     Production unpickler looks for 'XGBoostTimeSeriesModel'")
        print()
        print(f"  {BOLD}FIX — in ml/training.py KaggleUnpickler.find_class():{RST}")
        print(f"     if name == 'XGBoostModel':")
        print(f"         name = 'XGBoostTimeSeriesModel'")
    elif has_long:
        print(f"  {OK} pickle uses 'XGBoostTimeSeriesModel' — matches production")
    else:
        print(f"  {WARN} No XGBoostModel / XGBoostTimeSeriesModel found in pickle stream")

    # ── Try full unpickle ──
    print(f"\n  Full unpickle test:")
    try:
        obj = pickle.loads(raw)
        print(f"    {OK} Loaded OK — type: {type(obj).__name__}")
        if hasattr(obj, '__dict__'):
            print(f"    Attributes: {list(obj.__dict__.keys())[:10]}")
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"    {FAIL} {type(e).__name__}: {e}")
    except Exception as e:
        print(f"    {WARN} {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════
# ISSUE 3 — Catastrophic filter threshold
# ═══════════════════════════════════════════════
def diagnose_catastrophic_filter(repo_root: Path):
    header("ISSUE 3: Catastrophic filter threshold (50%)")

    ensemble_path = repo_root / "ml" / "ensemble.py"
    if not ensemble_path.exists():
        candidates = list(repo_root.rglob("ensemble.py"))
        ensemble_path = candidates[0] if candidates else None
        if not ensemble_path:
            print(f"  {FAIL} No ensemble.py found")
            return

    source = ensemble_path.read_text()
    lines = source.split('\n')

    print(f"  File: {ensemble_path}")

    # Show the core filter logic
    for i, line in enumerate(lines):
        if 'deviation > 0.5' in line:
            start, end = max(0, i-5), min(len(lines), i+8)
            print(f"\n  Core filter (lines {start+1}–{end}):")
            for j in range(start, end):
                marker = " >>>" if j == i else "    "
                print(f"  {marker} {j+1:4d} │ {lines[j].rstrip()}")
            break

    # Show the near-zero bypass
    for i, line in enumerate(lines):
        if 'abs(current_price)' in line and '5.0' in line:
            print(f"\n  Near-zero bypass (line {i+1}):")
            for j in range(max(0,i-1), min(len(lines),i+3)):
                print(f"       {j+1:4d} │ {lines[j].rstrip()}")
            break

    # ── Simulation at multiple prices ──
    model_preds = {
        "lstm_gru_ensemble":   -3.52,
        "enhanced_nbeats":      8.03,
        "xgboost":             20.33,
        "enhanced_tcn":        22.50,
        "advanced_transformer": 25.00,
        "cnn_lstm":            28.00,
        "enhanced_informer":   30.00,
    }

    print(f"\n  {BOLD}Filter simulation across price levels:{RST}\n")
    print(f"  {'Price':>7}  {'Survive':>7}  {'Killed':>6}  {'Status'}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*30}")

    for price in [3.0, 5.0, 10.0, 20.0, 33.0, 50.0, 80.0, 150.0]:
        if abs(price) <= 5.0:
            n_survive = len(model_preds)
            note = "bypass (≤€5)"
        else:
            n_survive = sum(
                1 for p in model_preds.values()
                if abs(p - price) / abs(price) <= 0.50
            )
            note = ""
        n_killed = len(model_preds) - n_survive
        icon = OK if n_survive >= 3 else (WARN if n_survive >= 1 else FAIL)
        print(f"  €{price:5.1f}  {n_survive:>7}  {n_killed:>6}  {icon} {note}")

    print(f"\n  {BOLD}Per-model detail at €33:{RST}\n")
    price = 33.0
    print(f"  {'Model':<28} {'Pred':>8} {'Dev':>7} {'Result':>8}")
    print(f"  {'─'*28} {'─'*8} {'─'*7} {'─'*8}")
    for name, pred in sorted(model_preds.items(), key=lambda x: abs(x[1]-price)):
        dev = abs(pred - price) / abs(price) * 100
        status = "PASS" if dev <= 50 else "KILLED"
        icon = OK if status == "PASS" else FAIL
        print(f"  {name:<28} €{pred:>6.2f} {dev:>5.1f}% {icon} {status}")


# ═══════════════════════════════════════════════
# BONUS — Config & file inventory
# ═══════════════════════════════════════════════
def diagnose_config(loader: ModelLoader, repo_root: Path):
    header("BONUS: Config, features & file inventory")

    # ── Inventory ──
    print(f"  {BOLD}Files at {loader.source_label}{RST}\n")
    files = loader.list_files()
    file_names = set()
    for f in files:
        size_kb = f['size'] / 1024
        print(f"    {OK} {f['name']} ({size_kb:.0f} KB)")
        file_names.add(f['name'])

    # Check expected files
    all_expected = {**MODEL_FILES, **CONFIG_FILES}
    missing = [v for v in all_expected.values() if v not in file_names]
    extra = file_names - set(all_expected.values())

    if missing:
        print(f"\n  Expected but missing:")
        for m in missing:
            print(f"    {FAIL} {m}")
    if extra:
        print(f"\n  Extra/unexpected files:")
        for e in sorted(extra):
            print(f"    {WARN} {e}")

    # ── Config ──
    print(f"\n  {BOLD}DK1_config.pkl:{RST}")
    config = loader.load_pickle(CONFIG_FILES["config_pkl"])
    if config is None:
        print(f"  {FAIL} Not found")
        return

    print(f"  Keys: {list(config.keys())}")

    # Try all feature key names
    feature_key = next(
        (k for k in ['features', 'feature_columns', 'used_features', 'feature_names']
         if config.get(k)), None
    )
    features = config.get(feature_key, []) if feature_key else []
    n_features = config.get('n_features', config.get('feature_count', '?'))

    print(f"\n  feature list key:  '{feature_key}' → {len(features)} items")
    print(f"  n_features:        {n_features}")
    print(f"  seq_len:           {config.get('seq_len', '?')}")
    print(f"  use_differencing:  {config.get('use_differencing', 'NOT SET')}")
    print(f"  price_range:       {config.get('price_range', 'NOT SET')}")
    print(f"  trained_on:        {config.get('trained_on', config.get('trained_at', '?'))}")
    print(f"  sample_weighting:  {config.get('sample_weighting', '?')}")
    print(f"  device:            {config.get('device', '?')}")
    print(f"  training_rows:     {config.get('training_rows', config.get('data_rows', '?'))}")

    if len(features) > 0:
        print(f"\n  Feature list (first 15):")
        for f in features[:15]:
            print(f"    • {f}")
        if len(features) == 97:
            print(f"\n  {OK} 97 features — matches expectation")
        else:
            print(f"\n  {WARN} Expected 97, got {len(features)}")
    elif isinstance(n_features, int):
        print(f"\n  {WARN} Feature LIST is empty, but n_features={n_features}")
        print(f"     Config stores the count, not the names.")
        print(f"     Production features.py must generate exactly {n_features} columns.")
    else:
        print(f"\n  {FAIL} No feature information in config")

    # Ensemble weights
    weights = config.get('ensemble_weights', {})
    if weights:
        print(f"\n  Ensemble weights (from training):")
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {name:<30} {w:.4f}")

    # Models info
    models_info = config.get('models', {})
    if models_info:
        print(f"\n  Model registry in config:")
        for name, info in models_info.items():
            if isinstance(info, dict):
                params = info.get('params', '?')
                print(f"    {name:<30} params: {params}")
            else:
                print(f"    {name:<30} {info}")

    # ── Cross-check features.py ──
    features_py = repo_root / "ml" / "features.py"
    if features_py.exists():
        src = features_py.read_text()
        sentinels = ["wind_power_proxy", "heating_degree", "spike_indicator", "temp_peak_interaction"]
        missing_s = [s for s in sentinels if s not in src]
        if missing_s:
            print(f"\n  {WARN} ml/features.py missing sentinel features: {missing_s}")
        else:
            print(f"\n  {OK} ml/features.py has all sentinel features")

    # ── Also check if DK1_config.json exists and differs ──
    json_config = loader.get_raw_bytes(CONFIG_FILES["config_json"])
    if json_config:
        import json
        try:
            jc = json.loads(json_config)
            json_feats = jc.get('features', jc.get('used_features', []))
            print(f"\n  DK1_config.json also present — {len(json_feats)} features in JSON")
            if len(json_feats) == 97:
                print(f"  {OK} JSON config has 97 features (may be more reliable than pkl)")
                print(f"  First 5: {json_feats[:5]}")
        except Exception as e:
            print(f"\n  {WARN} DK1_config.json parse error: {e}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="EnergyLens deployment diagnostics v3")
    parser.add_argument("--local", action="store_true",
                        help="Use local models/ dir instead of GCS")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Local models directory (with --local)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--issue", type=int, choices=[1, 2, 3])
    args = parser.parse_args()

    if args.local:
        loader = ModelLoader(local_dir=args.models_dir)
    else:
        loader = ModelLoader(bucket=args.bucket)

    print(f"{BOLD}EnergyLens Deployment Diagnostics v3{RST}")
    print(f"Source: {loader.source_label}")
    print(f"Repo:   {args.repo_root.resolve()}")

    if args.issue is None or args.issue == 1:
        diagnose_informer(loader)
    if args.issue is None or args.issue == 2:
        diagnose_xgboost(loader)
    if args.issue is None or args.issue == 3:
        diagnose_catastrophic_filter(args.repo_root)
    if args.issue is None:
        diagnose_config(loader, args.repo_root)

    print(f"\n{BOLD}{'═'*60}")
    print(f"  Done. Fix {FAIL} items, then deploy.")
    print(f"{'═'*60}{RST}\n")


if __name__ == "__main__":
    main()