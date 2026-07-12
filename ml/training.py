"""
EnergyLens Phase 2 — Training pipeline.

Handles:
  • Sequence preparation (RobustScaler, sliding window)
  • Neural network training with early stopping
  • XGBoost & sklearn ensemble training
  • Time-series cross-validation with ensemble weight calculation
  • Model persistence (save/load)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .models import (
    build_model_registry,
    XGBOOST_AVAILABLE,
    AdvancedTransformer, CNNLSTMAttention, EnhancedTCN,
    EnhancedInformer, EnhancedNBeats, LSTMGRUEnsemble,
    XGBoostTimeSeries, SklearnEnsemble,
)
from .features import get_energy_price_range

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

def prepare_sequences(
    df, feature_cols: list[str], time_step: int = 48
) -> tuple[np.ndarray | None, np.ndarray | None, RobustScaler | None, list[str] | None]:
    """
    Prepare sliding-window sequences for training.

    Args:
        df: DataFrame with feature columns
        feature_cols: ordered list of column names to use
        time_step: lookback window (default 48 hours for energy)

    Returns:
        (X, y, scaler, used_features) or (None,)*4 on failure.
        Close (the price target) is always placed at index 0.
    """
    try:
        available = [c for c in feature_cols if c in df.columns]
        if "Close" not in available:
            logger.error("Close column required for prediction target")
            return None, None, None, None

        # Ensure Close is first
        available.remove("Close")
        available.insert(0, "Close")

        clean = df[available].dropna()
        if len(clean) < time_step + 10:
            logger.error(f"Insufficient data: {len(clean)} rows (need {time_step + 10}+)")
            return None, None, None, None

        scaler = RobustScaler()
        scaled = scaler.fit_transform(clean.values)

        X, y = [], []
        for i in range(time_step, len(scaled)):
            X.append(scaled[i - time_step : i])
            y.append(scaled[i, 0])  # Close is at index 0

        X, y = np.array(X), np.array(y)
        logger.info(f"Prepared {len(X)} sequences — shape {X.shape}")
        return X, y, scaler, available

    except Exception as e:
        logger.error(f"Sequence preparation failed: {e}")
        return None, None, None, None


# ═════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_nn(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100, patience: int = 15, min_epochs: int = 10,
) -> nn.Module:
    """Train a PyTorch model with early stopping and gradient clipping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    if torch.isnan(X_t).any() or torch.isnan(y_t).any():
        logger.error("NaN in training data — aborting")
        return model.cpu()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        try:
            pred = model(X_t)
            loss = criterion(pred.squeeze(), y_t)
            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch}")
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        except Exception as e:
            logger.warning(f"Training error epoch {epoch}: {e}")
            break

        # Validate
        model.eval()
        with torch.no_grad():
            try:
                val_pred = model(X_v)
                val_loss = criterion(val_pred.squeeze(), y_v)
                if torch.isnan(val_loss):
                    break
            except Exception:
                val_loss = loss

        scheduler.step(val_loss)

        # Early stopping
        if epoch >= min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stop at epoch {epoch} (best val loss {best_val_loss:.6f})")
                    break
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            logger.info(f"  Epoch {epoch}: train={loss.item():.6f}  val={val_loss.item():.6f}")

    model = model.cpu()
    if best_state:
        model.load_state_dict(best_state)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def cross_validate(models_dict: dict, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5) -> tuple[dict, dict]:
    """
    Time-series cross-validation for all trained models.
    Returns (cv_results, ensemble_weights).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: dict[str, dict] = {}

    for name, model in models_dict.items():
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            try:
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # Clone and retrain
                if name in ("xgboost", "sklearn_ensemble"):
                    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
                    X_te_flat = X_te.reshape(X_te.shape[0], -1)
                    model.fit(X_tr_flat, y_tr)
                    y_pred = model.predict(X_te_flat)
                else:
                    # For neural nets, use the already-trained model for eval only
                    model.eval()
                    with torch.no_grad():
                        inp = torch.tensor(X_te, dtype=torch.float32)
                        if name == "enhanced_nbeats":
                            inp = inp.reshape(inp.shape[0], -1)
                        y_pred = model(inp).cpu().numpy().flatten()

                mse = mean_squared_error(y_te, y_pred)
                fold_scores.append(mse)
            except Exception as e:
                logger.warning(f"CV fold {fold} failed for {name}: {e}")

        if fold_scores:
            results[name] = {
                "mean_mse": float(np.mean(fold_scores)),
                "std_mse": float(np.std(fold_scores)),
                "n_folds": len(fold_scores),
            }
            logger.info(f"  CV {name}: MSE={results[name]['mean_mse']:.6f} ± {results[name]['std_mse']:.6f}")

    # Compute inverse-MSE ensemble weights
    weights = {}
    total_inv = 0.0
    for name, res in results.items():
        inv = 1.0 / (res["mean_mse"] + 1e-8)
        weights[name] = inv
        total_inv += inv
    if total_inv > 0:
        weights = {k: v / total_inv for k, v in weights.items()}

    logger.info(f"Ensemble weights: {weights}")
    return results, weights


# ═════════════════════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def train_all_models(
    df, feature_cols: list[str], zone: str = "DK1",
    time_step: int = 48, run_cv: bool = True,
) -> tuple[dict | None, RobustScaler | None, dict | None]:
    """
    Train all 8 models on the provided DataFrame.

    Returns:
        (trained_models, scaler, config)  or  (None, None, None) on failure.
    """
    logger.info(f"Training pipeline for zone {zone}")

    X, y, scaler, used_features = prepare_sequences(df, feature_cols, time_step)
    if X is None:
        return None, None, None

    # Train/test split (last 10% for validation)
    split = max(10, len(X) - max(5, len(X) // 10))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    n_features = X.shape[2]
    seq_len = X.shape[1]
    trained: dict = {}

    # ── Neural networks ──────────────────────────────────────────────
    registry = build_model_registry(n_features, seq_len)
    nn_models = ["cnn_lstm", "enhanced_tcn", "enhanced_informer",
                 "advanced_transformer", "enhanced_nbeats", "lstm_gru_ensemble"]

    for name in nn_models:
        if name not in registry:
            continue
        try:
            logger.info(f"Training {name}…")
            model = registry[name]()

            if name == "enhanced_nbeats":
                X_tr_flat = X_train.reshape(X_train.shape[0], -1)
                X_te_flat = X_test.reshape(X_test.shape[0], -1)
                model = train_nn(model, X_tr_flat, y_train, X_te_flat, y_test)
            else:
                model = train_nn(model, X_train, y_train, X_test, y_test)

            trained[name] = model
            logger.info(f"✅ {name} trained")
        except Exception as e:
            logger.warning(f"❌ {name} failed: {e}")

    # ── XGBoost ──────────────────────────────────────────────────────
    if "xgboost" in registry:
        try:
            logger.info("Training XGBoost…")
            xgb_model = registry["xgboost"]()
            xgb_model.fit(X_train, y_train)
            trained["xgboost"] = xgb_model
            logger.info("✅ XGBoost trained")
        except Exception as e:
            logger.warning(f"❌ XGBoost failed: {e}")

    # ── Sklearn ensemble ─────────────────────────────────────────────
    try:
        logger.info("Training sklearn ensemble…")
        sk = registry["sklearn_ensemble"]()
        sk.fit(X_train, y_train)
        trained["sklearn_ensemble"] = sk
        logger.info("✅ Sklearn ensemble trained")
    except Exception as e:
        logger.warning(f"❌ Sklearn ensemble failed: {e}")

    if not trained:
        logger.error("No models trained successfully")
        return None, None, None

    # ── Cross-validation ─────────────────────────────────────────────
    cv_results, ensemble_weights = {}, {}
    if run_cv and len(trained) > 1:
        logger.info("Running cross-validation…")
        cv_results, ensemble_weights = cross_validate(trained, X_train, y_train)

    # ── Dynamic price range from training data ───────────────────────
    if "Close" in df.columns:
        close = df["Close"].dropna()
        actual_min, actual_max = float(close.min()), float(close.max())
        buf = (actual_max - actual_min) * 0.25
        dynamic_range = (actual_min - buf, actual_max + buf)
    else:
        dynamic_range = get_energy_price_range(zone)

    config = {
        "zone": zone,
        "time_step": time_step,
        "n_features": n_features,
        "seq_len": seq_len,
        "used_features": used_features,
        "cv_results": cv_results,
        "ensemble_weights": ensemble_weights,
        "price_range": dynamic_range,
    }

    logger.info(f"Training complete: {len(trained)} models for {zone}")
    return trained, scaler, config


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

def save_models(trained: dict, scaler, config: dict, zone: str = "DK1",
                base_dir: str = "models") -> None:
    """Save all trained models, scaler, and config to disk."""
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)

    for name, model in trained.items():
        try:
            if isinstance(model, torch.nn.Module):
                fpath = path / f"{zone}_{name}.pt"
                torch.save(model.state_dict(), fpath, _use_new_zipfile_serialization=False)
            else:
                fpath = path / f"{zone}_{name}.pkl"
                with open(fpath, "wb") as f:
                    pickle.dump(model, f)
            logger.info(f"Saved {name} → {fpath}")
        except Exception as e:
            logger.warning(f"Failed to save {name}: {e}")

    # Scaler & config
    with open(path / f"{zone}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(path / f"{zone}_config.pkl", "wb") as f:
        pickle.dump(config, f)

    logger.info(f"All artifacts saved to {path}/")


def load_models(zone: str = "DK1", base_dir: str = "models") -> tuple[dict, dict]:
    """Load trained models and config from disk."""
    path = Path(base_dir)
    models = {}
    config = {}

    # Load config first (need n_features and seq_len)
    config_file = path / f"{zone}_config.pkl"
    if not config_file.exists():
        logger.warning(f"No config found for zone {zone}")
        return {}, {}

    with open(config_file, "rb") as f:
        config = pickle.load(f)

    n_features = config.get("n_features", 5)
    seq_len = config.get("seq_len", 48)

    # Load scaler into config
    scaler_file = path / f"{zone}_scaler.pkl"
    if scaler_file.exists():
        with open(scaler_file, "rb") as f:
            config["scaler"] = pickle.load(f)

    # Model type → constructor
    nn_constructors = {
        "cnn_lstm":             lambda: CNNLSTMAttention(n_features, seq_len),
        "enhanced_tcn":         lambda: EnhancedTCN(n_features),
        "enhanced_informer":    lambda: EnhancedInformer(n_features),
        "advanced_transformer": lambda: AdvancedTransformer(n_features, seq_len=seq_len),
        "enhanced_nbeats":      lambda: EnhancedNBeats(input_size=n_features * seq_len),
        "lstm_gru_ensemble":    lambda: LSTMGRUEnsemble(n_features),
    }

    # Custom unpickler: models trained on Kaggle have classes in __main__,
    # but locally they live in ml.models. This redirects the lookup.
    class _KaggleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" or module == "builtins":
                from . import models as ml_models
                cls = getattr(ml_models, name, None)
                if cls is not None:
                    return cls
            return super().find_class(module, name)

    model_names = list(nn_constructors.keys()) + ["xgboost", "sklearn_ensemble"]

    for name in model_names:
        pt_file = path / f"{zone}_{name}.pt"
        pkl_file = path / f"{zone}_{name}.pkl"

        try:
            if name in nn_constructors and pt_file.exists():
                model = nn_constructors[name]()
                state = torch.load(pt_file, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                models[name] = model
                logger.info(f"Loaded {name}")
            elif pkl_file.exists():
                with open(pkl_file, "rb") as f:
                    models[name] = _KaggleUnpickler(f).load()
                logger.info(f"Loaded {name}")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")

    logger.info(f"Loaded {len(models)} models for zone {zone}")
    return models, config