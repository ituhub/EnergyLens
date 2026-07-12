"""
EnergyLens Phase 2 — Neural network model definitions.

Adapted from MarketLens 8-model ensemble:
  1. AdvancedTransformer     – Self-attention over time series
  2. CNNLSTMAttention        – Conv → LSTM → Multi-head attention
  3. EnhancedTCN             – Temporal Convolutional Network (dilated)
  4. EnhancedInformer        – Lightweight Informer-style encoder
  5. EnhancedNBeats          – N-BEATS residual blocks
  6. LSTMGRUEnsemble         – Dual LSTM+GRU fusion
  7. XGBoostTimeSeries       – Gradient-boosted trees (sklearn-API)
  8. SklearnEnsemble         – RF + GBR + SVR + Ridge + Lasso

All PyTorch models accept (batch, seq_len, n_features) and output (batch, 1).
"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

# Optional deps
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size + output_size)
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        backcast, forecast = torch.split(out, [self.input_size, self.output_size], dim=1)
        return backcast, forecast


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1: ADVANCED TRANSFORMER
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, seq_len: int = 60, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.input_projection = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_norm(self.input_projection(x))
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x[:, -1, :])
        return self.output_projection(x)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2: CNN-LSTM WITH ATTENTION
# ═════════════════════════════════════════════════════════════════════════════

class CNNLSTMAttention(nn.Module):
    def __init__(self, n_features: int, seq_len: int = 60, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(64, 100, num_layers=2, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(100, num_heads=4, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(100, 50), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(50, 25), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        try:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            x = attn_out[:, -1, :]
        except Exception:
            x = lstm_out[:, -1, :]
        return self.fc(x)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3: ENHANCED TCN (Temporal Convolutional Network)
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedTCN(nn.Module):
    def __init__(self, n_features: int, num_channels=(64, 128, 256, 128),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = n_features if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                          padding=(kernel_size - 1) * dilation),
                nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            ]
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(num_channels[-1] // 2, 1),
        )

    def forward(self, x):
        x = self.input_norm(x).transpose(1, 2)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 4: ENHANCED INFORMER
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedInformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_norm(self.input_projection(x))
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.output_projection(x[:, -1, :])


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 5: ENHANCED N-BEATS
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedNBeats(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256,
                 output_size: int = 1, num_blocks: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_norm = nn.LayerNorm(input_size)
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, output_size, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        x = self.input_norm(x)
        residuals = x
        forecast = torch.zeros(x.size(0), self.output_size, device=x.device)
        for block in self.blocks:
            try:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                forecast = forecast + block_forecast
            except Exception as e:
                logger.warning(f"NBeats block error: {e}")
                break
        return forecast


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 6: LSTM-GRU ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════

class LSTMGRUEnsemble(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.gru = nn.GRU(n_features, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        combined = torch.cat([lstm_out[:, -1, :], gru_out[:, -1, :]], dim=1)
        fused = F.relu(self.fusion(combined))
        return self.output(fused)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 7: XGBOOST TIME SERIES
# ═════════════════════════════════════════════════════════════════════════════

class XGBoostTimeSeries:
    def __init__(self, n_estimators=300, max_depth=10, learning_rate=0.08,
                 subsample=0.9, colsample_bytree=0.9, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed — pip install xgboost")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, tree_method="hist",
        )
        self.feature_importance_ = None

    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        self.model.fit(X_flat, y)
        self.feature_importance_ = self.model.feature_importances_

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        return self.model.predict(X_flat)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 8: SKLEARN ENSEMBLE (RF + GBR + SVR + Ridge + Lasso)
# ═════════════════════════════════════════════════════════════════════════════

class SklearnEnsemble:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "svr": SVR(kernel="rbf", C=1.0),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.1),
        }
        if GP_AVAILABLE:
            self.models["gaussian_process"] = GaussianProcessRegressor(
                kernel=RBF() + Matern(), random_state=42
            )
        self.fitted = False

    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        failed = []
        for name, model in self.models.items():
            try:
                model.fit(X_flat, y)
            except Exception as e:
                logger.warning(f"SklearnEnsemble: {name} fit failed: {e}")
                failed.append(name)
        for name in failed:
            del self.models[name]
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("SklearnEnsemble not fitted")
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        preds = []
        for name, model in self.models.items():
            try:
                preds.append(model.predict(X_flat))
            except Exception as e:
                logger.warning(f"SklearnEnsemble: {name} predict failed: {e}")
        return np.mean(preds, axis=0) if preds else np.zeros(X_flat.shape[0])


# ═════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — factory for creating all 8 models
# ═════════════════════════════════════════════════════════════════════════════

def build_model_registry(n_features: int, seq_len: int) -> dict:
    """
    Return a dict of (name → model_factory_callable) for all 8 models.
    Each factory is a zero-arg callable that returns a fresh model instance.
    """
    registry = {
        "cnn_lstm":             lambda: CNNLSTMAttention(n_features, seq_len),
        "enhanced_tcn":         lambda: EnhancedTCN(n_features),
        "enhanced_informer":    lambda: EnhancedInformer(n_features),
        "advanced_transformer": lambda: AdvancedTransformer(n_features, seq_len=seq_len),
        "enhanced_nbeats":      lambda: EnhancedNBeats(input_size=n_features * seq_len),
        "lstm_gru_ensemble":    lambda: LSTMGRUEnsemble(n_features),
    }

    if XGBOOST_AVAILABLE:
        registry["xgboost"] = lambda: XGBoostTimeSeries()

    registry["sklearn_ensemble"] = lambda: SklearnEnsemble()

    return registry
