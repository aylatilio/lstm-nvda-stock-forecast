"""
Model service layer.

Responsibilities:
- Load trained artifacts (Keras model, scalers, metadata)
- Build the latest input window using the SAME feature schema used in training
- Run inference (predict next-day log-return)
- Reconstruct next-day close price in USD:
    close_hat(t+1) = close(t) * exp(logret_hat(t+1))

Notes:
- This file contains no FastAPI code on purpose. The API layer should call these functions.
- GPU is used automatically if TensorFlow detects one; otherwise it runs on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
import keras

from src.features import add_technical_features

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PredictionResult:
    """
    Output returned by the service for one-step-ahead inference.
    """
    symbol: str
    predicted_logret: float
    close_t_usd: float
    predicted_close_t1_usd: float


# =============================================================================
# Artifact loading
# =============================================================================

def enable_gpu_memory_growth() -> None:
    """
    GPU-first behavior:
    - If TensorFlow detects a GPU, enable memory growth.
    - If no GPU is detected, TensorFlow runs on CPU automatically.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        

def load_artifacts(models_dir: str | Path = "models", allow_legacy_scaler: bool = False) -> dict[str, Any]:
    """
    Load model/scalers/metadata from disk.

    Expected files:
    - models/lstm_nvda.keras
    - models/scaler_x.pkl
    - models/scaler_y.pkl
    - models/scaler.pkl  (fallback legacy only if allow_legacy_scaler=True)
    - models/meta.json
    """
    enable_gpu_memory_growth()

    models_dir = Path(models_dir)

    model_path = models_dir / "lstm_nvda.keras"
    scaler_x_path = models_dir / "scaler_x.pkl"
    scaler_y_path = models_dir / "scaler_y.pkl"
    scaler_legacy_path = models_dir / "scaler.pkl"
    meta_path = models_dir / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.models.load_model(model_path)

    # strict
    if scaler_x_path.exists() and scaler_y_path.exists():
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

    # Legacy (explicit opt-in)
    elif allow_legacy_scaler and scaler_legacy_path.exists():
        scaler_x = joblib.load(scaler_legacy_path)
        scaler_y = joblib.load(scaler_legacy_path)

    else:
        msg = (
            "Scalers not found for Option A. Expected scaler_x.pkl + scaler_y.pkl."
            "If you intentionally want legacy behavior, pass allow_legacy_scaler=True"
            "and ensure scaler.pkl exists."
        )
        raise FileNotFoundError(msg)

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "meta": meta,
    }

# =============================================================================
# Inference helpers
# =============================================================================

def _build_latest_window(
    df_merged,  # pd.DataFrame, kept untyped here to avoid importing pandas in API layer
    symbol: str,
    feature_cols: list[str],
    lookback: int,
    scaler_x,
) -> tuple[np.ndarray, float]:
    """
    Build the most recent LSTM input window using training feature schema.

    Returns:
    - X_window_scaled: shape (1, lookback, n_features)
    - close_t_usd: the last close price of the window in USD (used for reconstruction)
    """
    sym = symbol.lower()

    # --- Step 1: rebuild engineered features exactly like training
    df_feat = add_technical_features(df_merged, symbol=symbol)

    # --- Step 2: ensure we have enough rows for the lookback window
    if len(df_feat) < lookback:
        raise ValueError(f"Not enough data after feature engineering: need {lookback}, got {len(df_feat)}")

    # --- Step 3: take the last lookback rows and select the exact columns used in training
    df_last = df_feat.iloc[-lookback:].copy()

    missing = [c for c in feature_cols if c not in df_last.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for inference: {missing}")

    X_raw = df_last[feature_cols].astype(np.float32).values  # shape (lookback, n_features)

    # --- Step 4: scale with the TRAINED scaler_x (never refit at inference)
    X_scaled = scaler_x.transform(X_raw).astype(np.float32)

    # --- Step 5: reshape to LSTM batch shape
    X_window_scaled = X_scaled.reshape(1, lookback, -1)

    # --- Step 6: recover close(t) in USD (last row, close column)
    close_col_name = f"{sym}_close"
    if close_col_name not in df_last.columns:
        raise ValueError(f"Close column not found in engineered frame: {close_col_name}")

    close_t_usd = float(df_last[close_col_name].iloc[-1])

    return X_window_scaled, close_t_usd


def predict_next_close(
    artifacts: dict[str, Any],
    df_merged,
    symbol: str = "NVDA",
) -> PredictionResult:
    """
    Predict next-day close in USD using:
    - predicted log-return
    - last known close price close(t)

    Formula:
        close_hat(t+1) = close(t) * exp(logret_hat(t+1))
    """
    model: keras.Model = artifacts["model"]
    scaler_x = artifacts["scaler_x"]
    scaler_y = artifacts["scaler_y"]
    meta: dict[str, Any] = artifacts["meta"]

    # --- Read required inference config from metadata
    feature_cols: list[str] = meta["feature_cols"]
    lookback: int = int(meta["lookback"])

    # --- Build the latest input window
    X_window_scaled, close_t_usd = _build_latest_window(
        df_merged=df_merged,
        symbol=symbol,
        feature_cols=feature_cols,
        lookback=lookback,
        scaler_x=scaler_x,
    )

    # --- Predict scaled log-return, then inverse-transform to natural log-return space
    y_pred_scaled = model.predict(X_window_scaled, verbose=0)  # shape (1, 1)
    y_pred_logret = scaler_y.inverse_transform(y_pred_scaled)  # shape (1, 1)

    predicted_logret = float(y_pred_logret.reshape(-1)[0])

    # --- Reconstruct next-day close in USD
    predicted_close_t1_usd = float(close_t_usd * np.exp(predicted_logret))

    return PredictionResult(
        symbol=symbol,
        predicted_logret=predicted_logret,
        close_t_usd=float(close_t_usd),
        predicted_close_t1_usd=predicted_close_t1_usd,
    )
