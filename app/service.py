"""
Model service layer.

Responsibilities:
- Load trained artifacts (Keras model, scalers, metadata)
- Build the latest input window using the SAME feature schema used in training
- Run inference (predict N-day forward log-return)
- Reconstruct the N-day ahead close price in USD:
    close_hat(t+N) = close(t) * exp(logret_Nd_hat)
"""


from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
import keras

from src.features import add_technical_features


@dataclass
class PredictionResult:
    """
    Output returned by the service for N-day-ahead inference.
    """
    symbol: str
    forecast_horizon_days: int
    predicted_logret: float
    close_t_usd: float
    predicted_close_t_plus_h_usd: float



def _forecast_horizon_days(meta: dict) -> int:
    try:
        h = int(meta.get("forecast_horizon_days", 0) or 0)
        if h > 0:
            return h
    except Exception:
        pass

    target_col = str(meta.get("target_col", "") or "")
    m = re.search(r"_([0-9]+)d$", target_col)
    return int(m.group(1)) if m else 1


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

    if scaler_x_path.exists() and scaler_y_path.exists():
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
    elif allow_legacy_scaler and scaler_legacy_path.exists():
        scaler_x = joblib.load(scaler_legacy_path)
        scaler_y = joblib.load(scaler_legacy_path)
    else:
        msg = (
            "Scalers not found for Option A. Expected scaler_x.pkl + scaler_y.pkl. "
            "If you intentionally want legacy behavior, pass allow_legacy_scaler=True "
            "and ensure scaler.pkl exists."
        )
        raise FileNotFoundError(msg)

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {"model": model, "scaler_x": scaler_x, "scaler_y": scaler_y, "meta": meta}


def _build_latest_window(
    df_merged,
    symbol: str,
    feature_cols: list[str],
    lookback: int,
    scaler_x,
    exogenous: list[str],
    target_col: str = "",
) -> tuple[np.ndarray, float]:
    """
    Build the most recent LSTM input window using training feature schema.

    Returns:
    - X_window_scaled: shape (1, lookback, n_features)
    - close_t_usd: last close in USD (used for reconstruction)
    """
    sym = symbol.lower()

    df_feat = add_technical_features(df_merged, symbol=symbol, exogenous=exogenous)

    if target_col and target_col not in df_feat.columns:
        raise ValueError(
            f"Target column '{target_col}' not present after feature engineering. "
            f"Check add_technical_features() and training meta.json."
        )

    if len(df_feat) < lookback:
        raise ValueError(f"Not enough data after feature engineering: need {lookback}, got {len(df_feat)}")

    df_last = df_feat.iloc[-lookback:].copy()

    missing = [c for c in feature_cols if c not in df_last.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for inference: {missing}")

    X_raw = df_last[feature_cols].astype(np.float32).values
    X_scaled = scaler_x.transform(X_raw).astype(np.float32)
    X_window_scaled = X_scaled.reshape(1, lookback, -1)

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
    Predict N-day ahead close in USD.

    The model outputs the N-day forward log-return:
        logret_Nd(t) = log(close(t+N)) - log(close(t))

    Reconstruction:
        close_hat(t+N) = close(t) * exp(logret_Nd_hat)
    """
    model: keras.Model = artifacts["model"]
    scaler_x = artifacts["scaler_x"]
    scaler_y = artifacts["scaler_y"]
    meta: dict[str, Any] = artifacts["meta"]

    target_col = str(meta.get("target_col", "") or "")


    forecast_h = _forecast_horizon_days(meta)

    try:
        feature_cols = meta["feature_cols"]
        lookback = int(meta["lookback"])
        exogenous = meta.get("exogenous", ["SOXX", "MU", "QQQ"])
    except Exception as e:
        raise ValueError(f"Invalid meta.json for inference: {type(e).__name__}: {e}")

    X_window_scaled, close_t_usd = _build_latest_window(
        df_merged=df_merged,
        symbol=symbol,
        feature_cols=feature_cols,
        lookback=lookback,
        scaler_x=scaler_x,
        exogenous=exogenous,
        target_col=target_col,
    )
    
    y_pred_scaled = model.predict(X_window_scaled, verbose=0)
    y_pred_logret = scaler_y.inverse_transform(y_pred_scaled)

    predicted_logret = float(y_pred_logret.reshape(-1)[0])
    predicted_close = float(close_t_usd * np.exp(predicted_logret))

    return PredictionResult(
        symbol=symbol,
        forecast_horizon_days=forecast_h,
        predicted_logret=predicted_logret,
        close_t_usd=float(close_t_usd),
        predicted_close_t_plus_h_usd=predicted_close,
    )
