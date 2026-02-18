"""
Trains an LSTM model to predict next-day NVDA log-returns (logret_1d),
then reconstructs a next-day USD close forecast.

Pipeline:
- Data ingestion (NVDA + exogenous proxies)
- Feature engineering and scaling
- Supervised window creation for LSTM
- Training with callbacks (EarlyStopping + ReduceLROnPlateau)
- Evaluation in scaled space and log-return space
- Price reconstruction in USD (close(t) * exp(logret_hat))
- Baseline comparisons:
  - Zero log-return baseline (predict 0 => "no change")
  - Persistence price baseline (close(t+1) = close(t))
- Artifact saving (model, scalers, metadata)
"""
from __future__ import annotations

import os

# =============================================================================
# TensorFlow environment config (must be set BEFORE importing TensorFlow)
# =============================================================================

# TF_CPP_MIN_LOG_LEVEL:
# 0 = all logs, 1 = hide INFO, 2 = hide INFO+WARNING, 3 = hide INFO+WARNING+ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Prevent TensorFlow from grabbing all VRAM at once (GPU-friendly default)
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Silence the oneDNN informational message (safe)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import keras

from src.data import build_merged_frame
from src.features import prepare_datasets

# =============================================================================
# Runtime diagnostics
# =============================================================================

def log_runtime_environment() -> None:
    """
    Print a clean summary of what TensorFlow can see at runtime.

    Why this matters:
    - On WSL/Ubuntu it is common to have a GPU available at the OS level
      (nvidia-smi works) but not available to TensorFlow due to missing CUDA libs.
    - This function makes the runtime state explicit, so you don't need to guess
      based on scary TF init logs.
    """
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    print("\nRuntime environment:")
    print(f"- TensorFlow version: {tf.__version__}")
    print(f"- CPUs detected: {len(cpus)}")
    print(f"- GPUs detected: {len(gpus)}")

    for i, gpu in enumerate(gpus):
        print(f"  - GPU[{i}]: {gpu.name}")


def configure_tensorflow_runtime() -> None:
    """
    Configure TensorFlow runtime behavior.

    - If GPUs exist, enable 'memory growth' so TF allocates VRAM on demand.
    - If no GPUs exist, do nothing: TensorFlow will run on CPU automatically.
    - Use GPU when available, CPU otherwise.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


# =============================================================================
# Model definition
# =============================================================================

def build_lstm_model(input_shape: tuple[int, int]) -> keras.Model:
    """
    Build a compact LSTM model for one-step-ahead prediction.

    Target:
    - The model predicts next-day log-return (logret_1d) in scaled space.
    - Later we invert the scaler and reconstruct next-day close in USD.

    Architecture rationale:
    - Two stacked LSTMs allow the model to capture both short-term and mid-term
      temporal dynamics.
    - Dropout reduces overfitting.
    - A small Dense head maps the sequence representation into a single value.
    """
    inputs = keras.Input(shape=input_shape)

    # First LSTM returns sequences so we can stack another LSTM
    x = keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.2)(x)

    # Second LSTM summarizes the sequence into a vector
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.2)(x)

    # Dense head
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs)

    # MSE is the standard loss for regression
    # MAE/RMSE help interpret error magnitude during training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_mape: bool = True,
) -> dict:
    """
    Compute MAE and RMSE always; MAPE optionally.

    Why MAPE is optional:
    - MAPE divides by y_true.
    - If y_true is near zero (common for returns/log-returns), MAPE becomes unstable
      and misleading.
    - In USD price space, MAPE is usually meaningful and stable.

    Parameters
    ----------
    y_true : np.ndarray
        True values (shape (n, 1) or (n,)).
    y_pred : np.ndarray
        Predicted values (shape (n, 1) or (n,)).
    use_mape : bool
        Whether to include MAPE.

    Returns
    -------
    dict
        Dictionary with mae, rmse, and optionally mape.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    metrics = {"mae": mae, "rmse": rmse}

    if use_mape:
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0)
        metrics["mape"] = mape

    return metrics


# =============================================================================
# Training entry point
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Block 1: reproducibility
    # -------------------------------------------------------------------------
    # Training is stochastic due to random initialization and minibatch order
    # Seeds help you compare changes (features, model size, etc.)
    tf.random.set_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Block 2: runtime configuration (GPU-first)
    # -------------------------------------------------------------------------
    # Use GPU if visible to TensorFlow, otherwise run on CPU automatically
    configure_tensorflow_runtime()
    log_runtime_environment()

    # -------------------------------------------------------------------------
    # Block 3: experiment configuration (single place to change knobs)
    # -------------------------------------------------------------------------
    symbol = "NVDA"
    exogenous = ["SOXX", "MU", "QQQ"]
    start_date = "2018-01-01"

    lookback = 60
    horizon = 1
    target_col = "nvda_logret_1d"

    max_epochs = 60
    batch_size = 32

    # -------------------------------------------------------------------------
    # Block 4: artifact paths
    # -------------------------------------------------------------------------
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lstm_nvda.keras"
    scaler_x_path = models_dir / "scaler_x.pkl"
    scaler_y_path = models_dir / "scaler_y.pkl"
    meta_path = models_dir / "meta.json"

    # -------------------------------------------------------------------------
    # Block 5: data ingestion
    # -------------------------------------------------------------------------
    # build_merged_frame downloads and merges:
    # - NVDA OHLCV
    # - Exogenous proxies (SOXX, MU, QQQ) close/volume
    df = build_merged_frame(symbol=symbol, exogenous=exogenous, start=start_date)

    # -------------------------------------------------------------------------
    # Block 6: dataset preparation
    # -------------------------------------------------------------------------
    # prepare_datasets:
    # - adds technical features (returns, SMAs, vol, RSI, proxies)
    # - fits scalers (X and y separately)
    # - builds windows of shape (samples, lookback, n_features)
    # - splits chronologically into train/val/test
    bundle = prepare_datasets(
        df,
        symbol=symbol,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
    )

    # -------------------------------------------------------------------------
    # Block 7: build model
    # -------------------------------------------------------------------------
    model = build_lstm_model(
        input_shape=(bundle.X_train.shape[1], bundle.X_train.shape[2])
    )

    # -------------------------------------------------------------------------
    # Block 8: callbacks (training stability)
    # -------------------------------------------------------------------------
    # EarlyStopping:
    # - stops training when validation loss stops improving
    # - restores the best weights observed
    # ReduceLROnPlateau ("plateau" = when val_loss gets stuck on a flat region):
    # - reduces LR when validation loss plateaus (if val_loss doesn't improve for a few epochs, we lower the learning rate)
    # - helps escape flat regions and stabilize convergence
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]

    # -------------------------------------------------------------------------
    # Block 9: training loop
    # -------------------------------------------------------------------------
    history = model.fit(
        bundle.X_train,
        bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # Block 10: evaluation (scaled space)
    # -------------------------------------------------------------------------
    # Scaled-space metrics are mainly for debugging
    # MAPE is not meaningful here since y_true can be near zero
    y_pred_scaled = model.predict(bundle.X_test, verbose=0)
    y_true_scaled = bundle.y_test

    metrics_scaled = compute_metrics(y_true_scaled, y_pred_scaled, use_mape=False)

    # -------------------------------------------------------------------------
    # Block 11: evaluation (target space = log-return)
    # -------------------------------------------------------------------------
    # Inverse-transform to recover log-return values
    # MAPE is NOT meaningful here since y_true can be near zero
    y_pred_logret = bundle.scaler_y.inverse_transform(y_pred_scaled)
    y_true_logret = bundle.scaler_y.inverse_transform(y_true_scaled)

    metrics_logret = compute_metrics(y_true_logret, y_pred_logret, use_mape=False)

    # -------------------------------------------------------------------------
    # Block 12: baseline in target space (zero log-return)
    # -------------------------------------------------------------------------
    # Baseline: predict log-return = 0 for every sample
    # 0 means "tomorrow = today": logret(t+1) = log(close(t+1)) - log(close(t))
    # If logret_hat(t+1) = 0 => log(close_hat(t+1)) = log(close(t)) => close_hat(t+1) = close(t)
    baseline_zero_logret = np.zeros_like(y_true_logret)
    baseline_metrics_logret = compute_metrics(
        y_true_logret, baseline_zero_logret, use_mape=False
    )

    # -------------------------------------------------------------------------
    # Block 13: price reconstruction (USD)
    # -------------------------------------------------------------------------
    # We want a USD close forecast for interpretability:
    # close_hat(t+1) = close(t) * exp(logret_hat(t+1))
    # To reconstruct close(t) in USD:
    # - We inverse-transform X_test back to feature space (USD prices, volumes, etc)
    # - We pick the "nvda_close" feature from the last timestep of each window
    close_col_name = "nvda_close"
    close_idx = bundle.feature_cols.index(close_col_name)

    X_test_2d = bundle.X_test.reshape(-1, bundle.X_test.shape[-1])
    X_test_inv_2d = bundle.scaler_x.inverse_transform(X_test_2d)
    X_test_inv = X_test_inv_2d.reshape(bundle.X_test.shape)

    # close(t) is the last timestep's close value in each input window
    close_t_usd = X_test_inv[:, -1, close_idx].reshape(-1, 1)

    # Reconstruct predicted and true next-day closes (USD)
    close_pred_usd = close_t_usd * np.exp(y_pred_logret)
    close_true_usd = close_t_usd * np.exp(y_true_logret)

    metrics_price_usd = compute_metrics(close_true_usd, close_pred_usd, use_mape=True)

    # -------------------------------------------------------------------------
    # Block 14: baseline in price space (persistence)
    # -------------------------------------------------------------------------
    # Strong baseline for prices: close_hat(t+1) = close(t)
    # If your model cannot beat this, it is not adding value yet
    close_persistence_usd = close_t_usd
    baseline_metrics_price_usd = compute_metrics(
        close_true_usd, close_persistence_usd, use_mape=True
    )

    # -------------------------------------------------------------------------
    # Block 15: metadata
    # -------------------------------------------------------------------------
    # We save enough context to reproduce:
    # - what was trained (symbol, features, target, hyperparams)
    # - what the model achieved (metrics/baselines)
    bundle.meta.update(
        {
            "symbol": symbol,
            "exogenous": exogenous,
            "start_date": start_date,
            "lookback": lookback,
            "horizon": horizon,
            "target_col": target_col,
            "feature_cols": bundle.feature_cols,
            "close_col_name": close_col_name,
            "tensorflow_version": tf.__version__,
            "epochs_trained": len(history.history.get("loss", [])),
            "metrics_scaled": metrics_scaled,
            "metrics_logret": metrics_logret,
            "baseline_zero_logret": baseline_metrics_logret,
            "metrics_price_usd": metrics_price_usd,
            "baseline_persistence_price_usd": baseline_metrics_price_usd,
        }
    )

    # -------------------------------------------------------------------------
    # Block 16: save artifacts
    # -------------------------------------------------------------------------
    model.save(model_path)
    joblib.dump(bundle.scaler_x, scaler_x_path)
    joblib.dump(bundle.scaler_y, scaler_y_path)
    meta_path.write_text(json.dumps(bundle.meta, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Block 17: console summary
    # -------------------------------------------------------------------------
    print("\nTraining complete.")
    print("Saved artifacts:")
    print(f"- Model:    {model_path}")
    print(f"- Scaler X: {scaler_x_path}")
    print(f"- Scaler Y: {scaler_y_path}")
    print(f"- Meta:     {meta_path}")

    print("\nMetrics (scaled space):")
    print(metrics_scaled)

    print("\nMetrics (target space: log-return):")
    print(metrics_logret)

    print("\nBaseline (target space: zero log-return):")
    print(baseline_metrics_logret)

    print("\nMetrics (reconstructed close in USD):")
    print(metrics_price_usd)

    print("\nBaseline (close in USD: persistence close(t+1)=close(t)):")
    print(baseline_metrics_price_usd)


if __name__ == "__main__":
    main()
