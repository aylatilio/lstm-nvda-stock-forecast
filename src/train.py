"""
Trains an LSTM model to predict 5-day forward NVDA log-returns (logret_5d),
then reconstructs a 5-day forward USD close forecast.

Pipeline:
- Data ingestion (NVDA + exogenous proxies)
- Feature engineering and scaling
- Supervised window creation for LSTM
- Training with callbacks (EarlyStopping + ReduceLROnPlateau)
- Evaluation in scaled space and log-return space
- Price reconstruction in USD (close(t) * exp(logret_5d_hat))
- Baseline comparisons:
  - Zero log-return baseline (predict 0 => "no change over 5 days")
  - Persistence price baseline (close(t+5) = close(t))
- Artifact saving (model, scalers, metadata)
"""
# pyright: reportArgumentType=false
from __future__ import annotations

# ================= Standard Library =================
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

# ================= Environment config (before TF) =================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ================= Third-party =================
import joblib
import numpy as np
import tensorflow as tf
import keras
from keras import layers, optimizers, metrics

# ================= Local imports =================
from src.data import build_merged_frame
from src.backtest import generate_walk_forward_splits, FoldSplit
from src.features import add_technical_features, prepare_datasets_from_feature_frame


# =============================================================================
# Runtime diagnostics
# =============================================================================

def log_runtime_environment() -> None:
    """
    Print TensorFlow runtime diagnostics.

    Useful when running inside WSL/Docker to confirm
    CPU/GPU visibility.
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
    Enable GPU memory growth if GPUs are available.
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
    Build a compact LSTM model for 5-day forward return prediction.

    The model predicts scaled logret_5d.
    """

    inputs = keras.Input(shape=input_shape)

    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            metrics.MeanAbsoluteError(name="mae"),
            metrics.RootMeanSquaredError(name="rmse"),
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

@dataclass
class FoldResult:
    """
    Holds metrics, split metadata, and the trained artifacts for one fold.
    """
    fold_id: int
    split: dict[str, Any]

    metrics_logret: dict[str, float]
    baseline_zero_logret: dict[str, float]

    metrics_price_usd: dict[str, float]
    baseline_persistence_price_usd: dict[str, float]

    # Artifacts (kept in memory, we will persist only the best fold)
    model: keras.Model
    scaler_x: Any
    scaler_y: Any
    feature_cols: list[str]
    lookback: int
    horizon: int
    target_col: str
    close_col_name: str


def _aggregate_metric(fold_results: list[FoldResult], key: str) -> dict[str, dict[str, float]]:
    """
    Aggregate a metric dict across folds, returning mean and population std.

    Example:
      key="metrics_logret" aggregates fold.metrics_logret["mae"], ["rmse"], etc.
    """
    if not fold_results:
        return {}

    # Collect metric names from the first fold
    metric_names = list(getattr(fold_results[0], key).keys())
    out: dict[str, dict[str, float]] = {}

    for m in metric_names:
        values = [float(getattr(fr, key)[m]) for fr in fold_results]
        out[m] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        }

    return out


def train_and_evaluate_fold(
    df_feat,
    *,
    symbol: str,
    split: FoldSplit,
    lookback: int,
    horizon: int,
    target_col: str,
    max_epochs: int,
    batch_size: int,
) -> FoldResult:
    """
    Train and evaluate one walk-forward fold.

    Notes:
    - df_feat is the fully engineered dataframe (no NaNs).
    - Scalers are fit only on the training slice inside prepare_datasets_from_feature_frame().
    - Model is trained on X_train/y_train and validated on X_val/y_val.
    - Test metrics are computed on X_test/y_test.
    """
    bundle = prepare_datasets_from_feature_frame(
        df_feat,
        symbol=symbol,
        split=split,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        feature_cols=None,  # inferred consistently
    )

    model = build_lstm_model(
        input_shape=(bundle.X_train.shape[1], bundle.X_train.shape[2])
    )

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

    model.fit(
        bundle.X_train,
        bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # -------------------------
    # Test evaluation: target space (log-return)
    # -------------------------
    y_pred_scaled = model.predict(bundle.X_test, verbose=0)
    y_true_scaled = bundle.y_test

    y_pred_logret = bundle.scaler_y.inverse_transform(y_pred_scaled)
    y_true_logret = bundle.scaler_y.inverse_transform(y_true_scaled)

    metrics_logret = compute_metrics(y_true_logret, y_pred_logret, use_mape=False)

    baseline_zero_logret = np.zeros_like(y_true_logret)
    baseline_metrics_logret = compute_metrics(
        y_true_logret, baseline_zero_logret, use_mape=False
    )

    # -------------------------
    # Test evaluation: reconstructed USD price space
    # -------------------------
    close_col_name = "nvda_close"
    close_idx = bundle.feature_cols.index(close_col_name)

    X_test_2d = bundle.X_test.reshape(-1, bundle.X_test.shape[-1])
    X_test_inv_2d = bundle.scaler_x.inverse_transform(X_test_2d)
    X_test_inv = X_test_inv_2d.reshape(bundle.X_test.shape)

    close_t_usd = X_test_inv[:, -1, close_idx].reshape(-1, 1)

    close_pred_usd = close_t_usd * np.exp(y_pred_logret)
    close_true_usd = close_t_usd * np.exp(y_true_logret)

    metrics_price_usd = compute_metrics(close_true_usd, close_pred_usd, use_mape=True)

    close_persistence_usd = close_t_usd
    baseline_metrics_price_usd = compute_metrics(
        close_true_usd, close_persistence_usd, use_mape=True
    )

    return FoldResult(
        fold_id=split.fold_id,
        split=split.as_dict(),
        metrics_logret=metrics_logret,
        baseline_zero_logret=baseline_metrics_logret,
        metrics_price_usd=metrics_price_usd,
        baseline_persistence_price_usd=baseline_metrics_price_usd,
        model=model,
        scaler_x=bundle.scaler_x,
        scaler_y=bundle.scaler_y,
        feature_cols=bundle.feature_cols,
        lookback=lookback,
        horizon=horizon,
        target_col=target_col,
        close_col_name=close_col_name,
    )

# =============================================================================
# Training entry point
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    tf.random.set_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Runtime configuration
    # -------------------------------------------------------------------------
    configure_tensorflow_runtime()
    log_runtime_environment()

    # -------------------------------------------------------------------------
    # Experiment configuration
    # -------------------------------------------------------------------------
    symbol = "NVDA"
    exogenous = ["SOXX", "MU", "QQQ"]
    start_date = "2018-01-01"

    lookback = 60
    horizon = 1  # IMPORTANT: target already represents 5-day forward return
    target_col = "nvda_logret_5d"

    m = re.search(r"_([0-9]+)d$", target_col)
    forecast_horizon_days = int(m.group(1)) if m else 1

    max_epochs = 60
    batch_size = 32

    # -------------------------------------------------------------------------
    # Artifact paths
    # -------------------------------------------------------------------------
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lstm_nvda.keras"
    scaler_x_path = models_dir / "scaler_x.pkl"
    scaler_y_path = models_dir / "scaler_y.pkl"
    meta_path = models_dir / "meta.json"

    # -------------------------------------------------------------------------
    # Data ingestion
    # -------------------------------------------------------------------------
    df = build_merged_frame(
        symbol=symbol,
        exogenous=exogenous,
        start=start_date,
    )

    # -------------------------------------------------------------------------
    # Feature engineering ONCE (shared across all folds)
    # -------------------------------------------------------------------------
    df_feat = add_technical_features(df, symbol=symbol, exogenous=exogenous)

    # -------------------------------------------------------------------------
    # Walk-forward split generation (index-based)
    # -------------------------------------------------------------------------
    splits = generate_walk_forward_splits(
        n_rows=len(df_feat),
        n_folds=3,
        test_size=252,
        val_size=126,
        min_train_size=756,
    )

    print("\nWalk-forward splits:")
    for s in splits:
        print(s)

    # -------------------------------------------------------------------------
    # Train/evaluate each fold
    # -------------------------------------------------------------------------
    fold_results: list[FoldResult] = []

    for split in splits:
        print(f"\n=== Fold {split.fold_id} ===")
        fr = train_and_evaluate_fold(
            df_feat,
            symbol=symbol,
            split=split,
            lookback=lookback,
            horizon=horizon,
            target_col=target_col,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        fold_results.append(fr)

    # -------------------------------------------------------------------------
    # Aggregate results across folds
    # -------------------------------------------------------------------------
    agg = {
        "metrics_logret": _aggregate_metric(fold_results, "metrics_logret"),
        "baseline_zero_logret": _aggregate_metric(fold_results, "baseline_zero_logret"),
        "metrics_price_usd": _aggregate_metric(fold_results, "metrics_price_usd"),
        "baseline_persistence_price_usd": _aggregate_metric(fold_results, "baseline_persistence_price_usd"),
    }

    # Pick "best" fold by lowest test RMSE in log-return space
    best_fold = min(fold_results, key=lambda r: r.metrics_logret["rmse"])
    print(f"\nBest fold by test RMSE (logret): Fold {best_fold.fold_id}")

    # -------------------------------------------------------------------------
    # Persist BEST fold artifacts (this is what the API should serve)
    # -------------------------------------------------------------------------
    best_fold.model.save(model_path)
    joblib.dump(best_fold.scaler_x, scaler_x_path)
    joblib.dump(best_fold.scaler_y, scaler_y_path)

    # -------------------------------------------------------------------------
    # Metadata (walk-forward)
    # -------------------------------------------------------------------------
    meta = {
        "symbol": symbol,
        "exogenous": exogenous,
        "start_date": start_date,
        "lookback": lookback,
        "horizon": horizon,
        "target_col": target_col,
        "forecast_horizon_days": forecast_horizon_days,
        "tensorflow_version": tf.__version__,
        "walk_forward": {
            "n_folds": len(fold_results),
            "test_size": 252,
            "val_size": 126,
            "min_train_size": 756,
            "folds": [
                {
                    "fold_id": fr.fold_id,
                    "split": fr.split,
                    "metrics_logret": fr.metrics_logret,
                    "baseline_zero_logret": fr.baseline_zero_logret,
                    "metrics_price_usd": fr.metrics_price_usd,
                    "baseline_persistence_price_usd": fr.baseline_persistence_price_usd,
                }
                for fr in fold_results
            ],
            "aggregate": agg,
            "best_fold_by_test_rmse_logret": best_fold.fold_id,
        },
    }

    meta.update(
        {
            "feature_cols": best_fold.feature_cols,
            "close_col_name": best_fold.close_col_name,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nWalk-forward training complete.")
    print("Saved BEST fold artifacts:")
    print(f"- Model:    {model_path}")
    print(f"- Scaler X: {scaler_x_path}")
    print(f"- Scaler Y: {scaler_y_path}")
    print(f"- Meta:     {meta_path}")


if __name__ == "__main__":
    main()
