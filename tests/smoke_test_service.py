"""
Smoke test for the inference service runs a full end-to-end inference flow using the
latest trained artifacts and fresh data.

1) Loads trained artifacts from /models (Keras model + scaler_x + scaler_y + meta.json)
2) Downloads/merges fresh data (NVDA + proxies) using the same ingestion as training
3) Builds the latest lookback window using the SAME feature schema used in training
4) Predicts N-day-ahead log-return (logret_Nd) and reconstructs the t+N close in USD:
    close_hat(t+N) = close(t) * exp(logret_Nd_hat)

Run:
    python -m tests.smoke_test_service

Optional:
    python -m tests.smoke_test_service --symbol NVDA --models-dir models
    python -m tests.smoke_test_service --allow-legacy-scaler
    python -m tests.smoke_test_service --use-gpu
"""

from __future__ import annotations

import os
import re
import argparse
import math
import contextlib
from datetime import datetime, timedelta, timezone

from src.data import build_merged_frame
from app.service import load_artifacts, predict_next_close


# ---- TF log hygiene (must be set BEFORE importing tensorflow) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")        # hide INFO/WARN
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")       # silence oneDNN notice
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # better on laptops

def _forecast_horizon_days(meta: dict) -> int:
    """
    Prefer meta['forecast_horizon_days'].
    Fallback: parse suffix from meta['target_col'] like 'nvda_logret_5d' -> 5.
    Default: 1.
    """
    try:
        h = int(meta.get("forecast_horizon_days", 0) or 0)
        if h > 0:
            return h
    except Exception:
        pass

    target_col = str(meta.get("target_col", "") or "")
    m = re.search(r"_([0-9]+)d$", target_col)
    return int(m.group(1)) if m else 1

def _pick_start_date(lookback: int) -> str:
    days = int(5 * lookback + 90)
    start_dt = datetime.now(timezone.utc).date() - timedelta(days=days)
    return start_dt.isoformat()


def _safe_list_gpus(tf_mod) -> list[str]:
    """
    Probe GPU devices while suppressing noisy CUDA init logs (cuInit).
    """
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            gpus = tf_mod.config.list_physical_devices("GPU")
        return [g.name for g in gpus]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test inference service.")

    parser.add_argument("--models-dir", default="models", help="Directory containing trained artifacts.")
    parser.add_argument("--symbol", default="NVDA", help="Symbol to predict.")
    parser.add_argument(
        "--allow-legacy-scaler",
        action="store_true",
        help="Allow fallback to models/scaler.pkl (legacy mode). Default is strict mode.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD for data ingestion. If omitted, we auto-pick a safe window.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If set, do NOT disable GPU visibility. Default behavior is CPU-only for clean logs.",
    )

    args = parser.parse_args()

    # ---- Default: CPU-only smoke test to avoid CUDA noise outputs ----
    # Must be set BEFORE importing tensorflow.
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Import TensorFlow (stderr muted to avoid CUDA plugin spam on some builds)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import tensorflow as tf  # noqa: WPS433

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    gpu_names = _safe_list_gpus(tf)

    # -------------------------------------------------------------------------
    # 1) Load artifacts
    # -------------------------------------------------------------------------
    artifacts = load_artifacts(
        models_dir=args.models_dir,
        allow_legacy_scaler=args.allow_legacy_scaler,
    )

    meta = artifacts.get("meta", {})
    lookback = int(meta.get("lookback", 60))
    exogenous = meta.get("exogenous", ["SOXX", "MU", "QQQ"])

    # -------------------------------------------------------------------------
    # 2) Ingest latest data and build merged frame
    # -------------------------------------------------------------------------
    start_date = args.start or _pick_start_date(lookback=lookback)

    print("\n=== Inference smoke test ===")
    print(f"Symbol: {args.symbol}")
    print(f"Models dir: {args.models_dir}")
    print(f"Legacy scaler allowed: {args.allow_legacy_scaler}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs detected: {len(gpu_names)}")
    for i, name in enumerate(gpu_names):
        print(f"- GPU[{i}]: {name}")
    print(f"Lookback (from meta): {lookback}")
    print(f"Exogenous (from meta): {exogenous}")
    print(f"Start date: {start_date}")

    df_merged = build_merged_frame(
        symbol=args.symbol,
        exogenous=exogenous,
        start=start_date,
    )

    # -------------------------------------------------------------------------
    # Debug: print ingestion report (Yahoo vs Stooq per ticker)
    # -------------------------------------------------------------------------
    report = getattr(df_merged, "attrs", {}).get("ingestion_report")
    if report:
        print("\n--- Ingestion report ---")
        for r in report:
            t = r.get("ticker")
            src = r.get("source_used")
            yo = r.get("yahoo_ok")
            so = r.get("stooq_ok")
            print(f"- {t}: source={src} (yahoo_ok={yo}, stooq_ok={so})")

    # -------------------------------------------------------------------------
    # Fail-fast: ensure the dataset looks fresh and usable
    # -------------------------------------------------------------------------
    if not hasattr(df_merged, "index"):
        raise RuntimeError("build_merged_frame did not return a DataFrame-like object.")
    if len(df_merged.index) == 0:
        raise RuntimeError("build_merged_frame returned an empty dataframe (download failure).")

    required_cols = [f"{args.symbol.lower()}_close"] + [f"{t.lower()}_close" for t in exogenous]
    missing = [c for c in required_cols if c not in df_merged.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after build_merged_frame: {missing}")

    last_dt = df_merged.index.max()
    if hasattr(last_dt, "to_pydatetime"):
        last_dt = last_dt.to_pydatetime()

    max_lag_days = 10
    now_utc = datetime.now(timezone.utc)
    if (now_utc - last_dt.replace(tzinfo=timezone.utc)).days > max_lag_days:
        raise RuntimeError(
            f"Data looks stale. Last timestamp in df_merged is {last_dt} (>{max_lag_days} days old)."
        )

    # -------------------------------------------------------------------------
    # 3) Predict using service (build window -> predict logret -> reconstruct USD)
    # -------------------------------------------------------------------------
    result = predict_next_close(
        artifacts=artifacts,
        df_merged=df_merged,
        symbol=args.symbol,
    )

    # -------------------------------------------------------------------------
    # 4) Print a clean report
    # -------------------------------------------------------------------------
    forecast_h = _forecast_horizon_days(artifacts.get("meta", {}))
    pct_h = (math.exp(result.predicted_logret) - 1.0) * 100.0

    print("\n--- PredictionResult ---")
    print(f"symbol: {result.symbol}")
    print(f"predicted_logret_{forecast_h}d(t->t+{forecast_h}): {result.predicted_logret:.8f}")
    print(f"implied_return_pct_{forecast_h}d: {pct_h:+.3f}%")
    print(f"close(t) USD: {result.close_t_usd:.2f}")
    print(f"predicted_close(t+{forecast_h}) USD: {result.predicted_close_t_plus_h_usd:.2f}")


    print("\nOK: service inference pipeline executed end-to-end.\n")


if __name__ == "__main__":
    main()
