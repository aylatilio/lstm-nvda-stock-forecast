"""
Smoke test for the inference service runs a full end-to-end inference flow using the 
latest trained artifacts and fresh data.

1) Loads trained artifacts from /models (Keras model + scaler_x + scaler_y + meta.json)
2) Downloads/merges fresh data (NVDA + proxies) using the same ingestion as training
3) Builds the latest lookback window using the SAME feature schema used in training
4) Predicts next-day log-return and reconstructs next-day close in USD:
    close_hat(t+1) = close(t) * exp(logret_hat(t+1))

Run:
    python -m tests.smoke_test_service
Optional:
    python -m tests.smoke_test_service --symbol NVDA --models-dir models
    python -m tests.smoke_test_service --allow-legacy-scaler
"""

from __future__ import annotations
import os

# Silence TF startup spam for smoke test output readability
# 0=all, 1=hide INFO, 2=hide INFO+WARNING, 3=hide INFO+WARNING+ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Silence oneDNN notice
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Avoid TF grabbing all VRAM at once (better for laptops)
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


import argparse
import math

from datetime import datetime, timedelta, timezone
from src.data import build_merged_frame
from app.service import load_artifacts, predict_next_close


def _pick_start_date(lookback: int) -> str:
    """
    Pick a safe start date window for inference.

    - Feature engineering uses rolling windows (e.g., 21d vol, 14d RSI)
    - LSTM needs `lookback` rows AFTER feature engineering dropna()

    We'll request a generous amount of history:
    - 5 * lookback days + 90 extra calendar days.
    """
    # Rough calendar days; not trading-days-precise, but fine for yfinance
    days = int(5 * lookback + 90)

    # We avoid importing pandas here; build_merged_frame accepts YYYY-MM-DD strings 
    # and uses yfinance which is fine with calendar days. We just need a stable date 
    # that is "days ago" from today, and Python stdlib can do that without pandas.

    start_dt = datetime.now(timezone.utc).date() - timedelta(days=days)
    return start_dt.isoformat()


def main() -> None:
    import contextlib
    import sys

    # Import TensorFlow with stderr muted to avoid CUDA plugin spam in smoke test output
    with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
        import tensorflow as tf  # noqa: WPS433

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

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

    args = parser.parse_args()

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
    # 1) Must have a DatetimeIndex (yfinance should return this)
    if not hasattr(df_merged, "index"):
        raise RuntimeError("build_merged_frame did not return a DataFrame-like object.")

    if len(df_merged.index) == 0:
        raise RuntimeError("build_merged_frame returned an empty dataframe (download failure).")

    # 2) Must have the expected columns
    required_cols = [f"{args.symbol.lower()}_close"] + [f"{t.lower()}_close" for t in exogenous]
    missing = [c for c in required_cols if c not in df_merged.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after build_merged_frame: {missing}")

    # 3) Must be "recent enough" (avoid passing with stale cached data)
    last_dt = df_merged.index.max()
    if hasattr(last_dt, "to_pydatetime"):
        last_dt = last_dt.to_pydatetime()

    # Allow up to ~10 days lag to handle weekends/holidays/timezones
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
    pct = (math.exp(result.predicted_logret) - 1.0) * 100.0

    print("\n--- PredictionResult ---")
    print(f"symbol: {result.symbol}")
    print(f"predicted_logret(t+1): {result.predicted_logret:.8f}")
    print(f"implied_return_pct(t+1): {pct:+.3f}%")
    print(f"close(t) USD: {result.close_t_usd:.2f}")
    print(f"predicted_close(t+1) USD: {result.predicted_close_t1_usd:.2f}")

    print("\nOK: service inference pipeline executed end-to-end.\n")


if __name__ == "__main__":
    main()
