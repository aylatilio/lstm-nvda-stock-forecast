"""
FastAPI application layer.

- Expose HTTP endpoints (/health, /runtime, /predict)
- Load trained artifacts once (cached)
- Orchestrate data ingestion + inference service
- Return structured prediction responses
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any
from contextlib import asynccontextmanager
import contextlib
import os
import re
import time

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.data import build_merged_frame, DataDownloadError
from app.service import load_artifacts, predict_next_close, PredictionResult

import tensorflow as tf


def safe_list_gpus() -> list[str]:
    """
    List GPU devices, suppressing noisy CUDA init logs (cuInit) when present.

    This does NOT affect actual GPU usage. It's just log hygiene for demos.
    """
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            gpus = tf.config.list_physical_devices("GPU")
        return [g.name for g in gpus]
    except Exception:
        # If TF blows up here return empty and let /runtime show that no GPUs were detected
        return []

  
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


# =============================================================================
# Response models
# =============================================================================

class PredictResponse(BaseModel):
    symbol: str
    forecast_horizon_days: int
    predicted_logret: float
    close_t_usd: float
    predicted_close_t_plus_h_usd: float


# =============================================================================
# Artifact loader (cached singleton)
# =============================================================================

@lru_cache(maxsize=1)
def _get_artifacts() -> dict[str, Any]:
    return load_artifacts(
        models_dir="models",
        allow_legacy_scaler=False,
    )


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    cpus = tf.config.list_physical_devices("CPU")
    gpu_names = safe_list_gpus()

    print(f"[startup] TensorFlow {tf.__version__}")
    print(f"[startup] CPUs detected: {len(cpus)}")
    print(f"[startup] GPUs detected: {len(gpu_names)}")
    for i, name in enumerate(gpu_names):
        print(f"[startup] GPU[{i}]: {name}")

    yield

    print("[shutdown] API shutting down.")


app = FastAPI(
    title="NVDA LSTM Stock Forecast API",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runtime")
def runtime() -> dict[str, Any]:
    gpu_names = safe_list_gpus()
    return {
        "tensorflow_version": tf.__version__,
        "gpus_detected": len(gpu_names),
        "gpus": gpu_names,
    }


@app.get("/predict", response_model=PredictResponse)
def predict(
    symbol: str = Query(default="NVDA"),
    start: str | None = Query(default=None, description="YYYY-MM-DD"),
) -> PredictResponse:
    t0 = time.perf_counter()

    artifacts = _get_artifacts()
    meta = artifacts.get("meta", {})
    forecast_h = _forecast_horizon_days(meta)
    lookback = int(meta.get("lookback", 60))
    exogenous = meta.get("exogenous", ["SOXX", "MU", "QQQ"])

    if start is None:
        from datetime import datetime, timedelta, timezone
        days = int(5 * lookback + 90)
        start = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()

    try:
        df_merged = build_merged_frame(
            symbol=symbol,
            exogenous=exogenous,
            start=start,
            strict=True,
            allow_fallback=True,
            strict_exogenous=True,
        )
    except DataDownloadError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    result: PredictionResult = predict_next_close(
        artifacts=artifacts,
        df_merged=df_merged,
        symbol=symbol,
    )

    ms = (time.perf_counter() - t0) * 1000.0
    print(f"[predict] symbol={symbol} latency_ms={ms:.1f}")

    return PredictResponse(
        symbol=result.symbol,
        forecast_horizon_days=forecast_h,
        predicted_logret=result.predicted_logret,
        close_t_usd=result.close_t_usd,
        predicted_close_t_plus_h_usd=result.predicted_close_t_plus_h_usd,
    )





