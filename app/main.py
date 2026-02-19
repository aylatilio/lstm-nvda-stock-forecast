"""
FastAPI application layer.

- Expose HTTP endpoints (/health, /predict)
- Load trained artifacts once (cached)
- Orchestrate data ingestion + inference service
- Return structured prediction responses

Notes:
- No training logic here.
- No feature engineering logic here.
- Pure orchestration layer.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from src.data import build_merged_frame, DataDownloadError
from app.service import load_artifacts, predict_next_close, PredictionResult
import tensorflow as tf

# =============================================================================
# Response models
# =============================================================================

class PredictResponse(BaseModel):
    """
    API response schema for next-day prediction.
    """
    symbol: str
    predicted_logret: float
    close_t_usd: float
    predicted_close_t1_usd: float


# =============================================================================
# Artifact loader (cached singleton)
# =============================================================================

@lru_cache(maxsize=1)
def _get_artifacts() -> dict[str, Any]:
    """
    Load trained artifacts once per process.

    If legacy scaler support is ever needed:
        load_artifacts(allow_legacy_scaler=True)
    """
    return load_artifacts(
        models_dir="models",
        allow_legacy_scaler=False,
    )

# ==========================================================
# Lifespan (startup/shutdown hooks) + FastAPI instance
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Logs TensorFlow runtime and available devices at startup.
    """
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    print(f"[startup] TensorFlow {tf.__version__}")
    print(f"[startup] CPUs detected: {len(cpus)}")
    print(f"[startup] GPUs detected: {len(gpus)}")

    for i, gpu in enumerate(gpus):
        print(f"[startup] GPU[{i}]: {gpu.name}")

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
    """
    Health check endpoint.

    - Docker container validation
    - Orchestrator health checks
    - Load balancer checks
    """
    return {"status": "ok"}

@app.get("/runtime")
def runtime() -> dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "gpus_detected": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
    }

@app.get("/predict", response_model=PredictResponse)
def predict(
    symbol: str = Query(default="NVDA"),
    start: str | None = Query(default=None, description="YYYY-MM-DD"),
) -> PredictResponse:
    """
    Predict next-day close price.

    Flow:
    1) Load artifacts
    2) Download latest market data
    3) Build feature window
    4) Predict log-return
    5) Reconstruct USD close
    """
    artifacts = _get_artifacts()

    meta = artifacts.get("meta", {})
    lookback = int(meta.get("lookback", 60))
    exogenous = meta.get("exogenous", ["SOXX", "MU", "QQQ"])

    # -------------------------------------------------------------------------
    # Safe start date auto-selection
    # -------------------------------------------------------------------------
    if start is None:
        from datetime import datetime, timedelta, timezone

        days = int(5 * lookback + 90)
        start = (
            datetime.now(timezone.utc).date()
            - timedelta(days=days)
        ).isoformat()

    # -------------------------------------------------------------------------
    # Data ingestion
    # -------------------------------------------------------------------------
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
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    result: PredictionResult = predict_next_close(
        artifacts=artifacts,
        df_merged=df_merged,
        symbol=symbol,
    )

    return PredictResponse(
        symbol=result.symbol,
        predicted_logret=result.predicted_logret,
        close_t_usd=result.close_t_usd,
        predicted_close_t1_usd=result.predicted_close_t1_usd,
    )
