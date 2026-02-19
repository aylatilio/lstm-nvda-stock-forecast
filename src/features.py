"""
Features module is responsible for:
1) Creating robust, explainable features for a stock time series problem
2) Scaling features and target using separate scalers (better invertibility)
3) Building supervised learning windows for LSTM models
4) Returning a DatasetBundle with everything needed for training and evaluation

- Targets can be switched (e.g., close, returns, log-returns) without rewriting the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =========================
# Defaults / configuration
# =========================

DEFAULT_EXOGENOUS = ["SOXX", "MU", "QQQ"]


# =========================
# Helper functions
# =========================

def _safe_pct_change(s: pd.Series) -> pd.Series:
    """
    Percentage change with safety against inf values produced by division by zero.

    Parameters
    ----------
    s : pd.Series
        Input series.

    Returns
    -------
    pd.Series
        Percent change series with inf values replaced by NaN.
    """
    return s.pct_change().replace([np.inf, -np.inf], np.nan)


# =========================
# Feature engineering
# =========================

def add_technical_features(
    df: pd.DataFrame,
    symbol: str = "NVDA",
    exogenous: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add lightweight, production-friendly features for a stock time series model.

    - Returns: capture daily changes and momentum
    - Moving averages: capture trend
    - Rolling volatility: capture risk/uncertainty
    - RSI: capture momentum and mean-reversion behavior
    - Exogenous proxies: capture sector/macro correlation drivers

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe (target + exogenous) produced by data.build_merged_frame().
    symbol : str
        Target ticker symbol (e.g., "NVDA").
    exogenous : list[str] | None
        Exogenous proxy tickers (e.g., ["SOXX","MU","QQQ"]). Used to compute additional
        proxy features if the corresponding columns exist in the dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features added (NaNs dropped).
    """
    df = df.copy()

    # --- Block 0: normalize tickers (data.py uses lowercase prefixes in column names)
    sym = symbol.lower()
    exo = DEFAULT_EXOGENOUS if exogenous is None else exogenous
    exo = [t.lower() for t in exo]

    # --- Block 1: core price series used for most indicators
    close_col = f"{sym}_close"
    if close_col not in df.columns:
        raise ValueError(f"Expected column '{close_col}' not found in dataframe.")

    close = df[close_col].astype(float)

    # --- Block 2: returns (simple + log)
    # ret_1d: (close_t / close_{t-1}) - 1
    # logret_1d: log(close_t) - log(close_{t-1})
    df[f"{sym}_ret_1d"] = _safe_pct_change(close)
    df[f"{sym}_logret_1d"] = np.log(close).diff()
    
    # --- Block 2b: forward 5-day log-return (prediction target)
    # logret_5d(t) = log(close(t+5)) - log(close(t))
    # This represents the 5-day forward return starting at time t.
    df[f"{sym}_logret_5d"] = np.log(close.shift(-5)) - np.log(close)


    # --- Block 3: moving averages (trend proxies)
    df[f"{sym}_sma_7"] = close.rolling(7).mean()
    df[f"{sym}_sma_21"] = close.rolling(21).mean()

    # --- Block 4: rolling volatility (risk proxies)
    df[f"{sym}_vol_7"] = df[f"{sym}_ret_1d"].rolling(7).std()
    df[f"{sym}_vol_21"] = df[f"{sym}_ret_1d"].rolling(21).std()

    # --- Block 5: RSI(14) (momentum / mean-reversion signal)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df[f"{sym}_rsi_14"] = 100 - (100 / (1 + rs))

    # --- Block 6: exogenous proxies (sector and macro drivers)
    # Compute returns and volatility for each proxy close series if present
    for t in exo:
        proxy_close = f"{t}_close"
        if proxy_close in df.columns:
            df[f"{t}_ret_1d"] = _safe_pct_change(df[proxy_close].astype(float))
            df[f"{t}_vol_21"] = df[f"{t}_ret_1d"].rolling(21).std()

    # --- Block 7: drop rows with NaNs created by rolling computations
    return df.dropna()


# =========================
# Windowing / supervised dataset
# =========================

def make_windows_xy(
    X_values: np.ndarray,
    y_values: np.ndarray,
    lookback: int = 60,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw arrays into supervised windows for sequence models.

    Inputs:
    - X_values: shape (n_timesteps, n_features)
    - y_values: shape (n_timesteps, 1)
    - lookback: number of past timesteps used as input
    - horizon: how many steps ahead we predict (1 = next day)

    Outputs:
    - X: shape (samples, lookback, n_features)
    - y: shape (samples, 1)
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    n = len(X_values)

    # --- Block: slide a window over the time dimension
    for i in range(lookback, n - horizon + 1):
        # Input window uses the previous `lookback` days (ends at i-1)
        X_list.append(X_values[i - lookback : i, :])

        # Target is the day at (i + horizon - 1)
        # With horizon=1, this is y[i], which corresponds to the "next day"
        # relative to the last input timestamp (i-1).
        y_list.append(y_values[i + horizon - 1, :])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# =========================
# Bundle returned to training code
# =========================

@dataclass
class DatasetBundle:
    """
    Container holding all splits, scalers, columns and metadata
    required for training and evaluation.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    scaler_x: MinMaxScaler
    scaler_y: MinMaxScaler

    feature_cols: list[str]
    target_col: str
    df_feat: pd.DataFrame
    meta: dict


# =========================
# Main preparation function
# =========================

def prepare_datasets(
    df: pd.DataFrame,
    symbol: str = "NVDA",
    lookback: int = 60,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    target_col: str | None = None,
    exogenous: list[str] | None = None,
) -> DatasetBundle:
    """
    Prepare train/val/test datasets for LSTM.

    Key idea:
    - We allow switching the prediction target without rewriting the pipeline.
    - For financial series, predicting returns/log-returns is often more stable than
      predicting absolute prices.

    Important:
    - We intentionally exclude `target_col` from the input feature set if it is an
      engineered column (e.g., nvda_logret_1d) to avoid accidentally feeding the
      target series itself as a feature.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset from data.build_merged_frame().
    symbol : str
        Target ticker (e.g., "NVDA").
    lookback : int
        Number of historical timesteps in each input window.
    horizon : int
        Forecast horizon (1 = next day).
    train_ratio : float
        Fraction of samples used for training.
    val_ratio : float
        Fraction of samples used for validation.
    target_col : str | None
        Column to predict. If None, defaults to "{symbol}_close".
    exogenous : list[str] | None
        Exogenous tickers used for computing proxy features.

    Returns
    -------
    DatasetBundle
        Bundle with splits, scalers, column names and metadata.
    """
    # --- Block 1: feature engineering
    df_feat = add_technical_features(df, symbol=symbol, exogenous=exogenous)

    sym = symbol.lower()

    # --- Block 2: choose target column
    if target_col is None:
        target_col = f"{sym}_close"

    if target_col not in df_feat.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # --- Block 3: define base feature columns (always useful)
    # We include close in features even if it is the target, because close(t) is a
    # legitimate input for forecasting close(t+1).
    base_cols = [
        f"{sym}_open",
        f"{sym}_high",
        f"{sym}_low",
        f"{sym}_close",
        f"{sym}_volume",
        "soxx_close",
        "soxx_volume",
        "mu_close",
        "mu_volume",
        "qqq_close",
        "qqq_volume",
    ]

    # --- Block 4: include engineered columns, excluding the target itself
    engineered = [c for c in df_feat.columns if c not in base_cols and c != target_col]
    feature_cols = [c for c in base_cols if c in df_feat.columns] + engineered

    # --- Block 5: build raw arrays for X and y
    X_raw = df_feat[feature_cols].astype(np.float32).values
    y_raw = df_feat[[target_col]].astype(np.float32).values

    # --- Block 6: fit scalers (separate scalers for X and y)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    # --- Block 7: build supervised windows
    X, y = make_windows_xy(X_scaled, y_scaled, lookback=lookback, horizon=horizon)

    # --- Block 8: chronological split (no shuffling for time series)
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # --- Block 9: metadata for reproducibility and documentation
    meta = {
        "symbol": symbol,
        "target_col": target_col,
        "lookback": lookback,
        "horizon": horizon,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_samples": n,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "start_date": str(df_feat.index.min().date()),
        "end_date": str(df_feat.index.max().date()),
    }

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        target_col=target_col,
        df_feat=df_feat,
        meta=meta,
    )

def prepare_datasets_from_feature_frame(
    df_feat: pd.DataFrame,
    *,
    symbol: str,
    split,
    lookback: int,
    horizon: int,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> DatasetBundle:
    """
    Prepare train/val/test datasets for a specific walk-forward split.

    This function is designed for walk-forward backtesting:
    - Feature engineering is assumed to be DONE already (df_feat is clean).
    - Scalers are fit ONLY on the training slice to prevent leakage.
    - Windows are built within each slice independently.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Feature-engineered dataframe with no NaNs (output of add_technical_features()).
    symbol : str
        Target ticker symbol (e.g., "NVDA").
    split : FoldSplit
        Train/val/test index ranges (Python slicing [start, end)).
    lookback : int
        Number of timesteps in each input window.
    horizon : int
        Forecast horizon for windowing (keep 1 if target is forward-return like logret_5d).
    target_col : str
        Column to predict (must exist in df_feat).
    feature_cols : list[str] | None
        Optional explicit feature column list. If None, it is inferred similarly
        to prepare_datasets().

    Returns
    -------
    DatasetBundle
        Bundle with splits, scalers, column names and metadata.
    """
    sym = symbol.lower()

    if target_col not in df_feat.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature frame.")

    # --- Block 1: Feature columns: reuse the same default logic, but allow explicit override
    if feature_cols is None:
        base_cols = [
            f"{sym}_open",
            f"{sym}_high",
            f"{sym}_low",
            f"{sym}_close",
            f"{sym}_volume",
            "soxx_close",
            "soxx_volume",
            "mu_close",
            "mu_volume",
            "qqq_close",
            "qqq_volume",
        ]
        engineered = [c for c in df_feat.columns if c not in base_cols and c != target_col]
        feature_cols = [c for c in base_cols if c in df_feat.columns] + engineered

    # --- Block 2: Slice by split indices (IMPORTANT: no leakage)
    df_train = df_feat.iloc[split.train_start : split.train_end].copy()
    df_val = df_feat.iloc[split.val_start : split.val_end].copy()
    df_test = df_feat.iloc[split.test_start : split.test_end].copy()

    # --- Block 3: Extract raw arrays
    X_train_raw = df_train[feature_cols].astype(np.float32).values
    y_train_raw = df_train[[target_col]].astype(np.float32).values

    X_val_raw = df_val[feature_cols].astype(np.float32).values
    y_val_raw = df_val[[target_col]].astype(np.float32).values

    X_test_raw = df_test[feature_cols].astype(np.float32).values
    y_test_raw = df_test[[target_col]].astype(np.float32).values

    # --- Block 4: Fit scalers ONLY on train
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    # --- Block 5: Transform val/test using train-fitted scalers
    X_val_scaled = scaler_x.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)

    X_test_scaled = scaler_x.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # --- Block 6: Windowing inside each slice
    X_train, y_train = make_windows_xy(X_train_scaled, y_train_scaled, lookback=lookback, horizon=horizon)
    X_val, y_val = make_windows_xy(X_val_scaled, y_val_scaled, lookback=lookback, horizon=horizon)
    X_test, y_test = make_windows_xy(X_test_scaled, y_test_scaled, lookback=lookback, horizon=horizon)

    meta = {
        "symbol": symbol,
        "target_col": target_col,
        "lookback": lookback,
        "horizon": horizon,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "split": split.as_dict() if hasattr(split, "as_dict") else None,
        "start_date": str(df_feat.index.min().date()),
        "end_date": str(df_feat.index.max().date()),
    }

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        target_col=target_col,
        df_feat=df_feat,
        meta=meta,
    )
