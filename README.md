# NVDA Long Short Term Memory (LSTM) Stock Forecast API (FastAPI + Docker)

- LSTM-based RNN with Dense regression head trained with TensorFlow/Keras

This project trains an LSTM model to forecast the next-day closing price of NVDA
using a time-series pipeline with exogenous proxies (SOXX, MU, QQQ) and serves
predictions through a FastAPI REST API.

## Project Goals
- Build an deep learning pipeline for time series forecasting
- Train and evaluate an LSTM model with clear metrics (MAE/RMSE/MAPE)
- Save artifacts (model + scalers + metadata)
- Serve predictions via FastAPI with Swagger docs
- Containerize the API using Docker for reproducible deployment

---

## Tech Stack
- Python 3.12 (WSL)
- TensorFlow / Keras (LSTM)
- Pandas / NumPy / Scikit-learn
- yfinance + Stooq fallback (data ingestion)
- FastAPI + Uvicorn
- Docker

---

## Repository Structure

- `app/`
  - `main.py` — FastAPI entrypoint (Swagger UI)
  - `schemas.py` — Pydantic request/response models
  - `service.py` — model/scaler loading + inference logic
- `src/`
  - `utils.py` — shared utility helpers
  - `data.py` — data ingestion (Yahoo + Stooq fallback) and merging
  - `features.py` — feature engineering, scaling, windowing
  - `train.py` — training script (saves artifacts)
- `models/`
  - `lstm_nvda.keras` — trained LSTM model
  - `scaler_x.pkl` — feature scaler
  - `scaler_y.pkl` — target scaler
  - `scaler.pkl` — legacy scaler
  - `meta.json` — dataset + training metadata and metrics
- `notebooks/`
- `tests/`
- `Dockerfile`
- `requirements.txt` — Python dependencies
- `README.md`

.
├── app/
│   ├── main.py              # FastAPI entrypoint (Swagger UI)
│   ├── schemas.py           # Pydantic request/response models
│   └── service.py           # Model + scaler loading and inference logic
│
├── src/
│   ├── utils.py             # Shared utility helpers
│   ├── data.py              # Data ingestion (Yahoo + Stooq fallback) and merging
│   ├── features.py          # Feature engineering + scaling + windowing
│   └── train.py             # LSTM training script (saves artifacts)
│
├── models/
│   ├── lstm_nvda.keras      # Trained LSTM model
│   ├── scaler_x.pkl         # Feature scaler
│   ├── scaler_y.pkl         # Target scaler
│   ├── scaler.pkl           # Legacy scaler
│   └── meta.json            # Training metadata + evaluation metrics
│
├── notebooks/               # Eploratory notebooks
├── tests/                   # Unit / integration tests
│
├── Dockerfile               # Container definition (API deployment)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

---

## Quick Start

### 1. Install WSL (Windows PowerShell as Admin)
```powershell
wsl --install
wsl --set-default-version 2
```

### 2. Update Ubuntu and install Python tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip build-essential -y
python3 --version
```

### 3. Create project folder
```bash
mkdir -p ~/projects/lstm-nvda-api && cd ~/projects/lstm-nvda-api
```

### 4. Create and activate virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Install CUDA runtime 
```bash
pip uninstall -y tensorflow
pip install "tensorflow[and-cuda]==2.18.0"
python -c "import tensorflow as tf; print(tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 7. Install yfinance lib
```bash
pip install -U yfinance
```

## Training the Model

### 1. Run training
```bash
python -m src.train
```

### 2. Expected outputs (saved artifacts)

After training completes, you should have:

- models/lstm_nvda.keras
- models/scaler_x.pkl
- models/scaler_y.pkl
- models/scaler.pkl
- models/meta.json

## Running API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

- 🔎 API Documentation: [Swagger UI](http://localhost:8000/docs)
- 📦 Local API Base URL: `http://localhost:8000`


## Docker

```bash
docker build -t lstm-nvda-api .
docker run -p 8000:8000 lstm-nvda-api
```
## Notes

- Yahoo Finance sometimes fails inside certain networks/environments. This project includes a Stooq fallback to ensure reproducibility.
- Training and evaluation follow time-series best practices (chronological splits).

## Steps

- WSL + Docker setup
- Data ingestion with fallback
- Feature engineering + windowing
- LSTM training + artifacts
- Improve metrics reporting (USD space + baseline comparison)
- FastAPI inference service + Swagger
- Dockerfile + container test


## Common Issues

### 1. Yahoo Finance failing (YFTzMissingError / timezone error)

```bash
YFTzMissingError('$%ticker%: possibly delisted; no timezone found')
```

Yahoo Finance occasionally fails due to:
- Network restrictions
- Rate limiting
- Invalid JSON response
- Corporate firewall filtering

This project includes a fallback to **Stooq**.
If Yahoo fails, the system automatically attempts to download data from Stooq.

If both fail:
- Check internet access inside WSL
- Try pinging external domains:
```bash
  ping google.com
```
### 2. TensorFlow CUDA/cuDNN/GPU warnings

```bash
Could not find cuda drivers on your machine, GPU will not be used.
Unable to register cuDNN factory
```

TensorFlow was installed with GPU support, but CUDA drivers are not available.
Impact: None. The model runs normally on CPU.

If you want to silence logs add this before running training:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

### 3. Docker permission denied inside WSL

If docker run fails with permission errors:
```bash
sudo usermod -aG docker $USER
```
Then close and reopen the WSL terminal.

### 4. Model metrics seem high (MAPE > 20%)

Time series forecasting is inherently difficult. Always compare model performance against a simple baseline (persistence model).

If the model does not outperform baseline:

- Increase lookback window
- Adjust LSTM architecture
- Try predicting returns instead of raw price