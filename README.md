# 🛡 Webshell Detector

> **M.Tech Dissertation Project** — AI-powered detection of web-based malicious server-side scripts using PyTorch deep learning models.

## Architecture

Four deep learning models are benchmarked:

| Model | Architecture | Notes |
|---|---|---|
| **WebshellDetector** | Bidirectional GRU + Self-Attention | Best overall |
| **TextCNN** | Conv1D with multiple kernel sizes | Fastest inference |
| **TextRNN** | Bidirectional LSTM | Strong baseline |
| **TransformerClassifier** | Transformer Encoder | Highest capacity |

---

## Project Structure

```
webshell_detector/
├── backend/                  ← FastAPI application logic
├── frontend/                 ← Premium glassmorphism dashboard
├── src/                      ← Core ML codebase
│   ├── data/                 ← Data preprocessing pipeline
│   ├── features/             ← Modular feature engineering
│   ├── models/               ← PyTorch model architectures
│   └── evaluation/           ← Metrics & Visualizations
├── data/                     ← Standardized data tier (raw/processed)
├── experiments/              ← Logs, Metrics (CSV), and Graphs
├── models/                   ← Production-ready model weights (.pth)
├── configs/                  ← YAML configurations (ML hyperparameters)
├── notebooks/                ← Interactive Jupyter Notebooks for EDA
├── Makefile                  ← Standardized project commands
├── pyproject.toml            ← Modern python packaging & tool config
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API server

```bash
bash start_api.sh
```

The server starts at **http://localhost:8000**.

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**:      http://localhost:8000/redoc

### 3. Open the Dashboard

Simply open `frontend/index.html` in your browser. No web server needed.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check (models, device) |
| `GET` | `/models` | List loaded models |
| `POST` | `/predict` | Predict from JSON text |
| `POST` | `/predict/file` | Predict from uploaded file |
| `POST` | `/predict/bulk` | Batch scan up to 50 files |

### Example (cURL)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "<?php eval($_POST[\"cmd\"]); ?>", "model_name": "TextCNN"}'
```

Response:
```json
{
  "model": "TextCNN",
  "is_webshell": true,
  "confidence": 0.9834,
  "raw_score": 0.9834,
  "label": "Webshell Detected"
}
```

---

If you want to retrain from scratch using the new MLOps pipeline:

```bash
make train
```

This will run the full preprocessing, data splitting, and deep learning suite according to `configs/train_config.yaml`.

---

## Run Integration Tests

```bash
make test
```

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, PyTorch
- **Models**: GRU+Attention, TextCNN, BiLSTM, Transformer Encoder
- **Embeddings**: Word2Vec (Gensim)
- **Frontend**: Vanilla HTML/CSS/JS (zero dependencies, glassmorphism UI)
