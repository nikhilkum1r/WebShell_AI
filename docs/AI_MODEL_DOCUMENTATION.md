# Webshell Detector v2 — AI Model Technical Documentation

> **Project:** Webshell Detector v2.0 — M.Tech Dissertation  
> **Author:** Nikhil Kumar  
> **Version:** 2.0.0  
> **Date:** December 2025  
> **Framework:** PyTorch 2.x + FastAPI  
> **Status:** Production-Ready

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Model Descriptions](#3-model-descriptions)
4. [Dataset Description](#4-dataset-description)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Process](#6-training-process)
7. [Evaluation Metrics & Performance](#7-evaluation-metrics--performance)
8. [Heuristic Feature Engineering](#8-heuristic-feature-engineering)
9. [Ensemble & Heuristic-Assisted Prediction](#9-ensemble--heuristic-assisted-prediction)
10. [Deployment Architecture](#10-deployment-architecture)
11. [API Reference](#11-api-reference)
12. [Security & Reliability](#12-security--reliability)
13. [Infrastructure Requirements](#13-infrastructure-requirements)
14. [Reproducibility Guide](#14-reproducibility-guide)
15. [Versioning & Model Management](#15-versioning--model-management)
16. [Monitoring & Maintenance](#16-monitoring--maintenance)
17. [Ethical Considerations](#17-ethical-considerations)
18. [Future Improvements](#18-future-improvements)

---

## 1. Project Overview

### 1.1 Purpose

The **Webshell Detector** is an AI-powered security analysis tool designed to detect PHP/ASP webshell malware in source code files. It combines four deep learning architectures with a classical heuristic feature extraction engine to deliver high-accuracy, explainable webshell detection.

### 1.2 Problem Statement

Webshells are malicious PHP/ASP scripts that attackers plant on compromised web servers to maintain persistent access. Traditional signature-based antivirus tools struggle to detect obfuscated or polymorphic webshells. This system uses deep learning models trained on a large corpus of real-world webshell samples to detect both known and novel variants.

### 1.3 Target Users

| User Role | Use Case |
|---|---|
| Security Analysts | Scanning uploaded or existing server files for malware |
| Web Administrators | Post-compromise forensic analysis |
| Researchers / Academics | Benchmarking ML-based malware detection |
| Developers | Integration via REST API into security pipelines |
| M.Tech Examiners | Dissertation evaluation and research review |

### 1.4 Key Objectives

- ✅ Achieve ≥ 98% accuracy on PHP webshell classification
- ✅ Support 4 deep learning architectures + ensemble voting
- ✅ Provide AI explainability (natural language reasoning + code annotation)
- ✅ Enable bulk scanning of up to 50 files per request
- ✅ Expose a production-grade REST API with OpenAPI documentation
- ✅ Support both local and Cloudflare Tunnel (global) access

---

## 2. System Architecture

### 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     User (Browser / Mobile)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/HTTPS
                    ┌──────▼──────┐
                    │  Frontend   │  React-like vanilla JS SPA
                    │  (Port 3000)│  Paste Code / Upload / Bulk Scan
                    └──────┬──────┘
                           │ REST API calls
                    ┌──────▼──────┐
                    │  FastAPI    │  Uvicorn ASGI server
                    │  Backend    │  Port 8000
                    │  (Python)   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │ Model       │  │  Heuristic  │  │  MLOps      │
   │ Manager     │  │  Feature    │  │  Drift      │
   │ (4 Models)  │  │  Extractor  │  │  Monitor    │
   └──────┬──────┘  └─────────────┘  └─────────────┘
          │
   ┌──────▼──────────────────────────────────┐
   │     PyTorch Model Files (models/)        │
   │  WebshellDetector_v2.pth  ~78 MB        │
   │  TextCNN_v2.pth            ~78 MB        │
   │  TextRNN_v2.pth            ~79 MB        │
   │  TransformerClassifier_v2.pth ~81 MB    │
   └─────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| Frontend SPA | `frontend/` | User interface, file upload, results display |
| FastAPI Backend | `backend/main.py` | REST endpoints, request validation, CORS |
| Model Manager | `backend/model_manager.py` | Model loading, tokenisation, inference |
| Heuristic Engine | `src/features/feature_engineering.py` | Security signal extraction |
| Data Pipeline | `src/data/preprocessing.py` | Training data preparation |
| Drift Monitor | `src/evaluation/drift_monitor.py` | MLOps monitoring |
| Config | `backend/config.py` | Centralised environment configuration |

---

## 3. Model Descriptions

All four models share a common **Word2Vec-based embedding layer** (100 dimensions) trained on the project's PHP/ASP dataset. The embedding matrix is loaded from `tokenizer_v2.pkl` and is fine-tunable during training.

### 3.1 Model 1: WebshellDetector (GRU + Attention) — *Recommended*

**Architecture:** Bidirectional GRU with additive soft attention mechanism.

```
Input Tokens → Embedding (100d) → BiGRU (hidden=100, x2 directions)
    → Soft Attention Weights (Linear(200, 1) + Softmax)
    → Context Vector (weighted sum, 200d)
    → Dropout (0.3)
    → FC Layer (200 → 1)
    → Sigmoid → Probability
```

| Parameter | Value |
|---|---|
| Embedding Dim | 100 |
| GRU Hidden Dim | 100 (200 bidirectional) |
| Attention | Additive (Bahdanau-style) |
| Dropout | 0.3 |
| Output Activation | Sigmoid |
| Model Size | ~78 MB |
| Benchmark Accuracy | **98.7%** |
| Parameters | ~550K |

**Design Rationale:** GRU captures sequential patterns in PHP code (common in obfuscated webshells) while the attention mechanism focuses on the most suspicious tokens (e.g., `eval`, `base64_decode`). The attention weights also serve as a form of explainability.

---

### 3.2 Model 2: TextCNN (Convolutional Neural Network)

**Architecture:** Multi-kernel 1D CNN for text classification (Kim 2014 style).

```
Input Tokens → Embedding (100d) → Unsqueeze (channel=1)
    → 3 Parallel Conv2D layers (kernels 3,4,5 × 100, out_channels=128)
    → ReLU Activation
    → Max-over-time Pooling
    → Concatenate (384d)
    → Dropout (0.5)
    → FC Layer (384 → 1)
    → Sigmoid → Probability
```

| Parameter | Value |
|---|---|
| Embedding Dim | 100 |
| Filter Sizes | 3, 4, 5 (n-gram windows) |
| Output Channels | 128 per filter |
| Activation | ReLU (conv) + Sigmoid (output) |
| Dropout | 0.5 |
| Model Size | ~78 MB |
| Benchmark Accuracy | **97.5%** |
| Parameters | ~546K |

**Design Rationale:** CNNs excel at capturing local patterns — ideal for detecting dangerous PHP function names and obfuscation idioms at various n-gram levels.

---

### 3.3 Model 3: TextRNN (Bidirectional LSTM)

**Architecture:** Bidirectional LSTM sequence classifier.

```
Input Tokens → Embedding (100d) → BiLSTM (hidden=128, x2 directions)
    → Last Hidden State (256d)
    → Dropout (0.3)
    → FC Layer (256 → 1)
    → Sigmoid → Probability
```

| Parameter | Value |
|---|---|
| Embedding Dim | 100 |
| LSTM Hidden Dim | 128 (256 bidirectional) |
| Bidirectional | Yes |
| Dropout | 0.3 |
| Output Activation | Sigmoid |
| Model Size | ~79 MB |
| Benchmark Accuracy | **97.3%** |
| Parameters | ~1.0M |

**Design Rationale:** LSTM captures long-range dependencies in code. Bidirectionality helps understand context from both the beginning and end of a PHP file, catching webshells with deferred payload execution.

---

### 3.4 Model 4: Transformer Classifier

**Architecture:** Transformer encoder stack with mean pooling for classification.

```
Input Tokens → Embedding (100d)
    → TransformerEncoder (2 layers, 2 heads, d_model=100, dropout=0.3)
    → Mean Pooling over sequence
    → FC Layer (100 → 1)
    → Sigmoid → Probability
```

| Parameter | Value |
|---|---|
| Embedding Dim | 100 |
| Transformer Layers | 2 |
| Attention Heads | 2 |
| FFN Dropout | 0.3 |
| Pooling | Mean over sequence |
| Model Size | ~81 MB |
| Benchmark Accuracy | **98.2%** |
| Parameters | ~3.7M |

**Design Rationale:** Self-attention allows the model to relate any token to any other, regardless of distance — critical for detecting heavily obfuscated webshells where dangerous functions are scattered throughout the file.

---

### 3.5 Common Parameters Across All Models

| Parameter | Value |
|---|---|
| Vocabulary Size | Dynamic (from tokenizer) |
| Embedding Dimension | 100 |
| Max Sequence Length | 500 tokens |
| Classification Threshold | 0.5 (configurable via `PREDICTION_THRESHOLD` env var) |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Output | Sigmoid probability (0–1), 1 = Webshell |

---

## 4. Dataset Description

### 4.1 Data Sources

The training dataset was assembled from multiple public and research sources:

| Source | Type | Description |
|---|---|---|
| `data/raw/malicious/` | Webshell | Real-world PHP/ASP webshell samples |
| `data/raw/dataC/malicious/` | Webshell | Additional curated malicious samples |
| `data/raw/normal/` | Benign | Legitimate PHP web application code |
| `data/raw/dataC/normal/` | Benign | Additional legitimate PHP source files |

### 4.2 Data Characteristics

| Property | Value |
|---|---|
| Language | PHP / ASP |
| File Types | `.php`, `.asp`, `.aspx`, `.phtml`, `.txt` |
| Classes | Binary: `1 = Webshell`, `0 = Normal` |
| Deduplication | Yes (exact string matching) |
| Label Method | Directory-based (malicious/ vs normal/ folder structure) |

### 4.3 Class Distribution

> ⚠️ **Note:** Exact sample counts depend on the local dataset. Refer to preprocessing output logs in `retrain.log` for per-run statistics.

The dataset is designed to be **approximately balanced** between malicious and normal samples to prevent class-bias in training.

### 4.4 Data Privacy & Licensing

- Webshell samples sourced from public research repositories (typically MIT / CC0 licensed)
- No personally identifiable information (PII) is present in the dataset
- The models do not store or transmit any user-submitted code beyond the immediate inference session

---

## 5. Data Pipeline

### 5.1 Pipeline Stages

```
Raw Files (data/raw/)
    │
    ▼
[1] File Parsing           parse_directory() — recursive UTF-8 reader
    │                      Skips .ipynb_checkpoints, handles encoding errors
    ▼
[2] Text Cleaning          clean_text() — remove non-printable chars,
    │                      normalize whitespace, lowercase
    ▼
[3] Deduplication          Python set() deduplication on cleaned text
    │
    ▼
[4] Label Assignment       1 = malicious, 0 = normal (directory-based)
    │
    ▼
[5] Word2Vec Training      gensim Word2Vec(vector_size=100, window=5,
    │                      min_count=1, workers=4) — trained on all text
    ▼
[6] Tokenisation           word → integer index dictionary
    │                      OOV tokens → index 0
    ▼
[7] Embedding Matrix       vocab_size × 100 numpy array
    │                      Pre-populated from Word2Vec weights
    ▼
[8] Sequence Padding       Max length = 500 tokens, zero-padded
    │
    ▼
[9] Dataset Splitting      80% Train / 10% Validation / 10% Test
    │
    ▼
[10] Output Artifacts
    ├── data/processed/dataset.npz (X, y arrays)
    ├── data/processed/embedding_matrix.npy
    ├── data/processed/word2vec.model
    └── models/tokenizer_v2.pkl
```

### 5.2 Text Cleaning Details

```python
def clean_text(text: str) -> str:
    text = re.sub(r'[^\x20-\x7E\n\t\r]', '', text)   # Remove non-printable chars
    text = re.sub(r'\s+', ' ', text).strip()           # Normalize whitespace
    return text.lower()                                 # Lowercase
```

### 5.3 Tokenisation

- **Method:** Word-level tokenisation using Python `str.split()`
- **Vocabulary:** All unique whitespace-separated tokens in the training corpus
- **OOV Handling:** Unknown tokens at inference time are mapped to index `0`
- **Minimum token padding:** At least 10 tokens (for TextCNN's minimum filter size)
- **Maximum tokens:** 500 (controlled by `MAX_LEN` in `backend/config.py`)

---

## 6. Training Process

### 6.1 Framework and Environment

| Parameter | Value |
|---|---|
| Deep Learning Framework | PyTorch 2.x |
| Python Version | 3.11+ |
| Hardware Target | CPU (production), GPU (CUDA if available) |
| Training Device | Auto-detected (`cuda` if available, else `cpu`) |

### 6.2 Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Embedding Dimension | 100 | Word2Vec pre-trained, fine-tunable |
| Max Sequence Length | 500 | Tokens per sample |
| Batch Size | 64 | Typical for text classification |
| Learning Rate | 1e-3 (Adam) | Standard for NLP tasks |
| Epochs | Variable | With early stopping on validation loss |
| Optimizer | Adam | Adaptive learning rate |
| Loss Function | BCELoss (Binary Cross-Entropy) | Binary classification |
| Train/Val/Test Split | 80% / 10% / 10% | Stratified by class |

### 6.3 Regularisation Techniques

| Technique | Model | Value |
|---|---|---|
| Dropout | WebshellDetector | 0.3 (after GRU context vector) |
| Dropout | TextCNN | 0.5 (after feature concatenation) |
| Dropout | TextRNN | 0.3 (after LSTM output) |
| Dropout | TransformerClassifier | 0.3 (in encoder layers) |
| Embedding freeze | All | `freeze=False` — fine-tuned during training |

### 6.4 Training Pipeline Workflow

```
1. Load dataset.npz (X, y)
2. Load tokenizer_v2.pkl → build embedding matrix from Word2Vec
3. Instantiate model (vocab_size, embedding_matrix)
4. Define optimizer (Adam) and criterion (BCELoss)
5. For each epoch:
   a. Forward pass (model(X_batch) → predictions)
   b. Compute BCELoss(predictions, y_batch)
   c. Backward pass (loss.backward())
   d. Gradient step (optimizer.step())
   e. Evaluate on validation set
   f. Save checkpoint if val_loss improved
6. Load best checkpoint → save as models/{ModelName}_v2.pth
```

---

## 7. Evaluation Metrics & Performance

### 7.1 Primary Metrics (Classification)

| Metric | Formula | Goal |
|---|---|---|
| Accuracy | (TP+TN) / Total | ≥ 98% |
| Precision | TP / (TP+FP) | ≥ 97% |
| Recall | TP / (TP+FN) | ≥ 97% |
| F1 Score | 2 × (P×R)/(P+R) | ≥ 97% |

### 7.2 Benchmark Results (v2 Models)

| Model | Accuracy | Precision | Recall | F1 Score | Params |
|---|---|---|---|---|---|
| **WebshellDetector** (GRU+Attn) | **98.7%** | 98.5% | 98.9% | 98.7% | ~550K |
| TransformerClassifier | 98.2% | 98.0% | 98.4% | 98.2% | ~3.7M |
| TextCNN | 97.5% | 97.2% | 97.8% | 97.5% | ~546K |
| TextRNN (BiLSTM) | 97.3% | 97.0% | 97.6% | 97.3% | ~1.0M |
| **Ensemble (all 4)** | **~98.7%** | - | - | - | Combined |

### 7.3 Heuristic-Assisted False Positive Mitigation

A key improvement in v2 is the **Heuristic-Assisted Ensemble** logic introduced in `model_manager.py`:

```
IF (model predicts Webshell)
   AND (heuristic_risk < 20)
   AND (dangerous_func_count == 0)
   AND (text length < 100 chars)
   AND (raw_score < 0.95 for single model | < 0.90 for ensemble):
       → Override to: Normal Content
```

This prevents false positives on short, non-code text inputs (e.g., names, sentences) while preserving detection sensitivity for real webshells.

### 7.4 Known Limitations

| Limitation | Details |
|---|---|
| PHP/ASP Only | Models trained exclusively on PHP/ASP datasets; JSP/Python detection is unreliable |
| Obfuscated Text | Novel obfuscation techniques not in training data may reduce accuracy |
| Short Snippets | Very short code fragments (< 20 chars) are blocked at the API level |
| Context Window | Maximum 500 tokens; extremely long files are truncated |
| CPU Inference | No GPU in default deployment; inference latency ~200–600ms per file |

---

## 8. Heuristic Feature Engineering

The heuristic engine runs **independently of the deep learning models** and provides a complementary rule-based risk score for explainability.

### 8.1 Features Extracted

| Feature | Type | Description |
|---|---|---|
| `entropy` | float | Shannon entropy (bits/char) of the source text |
| `entropy_risk` | string | Low / Medium / High |
| `dangerous_func_count` | int | Number of dangerous PHP functions found |
| `dangerous_funcs_found` | list | Names of matched dangerous functions |
| `obfuscation_score` | int | Count of obfuscation pattern matches |
| `heuristic_risk` | float | Composite risk score 0–100 |
| `risk_label` | string | Low / Medium / High |

### 8.2 Dangerous Functions Monitored (60+ patterns)

```
eval, base64_decode, base64_encode, gzinflate, gzuncompress, 
system, exec, passthru, shell_exec, proc_open, popen,
$_GET, $_POST, $_REQUEST, $_COOKIE, $_SERVER,
file_get_contents, file_put_contents, fwrite, fopen,
create_function, call_user_func, assert, preg_replace,
curl_exec, move_uploaded_file, phpinfo, chr(), ord(),
Response.Write, Server.Execute, Runtime.getRuntime, ...
```

### 8.3 Heuristic Risk Scoring

```
Risk Score = min(30, (entropy/8.0) × 30)      # Entropy: 0–30 pts
           + min(50, dangerous_funcs × 5)       # Danger funcs: 0–50 pts
           + min(20, obfuscation_matches × 2)   # Obfuscation: 0–20 pts

Labels:
  ≥ 60 → High Risk
  ≥ 30 → Medium Risk
  < 30  → Low Risk
```

---

## 9. Ensemble & Heuristic-Assisted Prediction

### 9.1 Ensemble Voting Strategy

The ensemble mode runs **all 4 models** simultaneously and averages their raw sigmoid outputs:

```python
avg_raw = mean([score_model1, score_model2, score_model3, score_model4])
is_webshell = avg_raw > THRESHOLD  # Default: 0.5
```

1. Each model outputs a sigmoid probability (0–1)
2. Probabilities are averaged (not majority vote, but score averaging)
3. The final decision is based on the averaged score vs threshold
4. Heuristic-Assisted logic is then applied as a post-processing guard

### 9.2 Per-Model Vote Display

In ensemble mode, the API also returns individual model votes for transparency:

```json
"votes": [
    {"model": "WebshellDetector", "is_webshell": true, "raw_score": 0.95, "confidence": 0.95},
    {"model": "TextCNN",          "is_webshell": false, "raw_score": 0.40, "confidence": 0.60},
    ...
]
```

---

## 10. Deployment Architecture

### 10.1 Local Deployment

```
User Browser → Frontend (localhost:3000) → Backend API (localhost:8000) → Models
```

**Start command:**
```bash
bash start_api.sh
```

### 10.2 Global Deployment via Cloudflare Tunnel

```
Mobile / Remote User
    │
    ▼ HTTPS
Cloudflare Network (*.trycloudflare.com)
    │
    ▼
cloudflared (local process)
    │
    ├──▶ Frontend Tunnel → http://localhost:3000
    └──▶ Backend Tunnel  → http://localhost:8000
```

**Start command:**
```bash
bash start_api.sh --tunnel
```

The script:
1. Starts backend + frontend servers
2. Initiates `cloudflared tunnel` for both ports
3. Extracts tunnel URLs from logs
4. Updates `frontend/js/config.js` with the API tunnel URL
5. Prints terminal QR codes for mobile access

### 10.3 Docker Deployment

```bash
docker build -t webshell-detector .
docker run -p 8000:8000 webshell-detector
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - APP_ENV=production
      - MODEL_VERSION=v2
```

### 10.4 Inference Pipeline

```
Request (text/file)
    │
    ▼
Tokenisation:
  raw_tokens = re.findall(r"[\w']+|[^\w\s]", text)
  tokens = [tokenizer.get(w, 0) for w in raw_tokens]
  tokens = tokens[:500]       # Truncate
  padded = tokens + [0] * max(0, 10 - len(tokens))  # Min-pad for CNN
    │
    ▼
Model Forward Pass:
  with torch.no_grad():
      raw_score = model(tensor(tokens)).item()  # Sigmoid output
    │
    ▼
Heuristic FP Mitigation → Final Decision
    │
    ▼
Heuristic Feature Extraction (parallel)
    │
    ▼
Response (JSON)
```

### 10.5 Latency Profile

| Scenario | Latency (CPU) |
|---|---|
| Single model, text snippet | 200–400ms |
| Ensemble (all 4 models) | 600–1200ms |
| Single file upload | 300–600ms |
| Bulk (50 files) | 15–30 seconds |

---

## 11. API Reference

### 11.1 Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Predict from pasted code text |
| `POST` | `/predict/file` | Predict from uploaded file |
| `POST` | `/predict/bulk` | Bulk file analysis (up to 50 files) |
| `GET` | `/health` | Health check + loaded models list |
| `GET` | `/models` | Available models + benchmark stats |
| `GET` | `/docs` | OpenAPI interactive documentation |

### 11.2 Request Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "<?php eval($_POST[\"cmd\"]); ?>", "model_name": "ensemble"}'
```

### 11.3 Response Schema

```json
{
  "model": "ensemble",
  "is_webshell": true,
  "confidence": 0.872,
  "raw_score": 0.872,
  "label": "Webshell Detected",
  "explanation": "Flagged as a webshell by the Ensemble with 87.2% confidence. Supporting evidence: presence of dangerous functions (eval, $_POST).",
  "features": {
    "entropy": 4.21,
    "entropy_risk": "Medium",
    "dangerous_func_count": 2,
    "dangerous_funcs_found": ["eval", "$_POST"],
    "obfuscation_score": 1,
    "heuristic_risk": 22.5,
    "risk_label": "Low"
  },
  "votes": [...]
}
```

---

## 12. Security & Reliability

### 12.1 API Security

| Measure | Implementation |
|---|---|
| CORS Control | Regex-based origin allowlist (`*.trycloudflare.com` + localhost) |
| Input Validation | Pydantic schemas with `min_length=1` |
| Non-root Docker | `USER appuser` in Dockerfile |
| File Size Limits | `python-multipart` default limits |
| No File Storage | Uploaded files processed in-memory only, never persisted |

### 12.2 Model Integrity

- Model weights are stored as PyTorch `.pth` files, not executable code
- No remote model loading — all models loaded from local disk at startup
- Model loading wrapped in try/catch; partial loads are gracefully handled

### 12.3 Adversarial Input Protection

- **Minimum length enforcement:** Frontend blocks inputs < 20 characters
- **Heuristic FP Mitigation:** Overrides dubious high-confidence ML predictions when security evidence is absent
- **Threshold configurability:** `PREDICTION_THRESHOLD` env var allows tuning sensitivity

### 12.4 Logging & Monitoring

- MLOps Drift Monitor (`src/evaluation/drift_monitor.py`) tracks prediction confidence distribution
- All API requests logged via Uvicorn access logs (`server.log`)
- Docker `HEALTHCHECK` pings `/health` every 30 seconds

---

## 13. Infrastructure Requirements

### 13.1 Minimum System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB (models ~320 MB) | 5 GB |
| Python | 3.11+ | 3.13 |
| GPU | Not required | CUDA GPU (speeds up ensemble 5–10×) |
| OS | Linux / macOS / Windows (WSL2) | Ubuntu 22.04 LTS |

### 13.2 Network Requirements

| Endpoint | Port | Protocol |
|---|---|---|
| Backend API | 8000 | HTTP/HTTPS |
| Frontend SPA | 3000 | HTTP |
| Cloudflare Tunnel | 443 (outbound) | HTTPS |

---

## 14. Reproducibility Guide

### 14.1 Environment Setup

```bash
# 1. Clone repository
git clone <repo_url>
cd webshell_detector

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env as needed (MODEL_VERSION=v2, APP_ENV=development, etc.)
```

### 14.2 Training the Models

```bash
# 1. Place raw data
#    data/raw/malicious/  ← PHP/ASP webshell files
#    data/raw/normal/     ← Legitimate PHP files

# 2. Run preprocessing
python src/data/preprocessing.py

# 3. Train all models (see training scripts in src/models/)
python train_all.py  # or retrain_v2.py

# 4. Models saved to models/ directory
```

### 14.3 Starting the Application

```bash
# Local only
bash start_api.sh

# With Cloudflare Tunnel (global access + QR codes)
bash start_api.sh --tunnel
```

### 14.4 Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_VERSION` | `v2` | Which model files to load |
| `PREDICTION_THRESHOLD` | `0.5` | Classification decision boundary |
| `BULK_MAX_FILES` | `50` | Maximum files in bulk scan |
| `APP_HOST` | `0.0.0.0` | API bind address |
| `APP_PORT` | `8000` | API port |
| `APP_ENV` | `development` | Environment tag |
| `DEBUG` | `false` | Enable debug logs |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

### 14.5 Docker Reproducibility

```bash
# Build
docker build -t webshell-detector:v2 .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_VERSION=v2 \
  -e APP_ENV=production \
  webshell-detector:v2
```

---

## 15. Versioning & Model Management

### 15.1 Current Model Versions

| Artifact | Version | Date | Size |
|---|---|---|---|
| WebshellDetector | v2 | 2026-03 | 78.6 MB |
| TextCNN | v2 | 2026-03 | 78.7 MB |
| TextRNN | v2 | 2026-03 | 79.1 MB |
| TransformerClassifier | v2 | 2026-03 | 81.7 MB |
| Tokenizer | v2 | 2026-03 | 4.9 MB |

### 15.2 Version Naming Convention

```
{ModelName}_{version}.pth
tokenizer_{version}.pkl

v1 → Initial release models
v2 → Retrained with larger dataset and regularisation improvements
v3 → (Future) Extended to ASP.NET / JSP variants
```

### 15.3 Switching Model Versions

```bash
export MODEL_VERSION=v2  # or set in .env
```

---

## 16. Monitoring & Maintenance

### 16.1 Model Drift Detection

The `DriftMonitor` class in `src/evaluation/drift_monitor.py` tracks:
- Rolling confidence score distribution
- Alert threshold: If average confidence drops significantly below training-time distribution, retraining is recommended

### 16.2 Retraining Procedure

1. Collect new labeled PHP/ASP samples (malicious/normal)
2. Add to `data/raw/` directories
3. Re-run preprocessing: `python src/data/preprocessing.py`
4. Re-run training: `python retrain_v2.py`
5. Validate on held-out test set (target: ≥ 98% accuracy)
6. Replace `.pth` files in `models/` directory
7. Restart the API server

### 16.3 Performance Monitoring Checklist

| Item | Frequency | Tool |
|---|---|---|
| Review `/health` endpoint | Daily | curl / Uptime monitor |
| Check `server.log` for errors | Daily | tail -f server.log |
| Validate prediction accuracy on known samples | Weekly | Manual test cases |
| Monitor drift metrics | Weekly | DriftMonitor |
| Full retraining with new data | Quarterly | retrain_v2.py |

---

## 17. Ethical Considerations

### 17.1 Bias Analysis

| Potential Bias | Mitigation |
|---|---|
| PHP-centric training data | Clearly documented; ASP/JSP users should validate results |
| Obfuscation blind spots | Novel obfuscation not in training may slip through — heuristics provide a secondary layer |
| False positive risk on legitimate code | Heuristic-Assisted FP Mitigation added to reduce false alarms |

### 17.2 Transparency

- All model architectures are open source and documented here
- The "AI Reasoning" box in the UI explains every prediction in natural language
- Raw model scores are always shown alongside confidence percentages
- Ensemble vote breakdowns are visible per-model

### 17.3 Responsible Use Guidelines

> ⚠️ **This tool is designed for defensive security use only.**

- Do NOT use this system to develop or test offensive webshell payloads
- Treat all uploaded code as potentially sensitive — the system processes files in-memory only and does not persist them
- If deployed on a shared server, ensure access control to prevent unauthorized use
- False negatives (missed webshells) are possible — do not rely solely on this tool for production security

### 17.4 Research Ethics

This system was developed as part of a M.Tech dissertation in AI security. All training data consists of publicly available webshell samples from security research repositories. No live compromised systems were involved.

---

## 18. Future Improvements

### 18.1 Model Architecture

| Improvement | Priority | Description |
|---|---|---|
| CodeBERT / GraphCodeBERT | High | Pre-trained code-specific transformer for better PHP understanding |
| AST-based Features | Medium | Parse PHP Abstract Syntax Trees for structural features |
| One-class classification | Medium | Train on normal code only; webshells are anomalies |
| Larger Transformer | Low | Scale to 12-layer transformer with larger embedding dimension |

### 18.2 Dataset

| Improvement | Priority |
|---|---|
| Expand to JSP / Python / Node.js webshells | High |
| Include recent AI-generated webshells | High |
| Add obfuscated variant augmentation | Medium |
| Partner with AV vendors for labeled data | Low |

### 18.3 System

| Improvement | Priority |
|---|---|
| GPU inference support (Docker + CUDA) | Medium |
| Streaming bulk scan with progress WebSocket | Medium |
| REST API authentication (API keys) | High (pre-production) |
| Model Registry (MLflow / Neptune) | Low |
| SHAP / LIME explainability integration | Medium |

---

## Appendix A: File Structure Reference

```
webshell_detector/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── model_manager.py     # Model loading, inference, ensemble
│   ├── config.py            # Centralised configuration
│   └── schemas.py           # Pydantic request/response schemas
├── src/
│   ├── data/
│   │   ├── preprocessing.py # Data pipeline: clean → tokenize → embed
│   │   └── data_loader.py   # Dataset loader
│   ├── models/
│   │   └── model.py         # PyTorch model architectures (all 4)
│   ├── features/
│   │   └── feature_engineering.py  # Heuristic risk analysis
│   └── evaluation/
│       └── drift_monitor.py # MLOps confidence tracking
├── models/                  # Trained model weights
│   ├── WebshellDetector_v2.pth
│   ├── TextCNN_v2.pth
│   ├── TextRNN_v2.pth
│   ├── TransformerClassifier_v2.pth
│   └── tokenizer_v2.pkl
├── frontend/                # Web dashboard (vanilla JS SPA)
│   ├── index.html
│   ├── css/main.css
│   └── js/ (api.js, ui.js, app.js, config.js)
├── data/
│   ├── raw/                 # Original training files
│   └── processed/           # Preprocessed arrays
├── docs/
│   └── AI_MODEL_DOCUMENTATION.md  # ← This file
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── start_api.sh
└── .env.example
```

---

## Appendix B: Glossary

| Term | Definition |
|---|---|
| Webshell | A malicious script (typically PHP/ASP) planted on a server to give attackers remote control |
| BiGRU | Bidirectional Gated Recurrent Unit — processes sequences both forward and backward |
| BiLSTM | Bidirectional Long Short-Term Memory — LSTM variant with bidirectional processing |
| TextCNN | Text Convolutional Neural Network — uses 1D convolutions over n-gram windows |
| Ensemble | A combination of multiple models whose predictions are aggregated |
| Shannon Entropy | Measure of information density — high entropy often indicates obfuscation |
| Word2Vec | Neural word embedding technique that maps words to dense vectors |
| OOV | Out-of-Vocabulary — tokens not seen during training |
| BCELoss | Binary Cross-Entropy Loss — standard loss for binary classification |
| CORS | Cross-Origin Resource Sharing — browser security mechanism for API access |

---

*Documentation generated for: Webshell Detector v2 — M.Tech Dissertation*  
*Last Updated: March 2026*  
*Maintained by: Nikhil Kumar*
