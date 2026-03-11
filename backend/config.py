"""
Centralised configuration for the Webshell Detector API.
All settings are read from environment variables (or a .env file).
Relative project paths are resolved automatically — no hardcoded paths.
"""

import os
from pathlib import Path

# ── Try loading .env if python-dotenv is available ─────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# ── Project root (one level above this backend/ directory) ─────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model versioning ──────────────────────────────────────────────
# Change to "v2" after running retrain_v2.py to use more powerful models.
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
_v_sfx = f"_{MODEL_VERSION}" if MODEL_VERSION != "v1" else ""

# ── Model artefacts ────────────────────────────────────────────────────────
TOKENIZER_PATH = BASE_DIR / "models" / f"tokenizer{_v_sfx}.pkl"
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "WebshellDetector": f"WebshellDetector{_v_sfx}.pth",
    "TextCNN": f"TextCNN{_v_sfx}.pth",
    "TextRNN": f"TextRNN{_v_sfx}.pth",
    "TransformerClassifier": f"TransformerClassifier{_v_sfx}.pth",
}

# ── Embedding ──────────────────────────────────────────────────────────────
EMBEDDING_DIM = 100
MAX_LEN = 500

# ── Prediction ─────────────────────────────────────────────────────────────
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
BULK_MAX_FILES = int(os.getenv("BULK_MAX_FILES", "50"))

# ── API metadata ───────────────────────────────────────────────────────────
API_TITLE = "Webshell Detector API"
API_VERSION = "2.0.0"
API_DESCRIPTION = (
    "Production-grade REST API for AI-powered webshell detection. "
    "Supports text analysis, single-file upload, bulk scanning, "
    "model ensembling, and heuristic feature extraction for explainability."
)

# ── Server ─────────────────────────────────────────────────────────────────
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ── CORS ───────────────────────────────────────────────────────────────────
_raw_origins = os.getenv(
    "CORS_ORIGINS", "*"
)
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
