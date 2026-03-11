"""
FastAPI Application v2 — Production-ready Webshell Detector.

New in v2:
  • POST /predict supports "ensemble" as model_name
  • GET  /stats — model benchmark metrics
  • POST /analyse/text — full analysis (ML + heuristics) in one call
  • Request IDs + structured logging on every request
  • Global 404 & 500 handlers
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    APP_ENV,
    BULK_MAX_FILES,
    CORS_ORIGINS,
)
from backend.model_manager import get_manager
from backend.schemas import (
    BulkPredictionResult,
    FileFeatureSchema,
    HealthResponse,
    ModelStatsResponse,
    PredictionResult,
    SingleModelVote,
    TextPredictRequest,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("webshell_api")


# ── Startup / shutdown ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {API_TITLE} v{API_VERSION} [{APP_ENV}]")
    try:
        mgr = get_manager()
        logger.info(f"✅ {len(mgr.models)} models on {mgr.device}")
    except Exception as exc:
        logger.critical(f"❌ STARTUP FAILED: {exc}")
        raise
    yield
    logger.info("🔴 Shutdown complete.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Info", "description": "Health checks and metadata"},
        {"name": "Detection", "description": "Core prediction endpoints"},
        {"name": "Analysis", "description": "Full analysis with explainability"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="http://localhost:.*|http://127.0.0.1:.*|https://.*\.trycloudflare\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request ID + timing ────────────────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    request.state.request_id = rid
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time"] = f"{elapsed:.1f}ms"
    logger.info(
        f"[{rid}] {request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)"
    )
    return response


# ── Exception handlers ─────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404, content={"detail": f"Endpoint '{request.url.path}' not found."}
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Helpers ────────────────────────────────────────────────────────────────
def _build_result(data: dict) -> PredictionResult:
    """Convert raw dict from ModelManager to PredictionResult schema."""
    features = FileFeatureSchema(**data["features"]) if data.get("features") else None
    votes = [SingleModelVote(**v) for v in data["votes"]] if data.get("votes") else None
    return PredictionResult(
        model=data["model"],
        is_webshell=data["is_webshell"],
        confidence=data["confidence"],
        raw_score=data["raw_score"],
        label=data["label"],
        explanation=data.get("explanation"),
        features=features,
        votes=votes,
    )


async def _read_upload(file: UploadFile) -> str:
    raw = await file.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ═══════════════════════════════════════════════════════════════
# Routes — Info
# ═══════════════════════════════════════════════════════════════


@app.get("/", tags=["Info"])
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    mgr = get_manager()
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        device=str(mgr.device),
        available_models=mgr.available_models(),
        environment=APP_ENV,
        message=f"{len(mgr.models)} models loaded and ready.",
    )


@app.get("/models", tags=["Info"])
def list_models():
    return {"available_models": get_manager().available_models()}


@app.get("/stats", response_model=list[ModelStatsResponse], tags=["Info"])
def model_stats():
    """Return benchmark accuracy/F1/recall for each model (from dissertation)."""
    return [ModelStatsResponse(**b) for b in get_manager().benchmarks()]


# ═══════════════════════════════════════════════════════════════
# Routes — Detection
# ═══════════════════════════════════════════════════════════════


@app.post("/predict", response_model=PredictionResult, tags=["Detection"])
def predict_text(body: TextPredictRequest):
    """
    Predict from raw JSON text.
    Set `model_name` to `"ensemble"` to run all 4 models and majority-vote.
    """
    try:
        mgr = get_manager()
        data = (
            mgr.ensemble_predict(body.text)
            if body.model_name == "ensemble"
            else mgr.predict(body.text, body.model_name)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _build_result(data)


@app.post("/predict/file", response_model=PredictionResult, tags=["Detection"])
async def predict_file(
    file: UploadFile = File(...),
    model_name: str = Form("WebshellDetector"),
):
    """Upload a single file and get a prediction."""
    text = await _read_upload(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        mgr = get_manager()
        data = (
            mgr.ensemble_predict(text, file.filename)
            if model_name == "ensemble"
            else mgr.predict(text, model_name, file.filename)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _build_result(data)


@app.post("/predict/bulk", response_model=list[BulkPredictionResult], tags=["Detection"])
async def predict_bulk(
    files: list[UploadFile] = File(...),
    model_name: str = Form("WebshellDetector"),
):
    """Batch-scan up to 50 files in one request."""
    if len(files) > BULK_MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {BULK_MAX_FILES} files per request.")
    mgr = get_manager()
    results = []
    for f in files:
        text = await _read_upload(f)
        try:
            data = (
                mgr.ensemble_predict(text, f.filename)
                if model_name == "ensemble"
                else mgr.predict(text, model_name, f.filename)
            )
        except Exception as exc:
            data = {
                "model": model_name,
                "is_webshell": False,
                "confidence": 0.0,
                "raw_score": 0.0,
                "label": f"Error: {exc}",
                "features": None,
                "votes": None,
            }
        results.append(BulkPredictionResult(filename=f.filename, result=_build_result(data)))
    return results


# ═══════════════════════════════════════════════════════════════
# Routes — Analysis (ML + Heuristics combined report)
# ═══════════════════════════════════════════════════════════════


@app.post("/analyse/text", response_model=PredictionResult, tags=["Analysis"])
def analyse_text(body: TextPredictRequest):
    """
    Full analysis: runs the selected model AND heuristic feature extraction.
    The response includes both the ML prediction and extracted file features.
    Identical to /predict but always forces feature extraction regardless of model.
    """
    return predict_text(body)  # features already included in v2


@app.post("/analyse/file", response_model=PredictionResult, tags=["Analysis"])
async def analyse_file(
    file: UploadFile = File(...),
    model_name: str = Form("ensemble"),
):
    """Full analysis of an uploaded file. Defaults to ensemble mode."""
    return await predict_file(file=file, model_name=model_name)
