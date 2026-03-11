# ─────────────────────────────────────────────────────────────────
# Dockerfile — Webshell Detector API
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install CPU-only PyTorch first to keep image smaller
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/      ./src/
COPY backend/  ./backend/
COPY Output/   ./Output/

# Use non-root user
USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python3", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
