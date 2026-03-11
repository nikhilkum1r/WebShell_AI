#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# start_api.sh — Launch the Webshell Detector API (v2)
# Usage: bash start_api.sh [--prod]
# ─────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROD=""
TUNNEL=""
APP_ENV="development"
WORKERS=1
for arg in "$@"; do
  [[ "$arg" == "--prod" ]] && PROD="true" && APP_ENV="production" && WORKERS=4
  [[ "$arg" == "--tunnel" ]] && TUNNEL="true"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🛡  Webshell Detector API v2"
echo "  Mode: ${APP_ENV}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Prefer venv if present, else system Python
if [ -f "venv/bin/python3" ]; then
  PYTHON="./venv/bin/python3"
  PIP="./venv/bin/pip"
  UVICORN="./venv/bin/uvicorn"
else
  PYTHON="python3"
  PIP="pip3"
  UVICORN="python3 -m uvicorn"
fi

echo "🐍  Python: $($PYTHON --version)"

# ── Dependency Check ──────────────────────────────────────────
if ! $PYTHON -c "import fastapi" 2>/dev/null; then
  echo "📦  Installing API dependencies…"
  $PIP install fastapi uvicorn[standard] python-multipart -q
fi

# ── Cleanup ───────────────────────────────────────────────────
echo "🧹  Cleaning up previous processes…"
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true

# ── Setup ─────────────────────────────────────────────────────
# Reset config.js
mkdir -p frontend/js
echo "window.WEBSHELL_CONFIG = {};" > frontend/js/config.js

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "📄  Created .env from .env.example"
fi

export APP_ENV=$APP_ENV

# ── Starting Services ─────────────────────────────────────────
echo ""
echo "🚀  Starting Webshell Detector Services…"

# 1. Start Backend (in background)
if [[ -n "$PROD" ]]; then
  $UVICORN backend.main:app --host 0.0.0.0 --port 8000 --workers $WORKERS --log-level warning > server.log 2>&1 &
else
  $UVICORN backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level info > server.log 2>&1 &
fi
BACKEND_PID=$!
echo "✅  Backend PID $BACKEND_PID → http://localhost:8000"

# 2. Start Frontend (in background)
$PYTHON -m http.server 3000 --directory frontend > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✅  Frontend PID $FRONTEND_PID → http://localhost:3000"

# 3. Start Tunnels (if requested)
if [[ -n "$TUNNEL" ]]; then
  echo "🌐  Starting Cloudflare Tunnels…"
  if [ ! -f "./cloudflared" ]; then
    echo "⬇️  Downloading cloudflared…"
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
  fi

  # Reset logs
  > api_tunnel.log
  > gui_tunnel.log

  # Start API Tunnel
  ./cloudflared tunnel --url http://localhost:8000 > api_tunnel.log 2>&1 &
  API_LT_PID=$!
  
  # Start GUI Tunnel
  ./cloudflared tunnel --url http://localhost:3000 > gui_tunnel.log 2>&1 &
  GUI_LT_PID=$!
  
  echo "⏳  Waiting for tunnel URLs…"
  sleep 10
  
  API_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' api_tunnel.log | head -n 1)
  GUI_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' gui_tunnel.log | head -n 1)
  
  if [[ -z "$API_URL" ]]; then
      echo "⚠️  Failed to retrieve API Tunnel URL. Check api_tunnel.log"
  else
      echo "✅  API Tunnel:      $API_URL"
      echo "window.WEBSHELL_CONFIG = { API_BASE: '$API_URL' };" > frontend/js/config.js
      if command -v qrencode >/dev/null 2>&1; then
        echo "📱  Scan QR for API:"
        qrencode -t ANSIUTF8 "$API_URL"
      fi
  fi
  
  if [[ -z "$GUI_URL" ]]; then
      echo "⚠️  Failed to retrieve GUI Tunnel URL. Check gui_tunnel.log"
  else
      echo "✅  Dashboard Tunnel: $GUI_URL"
      if command -v qrencode >/dev/null 2>&1; then
        echo "📱  Scan QR for Dashboard:"
        qrencode -t ANSIUTF8 "$GUI_URL"
      fi
  fi
fi

echo ""
echo "📖  Documentation: http://localhost:8000/docs"
echo "🌐  Dashboard:     http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for backend to be ready
sleep 2

# Keep script running to manage child processes
trap "kill $BACKEND_PID $FRONTEND_PID ${API_LT_PID:-} ${GUI_LT_PID:-}; exit" INT TERM
wait
