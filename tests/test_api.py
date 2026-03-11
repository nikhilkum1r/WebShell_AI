#!/usr/bin/env python3
"""
Webshell Detector v2 — Comprehensive API Smoke Test
Tests all endpoints including ensemble and feature extraction.

Usage:  python3 test_api.py
        python3 test_api.py --host http://myserver.com:8000
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
for i, arg in enumerate(sys.argv):
    if arg == "--host" and i + 1 < len(sys.argv):
        BASE = sys.argv[i + 1]

# ── Sample payloads ────────────────────────────────────────────────────────
WEBSHELL = "<?php eval($_POST['cmd']); system($_GET['exec']); base64_decode($_POST['x']); ?>"
NORMAL = "<?php echo 'Hello World'; $name = 'Alice'; echo 'Welcome ' . $name; ?>"

GRN = "\033[92m"
RED = "\033[91m"
CYN = "\033[96m"
YLW = "\033[93m"
BLD = "\033[1m"
RST = "\033[0m"

passed = 0
total = 0


def req(method, path, data=None, form=None, files=None):
    url = BASE + path
    if form or files:
        boundary = "----FormBoundary7MA4YWxkTrZu0gW"
        body_parts = []
        for k, v in (form or {}).items():
            body_parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}'.encode()
            )
        for k, (fname, fcontent) in (files or {}).items():
            body_parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\nContent-Type: text/plain\r\n\r\n'.encode()
                + fcontent
            )
        body = b"\r\n".join(body_parts) + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    elif data:
        body = json.dumps(data).encode()
        headers = {"Content-Type": "application/json"}
    else:
        body, headers = None, {}

    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code


def check(label, passed_cond, note=""):
    global passed, total
    total += 1
    sym = f"{GRN}✓{RST}" if passed_cond else f"{RED}✗{RST}"
    extra = f"  {YLW}({note}){RST}" if note else ""
    print(f"  {sym}  {label}{extra}")
    if passed_cond:
        passed += 1
    return passed_cond


def section(title):
    print(f"\n{CYN}{BLD}── {title} ──{RST}")


def main():
    print(f"\n{BLD}{CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RST}")
    print(f"{BLD}{CYN}  Webshell Detector v2 — Smoke Test{RST}")
    print(f"{BLD}{CYN}  Target: {BASE}{RST}")
    print(f"{CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RST}")

    # ── 1. Info endpoints ─────────────────────────────────────────────────
    section("Info Endpoints")
    r, s = req("GET", "/")
    check("GET /  → 200 + API info", s == 200 and "name" in r)
    r, s = req("GET", "/health")
    check("GET /health → status == healthy", r.get("status") == "healthy")
    check("GET /health → version field present", "version" in r)
    check("GET /health → device field present", "device" in r, r.get("device", "—"))
    r, s = req("GET", "/models")
    check("GET /models → ensemble in list", "ensemble" in r.get("available_models", []))
    r, s = req("GET", "/stats")
    check("GET /stats → 4 benchmark rows", len(r) == 4)

    # ── 2. Single-model predictions ───────────────────────────────────────
    section("Single-Model Predictions")
    MODELS = ["WebshellDetector", "TextCNN", "TextRNN", "TransformerClassifier"]
    for m in MODELS:
        t0 = time.perf_counter()
        r, s = req("POST", "/predict", {"text": WEBSHELL, "model_name": m})
        ms = (time.perf_counter() - t0) * 1000
        ok = s == 200 and "is_webshell" in r and "confidence" in r and "features" in r
        check(f"POST /predict [{m}] webshell sample", ok, f"{ms:.0f}ms")

    r, s = req("POST", "/predict", {"text": NORMAL, "model_name": "WebshellDetector"})
    check("POST /predict normal content has is_webshell field", "is_webshell" in r)

    # ── 3. Ensemble ───────────────────────────────────────────────────────
    section("Ensemble Prediction")
    r, s = req("POST", "/predict", {"text": WEBSHELL, "model_name": "ensemble"})
    check("POST /predict [ensemble] → 200", s == 200)
    check("POST /predict [ensemble] → model == 'ensemble'", r.get("model") == "ensemble")
    check("POST /predict [ensemble] → votes list has 4 entries", len(r.get("votes") or []) == 4)
    check("POST /predict [ensemble] → features present", r.get("features") is not None)

    # ── 4. Feature extraction ─────────────────────────────────────────────
    section("Heuristic Feature Extraction")
    r, s = req("POST", "/predict", {"text": WEBSHELL, "model_name": "WebshellDetector"})
    feat = r.get("features", {})
    check(
        "features.entropy is float > 0",
        isinstance(feat.get("entropy"), float) and feat["entropy"] > 0,
    )
    check("features.dangerous_func_count > 0 for webshell", feat.get("dangerous_func_count", 0) > 0)
    check("features.heuristic_risk 0–100", 0 <= feat.get("heuristic_risk", -1) <= 100)

    # ── 5. File upload ────────────────────────────────────────────────────
    section("File Upload")
    ws_content = WEBSHELL.encode()
    r, s = req(
        "POST",
        "/predict/file",
        form={"model_name": "WebshellDetector"},
        files={"file": ("test_webshell.php", ws_content)},
    )
    check("POST /predict/file → 200", s == 200)
    check("POST /predict/file → features present", r.get("features") is not None)

    # ── 6. Bulk scan ──────────────────────────────────────────────────────
    section("Bulk Scan")
    fd_files = {"file": (f"file_{i}.php", WEBSHELL.encode()) for i in range(3)}
    # Build a proper multipart with multiple 'files' fields
    boundary = "----BulkBoundary7MA4YWxkTrZu0gW"
    parts = []
    for i in range(3):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="file_{i}.php"\r\nContent-Type: text/plain\r\n\r\n'.encode()
            + WEBSHELL.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="model_name"\r\n\r\nWebshellDetector'.encode()
    )
    body = b"\r\n".join(parts) + f"\r\n--{boundary}--\r\n".encode()
    rq = urllib.request.Request(
        BASE + "/predict/bulk",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(rq, timeout=20) as resp:
            bulk_r = json.loads(resp.read())
            bulk_s = resp.status
    except urllib.error.HTTPError as e:
        bulk_r = json.loads(e.read())
        bulk_s = e.code
    check("POST /predict/bulk → 200", bulk_s == 200)
    check("POST /predict/bulk → 3 results", isinstance(bulk_r, list) and len(bulk_r) == 3)

    # ── 7. Analysis endpoints ─────────────────────────────────────────────
    section("Analysis Endpoints")
    r, s = req("POST", "/analyse/text", {"text": WEBSHELL, "model_name": "WebshellDetector"})
    check("POST /analyse/text → 200 with features", s == 200 and r.get("features") is not None)

    # ── Summary ───────────────────────────────────────────────────────────
    color = GRN if passed == total else RED
    print(f"\n{color}{BLD}  {passed}/{total} tests passed{RST}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
