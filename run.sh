#!/usr/bin/env bash
# docRenderCut launcher (macOS / Linux / Git Bash)
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

if ! "$PY" -c "import fastapi, uvicorn, cv2, pymupdf" 2>/dev/null; then
  echo "[docRenderCut] installing requirements.txt ..."
  "$PY" -m pip install -r requirements.txt
fi

echo "[docRenderCut] starting http://127.0.0.1:8848"
"$PY" app.py
