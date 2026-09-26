@echo off
REM docRenderCut launcher (Windows) - pure ASCII to avoid codepage garbling
setlocal
cd /d "%~dp0"

python -c "import fastapi, uvicorn, cv2, pymupdf" 2>nul
if errorlevel 1 (
  echo [docRenderCut] installing requirements.txt ...
  python -m pip install -r requirements.txt
)

echo [docRenderCut] starting http://127.0.0.1:8848
start "" http://127.0.0.1:8848
python app.py
endlocal
