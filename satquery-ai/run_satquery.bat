@echo off
title SatQuery AI - Launcher
echo ==========================================================
echo    SATQUERY AI: ASK QUESTIONS. UNDERSTAND EARTH.
echo    SIH26167 Remote-Sensing Intelligent Assistant
echo ==========================================================

echo [1/2] Starting Backend Server on http://localhost:8000 ...
start "SatQuery Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Frontend Dashboard on http://localhost:5173 ...
start "SatQuery Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ==========================================================
echo SatQuery AI servers launched!
echo Frontend Dashboard: http://localhost:5173
echo Backend API Docs:   http://localhost:8000/docs
echo ==========================================================
