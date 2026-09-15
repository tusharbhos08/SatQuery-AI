# SatQuery AI - One-Command Launcher (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   SATQUERY AI: ASK QUESTIONS. UNDERSTAND EARTH.          " -ForegroundColor Cyan
Write-Host "   SIH26167 Remote-Sensing Intelligent Assistant          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Start FastAPI Backend
Write-Host "[1/2] Launching Backend on http://localhost:8000 ..." -ForegroundColor Yellow
$BackendJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; cd '$PSScriptRoot\backend'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" -PassThru

Start-Sleep -Seconds 2

# Start Vite Frontend
Write-Host "[2/2] Launching Frontend on http://localhost:5173 ..." -ForegroundColor Green
$FrontendJob = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; cd '$PSScriptRoot\frontend'; npm.cmd run dev -- --host" -PassThru

Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host "SatQuery AI is now running!" -ForegroundColor Green
Write-Host "• Frontend UI: http://localhost:5173" -ForegroundColor White
Write-Host "• Backend API & Swagger Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
