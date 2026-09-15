# SatQuery AI - Root Launcher
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$scriptPath = Join-Path $PSScriptRoot "satquery-ai\run_satquery.ps1"
if (Test-Path $scriptPath) {
    & $scriptPath
} else {
    Write-Error "Error: satquery-ai\run_satquery.ps1 not found."
}
