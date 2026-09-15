@echo off
if exist "%~dp0satquery-ai\run_satquery.bat" (
    call "%~dp0satquery-ai\run_satquery.bat"
) else (
    echo Error: satquery-ai folder not found in %~dp0
    pause
)
