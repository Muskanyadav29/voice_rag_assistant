@echo off
cd /d "%~dp0backend"
..\.venv\Scripts\uvicorn.exe app:app --reload --host 0.0.0.0 --port 8000
