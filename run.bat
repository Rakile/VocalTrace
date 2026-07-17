@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment missing.
  echo.
  echo Double-click installation.bat first ^(one-time setup^).
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launch.py
if errorlevel 1 pause
