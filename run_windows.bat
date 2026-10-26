@echo off
REM One-click setup for the Rule-Based Intelligent Chatbot (Windows)
REM Just double-click this file. It creates a virtual environment once,
REM installs the requirements, opens the browser, and starts the app.
REM
REM If you see "Python was not found... Microsoft Store", it means real
REM Python is not installed yet (Windows is showing a store shortcut,
REM not Python). Install Python from python.org first - see below.
cd /d %~dp0

REM Find a real Python. Try the "py" launcher first, because plain
REM "python" on Windows can be a Microsoft Store shortcut, not Python.
set PYCMD=
py --version >nul 2>&1
if not errorlevel 1 set PYCMD=py
if not defined PYCMD (
    python --version >nul 2>&1
    if not errorlevel 1 set PYCMD=python
)
if not defined PYCMD (
    echo ============================================================
    echo Python was not found on this PC.
    echo.
    echo Fix it like this:
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Download Python 3.11 or newer for Windows and run it.
    echo   3. On the FIRST install screen, tick "Add python.exe to PATH"
    echo      and then click "Install Now".
    echo   4. Close this window, then double-click run_windows.bat again.
    echo.
    echo If it still says "Python was not found" after installing:
    echo   Settings ^> Apps ^> Advanced app settings ^> App execution aliases
    echo   Turn OFF "python.exe" and "python3.exe", close all terminals,
    echo   then double-click run_windows.bat again.
    echo ============================================================
    pause
    exit /b 1
)

echo Using Python command: %PYCMD%
%PYCMD% --version
if not exist venv (
    echo Creating virtual environment (first run only)...
    %PYCMD% -m venv venv
)
call venv\Scripts\activate
echo Installing requirements (first run only, then it is quick)...
pip install -r requirements.txt
start http://127.0.0.1:5000
echo.
echo Chatbot is starting. Keep this window open while you use it.
echo To stop it, press Ctrl+C in this window.
python app.py
pause
