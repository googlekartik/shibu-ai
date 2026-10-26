@echo off
REM Start Shibu AI - double-click this file, no PowerShell needed.
REM
REM Why this file exists: Kartik's project has ended up in a nested
REM layout on his PC, like this:
REM   Downloads\rule_based_chatbot\rule_based_chatbot\app.py
REM while the venv he made first is in the outer folder. Opening a new
REM PowerShell window also starts in the home folder, not the project
REM folder, and PowerShell blocks venv activation by default. All three
REM traps have bitten him today, so this script removes all three:
REM   1. It finds app.py itself - outer folder, inner folder, or even
REM      the Downloads folder one level above both.
REM   2. It never runs Activate.ps1, so the execution-policy block
REM      cannot happen. It calls venv\Scripts\python.exe directly.
REM   3. It creates the venv inside the folder where app.py actually
REM      is, so the venv and the app can never be in different folders.
REM
REM Put this file anywhere inside Downloads\rule_based_chatbot
REM (outer folder, inner folder, or Downloads itself) and double-click it.
cd /d %~dp0

REM Find app.py, without moving if it is already here.
if not exist app.py if exist rule_based_chatbot\app.py cd rule_based_chatbot
if not exist app.py if exist rule_based_chatbot\rule_based_chatbot\app.py cd rule_based_chatbot\rule_based_chatbot

if not exist app.py (
    echo ============================================================
    echo Could not find app.py near this file.
    echo.
    echo This file is in: %CD%
    echo Put start_shibu.bat inside Downloads\rule_based_chatbot
    echo ^(the outer folder, or the inner one with the same name^),
    echo then double-click it again.
    echo ============================================================
    dir
    pause
    exit /b 1
)

echo Found Shibu AI here:
echo %CD%
echo.

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
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Download Python 3.11 or newer for Windows and run it.
    echo   3. On the FIRST install screen, tick "Add python.exe to PATH"
    echo      and then click "Install Now".
    echo   4. Close this window, then double-click start_shibu.bat again.
    echo ============================================================
    pause
    exit /b 1
)

echo Using Python command: %PYCMD%
%PYCMD% --version
if not exist venv\Scripts\python.exe (
    echo Creating the virtual environment in this folder ^(first run only^)...
    %PYCMD% -m venv venv
)
if not exist venv\Scripts\python.exe (
    echo ============================================================
    echo Could not create the virtual environment in this folder.
    echo Check that this folder is writable, then try again.
    echo ============================================================
    pause
    exit /b 1
)

echo Installing requirements ^(first run only, then it is quick^)...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ============================================================
    echo Installing requirements failed. Check your internet connection,
    echo then double-click start_shibu.bat again - it picks up where
    echo it left off and does not start from scratch.
    echo ============================================================
    pause
    exit /b 1
)

start http://127.0.0.1:5000
echo.
echo Shibu AI is starting. Keep this window open while you use it.
echo To stop it, press Ctrl+C in this window, then double-click
echo start_shibu.bat again to restart it - no PowerShell steps needed.
venv\Scripts\python.exe app.py
pause
