@echo off
REM One-click Python diagnostic for the Rule-Based Intelligent Chatbot.
REM Put this file in the same folder as run_windows.bat and double-click it.
REM It does NOT run bare "python --version", so it will not open the
REM Microsoft Store even if "python" is still just a Store shortcut.
cd /d %~dp0
echo ============================================================
echo Python check - copy everything in this window to Maren
echo ============================================================
echo.
echo --- Where Windows finds Python ---
where py
where python
echo.
echo --- The "py" launcher (safe to run) ---
py --version
if errorlevel 1 echo py launcher: NOT WORKING
echo.
echo --- The chatbot's virtual environment, if already created ---
if exist venv\Scripts\python.exe (
    echo venv python: FOUND
    venv\Scripts\python.exe --version
) else (
    echo venv python: NOT FOUND yet
    echo That just means run_windows.bat has not finished creating it.
)
echo.
echo --- How to read this ---
echo If "where python" points to WindowsApps, that is the Microsoft
echo Store shortcut, not real Python - install Python from
echo https://www.python.org/downloads/ with "Add python.exe to PATH"
echo ticked, close all terminal windows, then run this check again.
echo If "py --version" shows a version like Python 3.12.x, then
echo run_windows.bat should work - double-click it again.
echo ============================================================
pause
