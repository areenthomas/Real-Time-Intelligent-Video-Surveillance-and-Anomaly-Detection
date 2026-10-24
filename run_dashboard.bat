@echo off
setlocal
echo =====================================================================
echo  🛡️ Launching Intelligent Video Surveillance & Anomaly Dashboard
echo =====================================================================

set "PYDIR=C:\Users\Anjali\AppData\Local\Programs\Python\Python311"
set "PATH=%PYDIR%;%PYDIR%\Scripts;%PATH%"
set PYTHONIOENCODING=utf-8

:: Ensure Python is accessible
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH!
    pause
    exit /b 1
)

echo [1/2] Checking Streamlit dependencies...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [!] Installing Streamlit and visualization libraries...
    python -m pip install streamlit plotly pandas opencv-python pillow torch --extra-index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
)

echo [2/2] Starting Streamlit Real-Time Operations Dashboard...
echo [✓] Opening web interface in your browser: http://localhost:8501
python -m streamlit run dashboard/app.py

pause
