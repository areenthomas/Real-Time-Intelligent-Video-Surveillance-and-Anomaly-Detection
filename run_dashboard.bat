@echo off
setlocal
echo =====================================================================
echo  🛡️ Launching Intelligent Video Surveillance & Anomaly Dashboard
echo =====================================================================

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
    pip install streamlit plotly pandas opencv-python pillow
)

echo [2/2] Starting Streamlit Real-Time Operations Dashboard...
echo [✓] Opening web interface in your browser: http://localhost:8501
streamlit run dashboard/app.py

pause

