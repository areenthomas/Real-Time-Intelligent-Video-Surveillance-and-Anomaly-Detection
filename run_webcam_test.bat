@echo off
setlocal
echo =====================================================================
echo  🛡️ Launching Live Laptop Webcam Surveillance & Anomaly Tracker
echo =====================================================================

set "PYDIR=C:\Users\Anjali\AppData\Local\Programs\Python\Python311"
set "PATH=%PYDIR%;%PYDIR%\Scripts;%PATH%"

echo [1/2] Connecting to Laptop Camera (Device 0)...
echo [2/2] Initializing AI Anomaly Detection Engine...
echo.
echo [INFO] A native surveillance video window will appear.
echo [INFO] Press 'q' inside the video window anytime to exit.
echo.

python frontend_inference.py --source 0 --threshold 0.55

pause

