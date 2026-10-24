Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " 🛡️ Launching Live Laptop Webcam Surveillance & Anomaly Tracker" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan

$pyDir = "C:\Users\Anjali\AppData\Local\Programs\Python\Python311"
$scriptsDir = "$pyDir\Scripts"
$env:Path = "$pyDir;$scriptsDir;" + $env:Path

Write-Host "[1/2] Connecting to Laptop Camera (Device 0)..." -ForegroundColor Yellow
Write-Host "[2/2] Initializing AI Anomaly Detection Engine..." -ForegroundColor Yellow
Write-Host "`n[INFO] A native surveillance video window will appear." -ForegroundColor Cyan
Write-Host "[INFO] Press 'q' inside the video window anytime to exit.`n" -ForegroundColor Cyan

& "$pyDir\python.exe" frontend_inference.py --source 0 --threshold 0.55

