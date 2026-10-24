Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " 🛡️ Launching Intelligent Video Surveillance & Anomaly Dashboard" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan

$pyDir = "C:\Users\Anjali\AppData\Local\Programs\Python\Python311"
$scriptsDir = "$pyDir\Scripts"
$env:Path = "$pyDir;$scriptsDir;" + $env:Path
$env:PYTHONIOENCODING = "utf-8"

# Check Streamlit
try {
    & "$pyDir\python.exe" -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Installing Streamlit & dependencies..." -ForegroundColor Yellow
        & "$pyDir\python.exe" -m pip install streamlit plotly pandas opencv-python pillow torch --extra-index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
    }
} catch {
    Write-Host "[!] Checking Python environment..." -ForegroundColor Yellow
}

Write-Host "[✓] Starting Streamlit Real-Time Operations Dashboard at http://localhost:8501" -ForegroundColor Green
& "$pyDir\python.exe" -m streamlit run dashboard/app.py --server.headless=false
