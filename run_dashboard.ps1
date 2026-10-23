Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " 🛡️ Launching Intelligent Video Surveillance & Anomaly Dashboard" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan

# Check Streamlit
try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Installing Streamlit..." -ForegroundColor Yellow
        pip install streamlit plotly pandas opencv-python pillow
    }
} catch {
    Write-Host "[!] Checking Python environment..." -ForegroundColor Yellow
}

Write-Host "[✓] Starting Streamlit Real-Time Operations Dashboard at http://localhost:8501" -ForegroundColor Green
streamlit run dashboard/app.py

