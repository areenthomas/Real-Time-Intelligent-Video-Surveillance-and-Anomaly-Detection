# Git Setup & Push Automation for PowerShell
# Target Repository: https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Git Setup & Push Automation" -ForegroundColor Cyan
Write-Host " Repository: https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Check if Git is installed
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "[!] Git is not found in PATH." -ForegroundColor Yellow
    Write-Host "Attempting installation via Winget..." -ForegroundColor Yellow
    try {
        winget install --id Git.Git -e --source winget
        Write-Host "[✓] Git installed. Please restart your PowerShell session and rerun this script." -ForegroundColor Green
    } catch {
        Write-Host "[X] Winget installation failed. Please install Git manually from https://git-scm.com/" -ForegroundColor Red
    }
    exit 1
}

# Initialize git if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "[1/5] Initializing git repository..." -ForegroundColor Green
    git init
} else {
    Write-Host "[1/5] Git repository already initialized." -ForegroundColor Green
}

# Configure remote origin
Write-Host "[2/5] Setting remote origin..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin "https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git"

# Stage all files
Write-Host "[3/5] Staging files..." -ForegroundColor Green
git add .

# Commit files
Write-Host "[4/5] Committing changes..." -ForegroundColor Green
git commit -m "feat: complete research compendium, literature matrix, gap analysis, and UCF-Crime test framework"

# Push to GitHub
Write-Host "[5/5] Pushing to main branch..." -ForegroundColor Green
git branch -M main
git push -u origin main

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " Successfully pushed to GitHub!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

