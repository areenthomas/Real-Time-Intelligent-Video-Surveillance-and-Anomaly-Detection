@echo off
echo =====================================================================
echo  Git Setup & Push Automation
echo  Repository: https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git
echo =====================================================================

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH!
    echo Installing Git via Winget...
    winget install --id Git.Git -e --source winget
    echo Please reopen your terminal after Git finishes installing and run this script again.
    pause
    exit /b 1
)

echo [1/5] Initializing Git repository...
if not exist .git (
    git init
)

echo [2/5] Configuring remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git

echo [3/5] Staging files...
git add .

echo [4/5] Creating commit...
git commit -m "feat: project research compendium, literature review, gap analysis, and UCF-Crime test framework"

echo [5/5] Pushing to main branch...
git branch -M main
git push -u origin main

echo =====================================================================
echo  Completed successfully!
echo =====================================================================
pause

