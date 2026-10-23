@echo off
setlocal
echo =====================================================================
echo  Git Setup & Push Automation
echo  Repository: https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git
echo =====================================================================

:: Ensure Git is in PATH
set "PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\Git\bin"

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git was not found in PATH or at C:\Program Files\Git\cmd!
    echo Installing Git via Winget...
    winget install --id Git.Git -e --source winget
    pause
    exit /b 1
)

echo [1/5] Checking Git repository...
if not exist .git (
    git init
    git config user.name "areenthomas"
    git config user.email "areenthomas@users.noreply.github.com"
)

echo [2/5] Configuring remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection.git

echo [3/5] Staging files...
git add .

echo [4/5] Committing changes...
git commit -m "feat: complete research compendium, literature matrix, gap analysis, and UCF-Crime test framework" 2>nul

echo [5/5] Pushing to GitHub (Sign in with your browser if prompted)...
git branch -M main
git push -u origin main

if %errorlevel% equ 0 (
    echo =====================================================================
    echo  [SUCCESS] Code successfully pushed to GitHub!
    echo  https://github.com/areenthomas/Real-Time-Intelligent-Video-Surveillance-and-Anomaly-Detection
    echo =====================================================================
) else (
    echo =====================================================================
    echo  [NOTE] If GitHub asked for authentication:
    echo  Ensure you have push rights or sign in via the browser popup.
    echo =====================================================================
)
pause
