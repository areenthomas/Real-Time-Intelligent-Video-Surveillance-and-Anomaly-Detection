@echo off
setlocal
echo =====================================================================
echo  🛡️ Testing UCF-Crime Normal Surveillance Video: Normal_Videos_872_x264.mp4
echo =====================================================================

set "PYDIR=C:\Users\Anjali\AppData\Local\Programs\Python\Python311"
set "PATH=%PYDIR%;%PYDIR%\Scripts;%PATH%"
set PYTHONIOENCODING=utf-8

python frontend_inference.py --source data/test_videos/Normal_Videos_872_x264.mp4 --threshold 0.55 --scenario normal

pause

