@echo off
rem Transcribe system audio with Whisper. Stop with Ctrl+C.
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
set PYTHONUTF8=1
rem Keep the ~3GB model cache inside this folder instead of C:\Users\...\.cache
set "HF_HOME=%ROOT%models"
"%ROOT%venv\Scripts\python.exe" "%ROOT%transcribe.py" %*
echo.
pause
