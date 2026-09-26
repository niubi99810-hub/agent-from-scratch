@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0."
set "PYTHONIOENCODING=utf-8"
set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m unittest discover -s tests -v