@echo off
rem One-time setup on Windows. Double-click this file.
cd /d "%~dp0"
py -3.12 -m venv .venv 2>nul || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python captions.py download
echo.
echo Setup done. Next: start.bat
pause
