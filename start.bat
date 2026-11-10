@echo off
rem Start captioning and open the caption screen + control page.
cd /d "%~dp0"
.venv\Scripts\python.exe captions.py --open %*
pause
