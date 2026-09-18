@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python and add it to PATH.
  pause
  exit /b 1
)
echo [1/2] Installing Flask and NumPy into libs...
python -m pip install -r requirements.txt --target libs --upgrade
if errorlevel 1 goto failed
echo [2/2] Installing CPU PyTorch into libs...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --target libs --upgrade
if errorlevel 1 goto failed
echo.
echo Dependencies installed. Next run: start.bat
echo Before training: python download_data.py
pause
exit /b 0
:failed
echo.
echo [ERROR] Installation failed. Check the network, disk space and Python version.
pause
exit /b 1
