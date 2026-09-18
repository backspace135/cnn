@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python and add it to PATH.
  pause
  exit /b 1
)
python -c "import paths, flask, numpy, torch" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Dependencies are missing. Run install.bat first.
  pause
  exit /b 1
)
if not exist "data\mnist.npz" (
  echo [ERROR] data\mnist.npz is missing. Run: python download_data.py
  echo The first three examples do not require MNIST data.
  pause
  exit /b 1
)
if "%PORT%"=="" set PORT=5000
if not "%CNN_NO_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%"
python server.py
if errorlevel 1 (
  echo [ERROR] Server failed to start. Check whether port %PORT% is in use.
)
pause
