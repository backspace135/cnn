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
chcp 65001 >nul
if not exist "data\mnist.npz" (
  echo [提示] 未找到 MNIST 数据 data\mnist.npz，仍会启动 Flask 课程页面。
  echo [提示] 训练与预测暂不可用；请运行 python download_data.py 下载数据。
)
if "%PORT%"=="" set PORT=5000
if not "%CNN_NO_BROWSER%"=="1" start "" "http://127.0.0.1:%PORT%"
python server.py
if errorlevel 1 (
  echo [ERROR] Server failed to start. Check whether port %PORT% is in use.
)
pause
