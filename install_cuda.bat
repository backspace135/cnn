@echo off
cd /d "%~dp0"
echo Installing PyTorch with CUDA 12.8 support into libs-cuda...
echo Close the training server before upgrading an existing installation.
python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128 --target libs-cuda --upgrade
if errorlevel 1 (
    echo Installation failed. Check your connection and Python version.
    pause
    exit /b 1
)
python -c "import paths, torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available - check NVIDIA driver')"
echo Restart start.bat, then choose Auto or CUDA in the experiment settings.
pause
