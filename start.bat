@echo off
cd /d "%~dp0"
rem 一键启动：先开浏览器，再启动训练服务器
start "" "http://127.0.0.1:5000"
python server.py
pause
