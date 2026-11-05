@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" /min python server.py
timeout /t 3 /nobreak >nul
start "" chrome "http://localhost:3700"
