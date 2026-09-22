@echo off
if not exist "%~dp0config.bat" (
    echo [ERROR] 找不到 %~dp0config.bat
    echo 請複製 config.example.bat 為 config.bat 並填入你的 DGX 連線資訊。
    pause
    exit /b 1
)
call "%~dp0config.bat"
