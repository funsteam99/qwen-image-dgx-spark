@echo off
call "%~dp0_load_config.bat" || exit /b 1
echo ===================================================
echo Syncing local gradio_app.py to DGX Spark...
echo ===================================================
scp "%~dp0..\app\gradio_app.py" %SSH_USER%@%DGX_HOST%:%REMOTE_DIR%/gradio_app.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upload gradio_app.py!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Restarting qwen-gradio-ui container on DGX Spark...
ssh %SSH_USER%@%DGX_HOST% "docker restart qwen-gradio-ui"
echo.
echo [SUCCESS] Synced and restarted successfully!
pause
