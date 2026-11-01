@echo off
REM 把本機改好的 gradio_app.py 推到 Spark 上的 qwen-image 資料夾，再重啟 qwen-gradio-ui 容器。
REM 只重啟 Gradio，不動 qwen-image-comfyui（模型服務不需要跟著重啟）。
call "%~dp0_load_config.bat" || exit /b 1

echo ===================================================
echo  部署 gradio_app.py → %SSH_USER%@%DGX_HOST%:%REMOTE_DIR%
echo ===================================================

scp "%~dp0..\app\gradio_app.py" %SSH_USER%@%DGX_HOST%:%REMOTE_DIR%/gradio_app.py
if errorlevel 1 (
    echo [ERROR] scp 失敗，中止。
    pause
    exit /b 1
)

echo 重啟 qwen-gradio-ui...
ssh %SSH_USER%@%DGX_HOST% "docker restart qwen-gradio-ui"
if errorlevel 1 (
    echo [ERROR] 重啟容器失敗，請手動檢查。
    pause
    exit /b 1
)

echo.
echo 完成。幾秒後可到 http://%DGX_HOST%:7860 確認新版已生效。
echo （手機走 8443 要用 Tailscale 主機名稱，不能用 IP，見 DEPLOYMENT_NOTES.md 第 17 節。）
pause
