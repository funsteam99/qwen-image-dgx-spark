@echo off
call "%~dp0_load_config.bat" || exit /b 1
echo ===================================================
echo  DGX ComfyUI 產物清理（預演模式，不會刪除）
echo ===================================================
ssh %SSH_USER%@%DGX_HOST% "bash %REMOTE_DIR%/cleanup_dgx.sh %*"
echo.
echo 若確認無誤，實際刪除請執行：
echo    cleanup_dgx.bat --apply
pause
