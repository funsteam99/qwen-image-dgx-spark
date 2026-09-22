@echo off
call "%~dp0_load_config.bat" || exit /b 1
echo Opening ComfyUI Backend in browser...
start http://%DGX_HOST%:8188
