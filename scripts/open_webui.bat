@echo off
call "%~dp0_load_config.bat" || exit /b 1
echo Opening Qwen-Image-2.1 WebUI in browser...
start http://%DGX_HOST%:7860
