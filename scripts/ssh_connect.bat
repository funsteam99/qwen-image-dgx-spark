@echo off
call "%~dp0_load_config.bat" || exit /b 1
echo Connecting to NVIDIA DGX Spark (%DGX_HOST%)...
ssh %SSH_USER%@%DGX_HOST%
pause
