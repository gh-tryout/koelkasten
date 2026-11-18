@echo off
cd /d "%~dp0"
echo Start Koelkastanalyse in debugmodus.
echo Log: %cd%\koelkastanalyse.log
echo.
set KOELKAST_DEBUG=1
"Koelkastanalyse.exe" --debug
echo.
echo Exitcode=%ERRORLEVEL%
pause
