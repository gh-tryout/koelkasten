@echo off
setlocal
cd /d "%~dp0"
echo.
echo Koelkastanalyse — diagnose
echo Map: %cd%
echo.

set LOG=koelkastanalyse.log
echo ===== diagnose %date% %time% =====>> "%LOG%"
echo map=%cd%>> "%LOG%"
echo computer=%COMPUTERNAME% gebruiker=%USERNAME%>> "%LOG%"
ver >> "%LOG%" 2>&1

if not exist "Koelkastanalyse.exe" (
    echo FOUT: Koelkastanalyse.exe ontbreekt in deze map.
    echo FOUT: Koelkastanalyse.exe ontbreekt>> "%LOG%"
    goto :end
)
if not exist "_internal" (
    echo FOUT: map _internal ontbreekt. Kopieer de hele map, niet alleen de exe.
    echo FOUT: _internal ontbreekt>> "%LOG%"
    goto :end
)

echo Controle-bestanden:
dir /b "Koelkastanalyse.exe" "_internal" >> "%LOG%" 2>&1
dir /b "Koelkastanalyse.exe"
echo.
echo Start zelftest. Het venster blijft open.
echo Output gaat ook naar %LOG%
echo.
set KOELKAST_DEBUG=1
"Koelkastanalyse.exe" --selftest
echo.
echo Exitcode=%ERRORLEVEL%
echo Exitcode=%ERRORLEVEL%>> "%LOG%"

:end
echo.
echo Klaar. Stuur %LOG% mee als het thuis wel en op het werk niet start.
echo.
pause
endlocal
