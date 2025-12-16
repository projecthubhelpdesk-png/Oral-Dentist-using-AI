@echo off
echo ========================================
echo    OralCare AI - Stopping Services
echo ========================================
echo.

:: Kill PHP processes
echo Stopping PHP servers...
taskkill /F /IM php.exe 2>nul

:: Kill Node processes (frontend)
echo Stopping Node servers...
taskkill /F /IM node.exe 2>nul

:: Kill Python processes (ML API)
echo Stopping Python servers...
taskkill /F /IM python.exe 2>nul

echo.
echo ========================================
echo    All services stopped!
echo ========================================
