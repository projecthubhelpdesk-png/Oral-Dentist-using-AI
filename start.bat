@echo off
echo ========================================
echo    OralCare AI - Starting Services
echo ========================================
echo.

:: Start PHP Backend Server
echo [1/2] Starting PHP Backend Server on port 8080...
start "PHP Backend" cmd /k "cd /d %~dp0backend-php && php -S localhost:8080 -t ."

:: Wait a moment for backend to initialize
timeout /t 2 /nobreak > nul

:: Start Frontend Dev Server
echo [2/2] Starting Frontend Dev Server on port 5173...
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo    All services started!
echo ========================================
echo.
echo    Backend:  http://localhost:8080
echo    Frontend: http://localhost:5173
echo.
echo    To stop: Close the terminal windows
echo ========================================
