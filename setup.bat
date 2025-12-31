@echo off
echo ========================================
echo    OralCare AI - Setup Script
echo ========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as administrator. Some operations may fail.
    echo.
)

:: Check prerequisites
echo [STEP 1/7] Checking prerequisites...
echo.

:: Check Node.js
where node >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Node.js not found! 
    echo         Download from: https://nodejs.org/
    echo.
    set MISSING_PREREQ=1
) else (
    for /f "tokens=*" %%i in ('node --version') do echo [OK] Node.js: %%i
)

:: Check npm
where npm >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] npm not found!
    set MISSING_PREREQ=1
) else (
    for /f "tokens=*" %%i in ('npm --version') do echo [OK] npm: %%i
)

:: Check Python
where python >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found!
    echo         Download from: https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH"
    set MISSING_PREREQ=1
) else (
    for /f "tokens=*" %%i in ('python --version') do echo [OK] %%i
)

:: Check pip
where pip >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip not found!
    set MISSING_PREREQ=1
) else (
    for /f "tokens=*" %%i in ('pip --version') do echo [OK] pip found
)

:: Check PHP
where php >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] PHP not in PATH. Checking XAMPP...
    if exist "C:\xampp\php\php.exe" (
        echo [OK] PHP found in XAMPP: C:\xampp\php\php.exe
        set "PATH=%PATH%;C:\xampp\php"
    ) else (
        echo [ERROR] PHP not found! Install XAMPP from https://www.apachefriends.org/
        set MISSING_PREREQ=1
    )
) else (
    for /f "tokens=*" %%i in ('php --version 2^>^&1') do (
        echo [OK] %%i
        goto :php_done
    )
)
:php_done

:: Check MySQL
where mysql >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] MySQL not in PATH. Checking XAMPP...
    if exist "C:\xampp\mysql\bin\mysql.exe" (
        echo [OK] MySQL found in XAMPP
        set "PATH=%PATH%;C:\xampp\mysql\bin"
    ) else (
        echo [ERROR] MySQL not found! Make sure XAMPP is installed and MySQL is running.
        set MISSING_PREREQ=1
    )
) else (
    echo [OK] MySQL found
)

:: Check Composer
where composer >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Composer not found. Will try to install PHP dependencies anyway.
) else (
    for /f "tokens=*" %%i in ('composer --version') do echo [OK] %%i
)

echo.
if defined MISSING_PREREQ (
    echo [ERROR] Some prerequisites are missing. Please install them first.
    echo.
    pause
    exit /b 1
)

echo ========================================
echo [STEP 2/7] Setting up environment files...
echo ========================================
echo.

:: Create backend .env if not exists
if not exist "backend-php\.env" (
    if exist "backend-php\.env.example" (
        copy "backend-php\.env.example" "backend-php\.env"
        echo [OK] Created backend-php\.env from example
    ) else (
        echo [CREATING] backend-php\.env
        (
            echo DB_HOST=localhost
            echo DB_NAME=oral_care_ai
            echo DB_USER=root
            echo DB_PASS=
            echo JWT_SECRET=oral-care-ai-jwt-secret-key-32chars
            echo ENCRYPTION_KEY=oral-care-ai-encryption-key-32ch
            echo USE_REAL_ML=true
            echo ML_API_URL=http://localhost:8000
        ) > "backend-php\.env"
        echo [OK] Created backend-php\.env
    )
) else (
    echo [OK] backend-php\.env already exists
)

:: Create frontend .env if not exists
if not exist "frontend\.env" (
    if exist "frontend\.env.example" (
        copy "frontend\.env.example" "frontend\.env"
        echo [OK] Created frontend\.env from example
    ) else (
        echo [CREATING] frontend\.env
        echo VITE_API_URL=http://localhost:8080/api > "frontend\.env"
        echo [OK] Created frontend\.env
    )
) else (
    echo [OK] frontend\.env already exists
)

echo.
echo ========================================
echo [STEP 3/7] Setting up Database...
echo ========================================
echo.
echo [INFO] Make sure MySQL is running in XAMPP!
echo.

:: Try to create database and import schema
echo [RUNNING] Creating database and importing schema...
mysql -u root -e "CREATE DATABASE IF NOT EXISTS oral_care_ai;" 2>nul
if %errorLevel% neq 0 (
    echo [WARNING] Could not connect to MySQL. Make sure:
    echo          1. XAMPP is running
    echo          2. MySQL service is started
    echo          3. Try running: mysql -u root -e "CREATE DATABASE IF NOT EXISTS oral_care_ai;"
    echo.
) else (
    echo [OK] Database created/verified
    
    mysql -u root oral_care_ai < database\schema.sql 2>nul
    if %errorLevel% neq 0 (
        echo [WARNING] Could not import schema. You may need to run manually:
        echo          mysql -u root oral_care_ai ^< database\schema.sql
    ) else (
        echo [OK] Schema imported
    )
    
    mysql -u root oral_care_ai < database\seed.sql 2>nul
    if %errorLevel% neq 0 (
        echo [WARNING] Could not import seed data. You may need to run manually:
        echo          mysql -u root oral_care_ai ^< database\seed.sql
    ) else (
        echo [OK] Seed data imported
    )
)

echo.
echo ========================================
echo [STEP 4/7] Installing PHP dependencies...
echo ========================================
echo.
cd backend-php
where composer >nul 2>&1
if %errorLevel% equ 0 (
    composer install --no-interaction
    echo [OK] PHP dependencies installed
) else (
    echo [SKIP] Composer not found, skipping PHP dependencies
    echo        Install Composer from: https://getcomposer.org/
)
cd ..

echo.
echo ========================================
echo [STEP 5/7] Installing Frontend dependencies...
echo ========================================
echo.
cd frontend
call npm install
if %errorLevel% neq 0 (
    echo [ERROR] Failed to install frontend dependencies
    echo         Try running: cd frontend ^&^& npm install
) else (
    echo [OK] Frontend dependencies installed
)
cd ..

echo.
echo ========================================
echo [STEP 6/7] Installing Python ML dependencies...
echo ========================================
echo.
cd ml-model
pip install -r requirements_enhanced.txt
if %errorLevel% neq 0 (
    echo [WARNING] Some Python packages may have failed to install
    echo          Try running: cd ml-model ^&^& pip install -r requirements_enhanced.txt
) else (
    echo [OK] Python dependencies installed
)
cd ..

echo.
echo ========================================
echo [STEP 7/7] Creating storage directories...
echo ========================================
echo.
if not exist "backend-php\storage\scans" mkdir "backend-php\storage\scans"
if not exist "backend-php\storage\spectral" mkdir "backend-php\storage\spectral"
if not exist "storage\scans" mkdir "storage\scans"
echo [OK] Storage directories created

echo.
echo ========================================
echo    SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo   1. Make sure XAMPP Apache and MySQL are running
echo   2. Run: start-all.bat
echo.
echo URLs:
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8080
echo   ML API:   http://localhost:8000
echo.
echo Demo Accounts:
echo   Admin:   admin@oralcare.ai / password123
echo   Dentist: dr.sarah.chen@dental.com / password123
echo   Patient: john.doe@example.com / password123
echo.
echo ========================================
pause
