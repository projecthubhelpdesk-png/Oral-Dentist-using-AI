# 🔧 Troubleshooting Guide - Oral Care AI

## Common Issues and Solutions

---

## 1. Prerequisites Not Found

### Node.js / npm not found
```
[ERROR] Node.js not found!
```
**Solution:**
1. Download from https://nodejs.org/ (LTS version)
2. Run installer with default options
3. **Restart your terminal/command prompt**
4. Verify: `node --version`

### Python not found
```
[ERROR] Python not found!
```
**Solution:**
1. Download from https://www.python.org/downloads/
2. **IMPORTANT:** Check ✅ "Add Python to PATH" during installation
3. Restart terminal
4. Verify: `python --version`

### PHP not found
```
[ERROR] PHP not found!
```
**Solution:**
1. Install XAMPP from https://www.apachefriends.org/
2. Add PHP to PATH:
   - Open System Properties → Environment Variables
   - Add `C:\xampp\php` to PATH
3. Or use PHP directly: `C:\xampp\php\php.exe`

### MySQL not found
```
[ERROR] MySQL not found!
```
**Solution:**
1. Make sure XAMPP is installed
2. Start MySQL from XAMPP Control Panel
3. Add to PATH: `C:\xampp\mysql\bin`

---

## 2. Database Issues

### Cannot connect to MySQL
```
ERROR 2002 (HY000): Can't connect to MySQL server
```
**Solution:**
1. Open XAMPP Control Panel
2. Click "Start" next to MySQL
3. Wait for it to turn green
4. Try again

### Database doesn't exist
```
ERROR 1049 (42000): Unknown database 'oral_care_ai'
```
**Solution:**
```bash
mysql -u root -e "CREATE DATABASE oral_care_ai;"
mysql -u root oral_care_ai < database/schema.sql
mysql -u root oral_care_ai < database/seed.sql
```

### Access denied for user 'root'
```
ERROR 1045 (28000): Access denied for user 'root'@'localhost'
```
**Solution:**
1. Check if MySQL has a password set
2. Update `backend-php/.env`:
   ```
   DB_PASS=your_mysql_password
   ```

---

## 3. Frontend Issues

### npm install fails
```
npm ERR! code ERESOLVE
```
**Solution:**
```bash
cd frontend
npm install --legacy-peer-deps
```

### Port 5173 already in use
```
Error: Port 5173 is already in use
```
**Solution:**
```bash
# Find and kill the process
netstat -ano | findstr :5173
taskkill /PID <PID_NUMBER> /F

# Or change port in vite.config.ts
```

### CORS errors in browser
```
Access to XMLHttpRequest blocked by CORS policy
```
**Solution:**
1. Make sure PHP backend is running on port 8080
2. Check `frontend/.env`:
   ```
   VITE_API_URL=http://localhost:8080/api
   ```

---

## 4. Backend PHP Issues

### PHP server won't start
```
Failed to listen on localhost:8080
```
**Solution:**
1. Check if port 8080 is in use:
   ```bash
   netstat -ano | findstr :8080
   ```
2. Kill the process or use different port:
   ```bash
   php -S localhost:8081 -t .
   ```

### Composer not found
**Solution:**
1. Download from https://getcomposer.org/download/
2. Run Composer-Setup.exe
3. Restart terminal
4. Verify: `composer --version`

### JWT errors / 401 Unauthorized
**Solution:**
1. Check `backend-php/.env` has valid JWT_SECRET (min 32 chars):
   ```
   JWT_SECRET=your-very-long-secret-key-at-least-32-characters
   ```
2. Clear browser localStorage and login again

---

## 5. ML Server Issues

### Python packages fail to install
```
ERROR: Could not install packages
```
**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements_enhanced.txt

# Or install core packages only
pip install tensorflow fastapi uvicorn opencv-python pillow numpy
```

### TensorFlow installation fails
**Solution:**
```bash
# For Windows, try:
pip install tensorflow-cpu

# Or specific version:
pip install tensorflow==2.13.0
```

### ML server won't start
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution:**
```bash
cd ml-model
pip install <missing_module>
```

### Port 8000 already in use
**Solution:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

---

## 6. General Issues

### "start-all.bat" doesn't work
**Solution:**
Run each service manually in separate terminals:

**Terminal 1 - PHP Backend:**
```bash
cd backend-php
php -S localhost:8080 -t .
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - ML Server:**
```bash
cd ml-model
python api_pytorch.py
```

### Login not working
**Solution:**
1. Make sure database is seeded:
   ```bash
   mysql -u root oral_care_ai < database/seed.sql
   ```
2. Use correct credentials:
   - Admin: `admin@oralcare.ai` / `password123`
   - Dentist: `dr.sarah.chen@dental.com` / `password123`

### Page shows blank or errors
**Solution:**
1. Open browser DevTools (F12) → Console tab
2. Check for errors
3. Make sure all 3 services are running
4. Clear browser cache: Ctrl+Shift+Delete

---

## 7. Quick Reset

If nothing works, try a complete reset:

```bash
# 1. Stop all services
stop.bat

# 2. Delete node_modules and reinstall
cd frontend
rmdir /s /q node_modules
npm install
cd ..

# 3. Reset database
mysql -u root -e "DROP DATABASE IF EXISTS oral_care_ai;"
mysql -u root -e "CREATE DATABASE oral_care_ai;"
mysql -u root oral_care_ai < database/schema.sql
mysql -u root oral_care_ai < database/seed.sql

# 4. Recreate .env files
del backend-php\.env
del frontend\.env
setup.bat

# 5. Start fresh
start-all.bat
```

---

## 8. Getting Help

If you're still stuck:

1. **Check the logs** in each terminal window
2. **Browser Console** (F12) for frontend errors
3. **Network tab** (F12) for API errors
4. Make sure all **3 services are running**:
   - Frontend: http://localhost:5173
   - Backend: http://localhost:8080
   - ML API: http://localhost:8000

---

## Service URLs

| Service | URL | Check |
|---------|-----|-------|
| Frontend | http://localhost:5173 | Should show login page |
| Backend | http://localhost:8080/api/health | Should return JSON |
| ML API | http://localhost:8000/docs | Should show FastAPI docs |
