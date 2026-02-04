@echo off
echo Starting AI Health Advisor...

:: Start Backend in a new window
start "AI Health Advisor Backend" cmd /k "cd backend && python app.py"

:: Start Frontend in a new window
start "AI Health Advisor Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo   Servers are starting in new windows!
echo   Please wait a moment...
echo.
echo   Once ready, open: http://localhost:5173
echo ===================================================
echo.
pause
