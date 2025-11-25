@echo off
echo ========================================
echo NBA Predictor Backend Startup Script
echo ========================================
echo.

cd /d %~dp0

REM Activate virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create it with: py -3.10 -m venv .venv
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if .env exists
if not exist .env (
    echo ERROR: .env file not found!
    echo Please create a .env file with the following variables:
    echo   BALLDONTLIE_API_KEY=your_api_key_here
    echo   BACKEND_HOST=localhost
    echo   BACKEND_PORT=8000
    echo.
    echo You can copy .env.example to .env and fill in your values.
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Starting backend server...
echo Server will be available at: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Health check: http://localhost:8000/health
echo ========================================
echo Press Ctrl+C to stop the server
echo.

REM Try port 8000 first, if it fails, use 8001
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload 2>nul || python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload

pause

