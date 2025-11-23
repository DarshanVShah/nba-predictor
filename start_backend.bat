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
    echo WARNING: .env file not found!
    echo Creating .env file with default values...
    (
        echo BALLDONTLIE_API_KEY=bfdc4ecf-c070-4e93-b9ac-cb36f049efb1
        echo BACKEND_HOST=localhost
        echo BACKEND_PORT=8000
    ) > .env
    echo.
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

