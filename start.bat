@echo off
REM Enhanced Options Trading Platform - Windows Startup Script
REM Ali's Lenovo Server Configuration

echo ========================================
echo Enhanced Options Trading Platform
echo Ali's Configuration - Account U4312675
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "main.py" (
    echo ERROR: main.py not found
    echo Please run this script from the trading platform directory
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist "C:\Users\Lenovo\Desktop\Trading_bot2" mkdir "C:\Users\Lenovo\Desktop\Trading_bot2"
if not exist "C:\Users\Lenovo\Desktop\Trading_bot2\logs" mkdir "C:\Users\Lenovo\Desktop\Trading_bot2\logs"
if not exist "C:\Users\Lenovo\Desktop\Trading_bot2\data" mkdir "C:\Users\Lenovo\Desktop\Trading_bot2\data"

echo Checking Python packages...
pip install -r requirements.txt --quiet

echo.
echo Starting Enhanced Options Trading Platform...
echo.
echo IBKR Account: U4312675 (Live Trading)
echo IBKR Port: 4002
echo Telegram: @alialkhtateebtradingbot (938948925)
echo Server IP: 100.105.11.85:5000
echo.
echo Make sure IB Gateway is running on port 4002!
echo.

REM Start the platform
python main.py

REM If we get here, the platform has stopped
echo.
echo Platform has stopped.
pause

