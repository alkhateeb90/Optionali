#!/bin/bash
# Enhanced Options Trading Platform - Linux/Mac Startup Script
# Ali's Lenovo Server Configuration

echo "========================================"
echo "Enhanced Options Trading Platform"
echo "Ali's Configuration - Account U4312675"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found"
    echo "Please run this script from the trading platform directory"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "/home/ubuntu/Trading_bot2"
mkdir -p "/home/ubuntu/Trading_bot2/logs"
mkdir -p "/home/ubuntu/Trading_bot2/data"

echo "Checking Python packages..."
pip3 install -r requirements.txt --quiet

echo
echo "Starting Enhanced Options Trading Platform..."
echo
echo "IBKR Account: U4312675 (Live Trading)"
echo "IBKR Port: 4002"
echo "Telegram: @alialkhtateebtradingbot (938948925)"
echo "Server IP: 100.105.11.85:5000"
echo
echo "Make sure IB Gateway is running on port 4002!"
echo

# Start the platform
python3 main.py

# If we get here, the platform has stopped
echo
echo "Platform has stopped."
read -p "Press Enter to continue..."

