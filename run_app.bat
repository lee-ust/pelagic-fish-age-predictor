@echo off
title 🐟 Pelagic Fish Age Predictor
echo ========================================
echo   🐟 Pelagic Fish Age Predictor
echo   Machine Learning Web Application
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

:: Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Missing dependencies. Installing...
    pip install -r requirements.txt
    echo.
)

:: Check if model exists
if not exist "scripts\fish_age_model.pkl" (
    echo ⚠️ Warning: Model file not found
    echo Expected: scripts\fish_age_model.pkl
    echo.
)

:: Launch the app
echo 🚀 Launching web application...
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run scripts\front_end.py --server.port 8501

:: Keep window open if error
if errorlevel 1 (
    echo.
    echo ❌ Failed to launch application
    pause
)