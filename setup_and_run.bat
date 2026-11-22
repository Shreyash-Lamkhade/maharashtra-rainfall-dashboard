@echo off
setlocal

set PYTHON=C:\Users\Shreyash\AppData\Local\Programs\Python\Python311\python.exe

echo ============================================================
echo  Step 1: Installing required packages
echo ============================================================
"%PYTHON%" -m pip install streamlit pandas numpy plotly scikit-learn matplotlib mlxtend
if %errorlevel% neq 0 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 2: Verifying packages
echo ============================================================
"%PYTHON%" -c "import streamlit, pandas, numpy, plotly, sklearn, matplotlib, mlxtend; print('All packages installed OK!')"
if %errorlevel% neq 0 (
    echo ERROR: Package verification failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 3: Launching dashboard at http://localhost:8501
echo ============================================================
cd /d "%~dp0"
"%PYTHON%" -m streamlit run app.py --server.port 8501

pause
