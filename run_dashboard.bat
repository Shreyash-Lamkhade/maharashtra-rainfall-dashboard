@echo off
echo ============================================================
echo  Maharashtra Rainfall DA Dashboard
echo ============================================================

REM -- Try to find Python in common locations --
set PYTHON=
for %%P in (
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python310\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
    "C:\ProgramData\Anaconda3\python.exe"
    "C:\Users\%USERNAME%\anaconda3\python.exe"
    "C:\Users\%USERNAME%\miniconda3\python.exe"
) do (
    if exist %%P (
        set PYTHON=%%~P
        goto :found
    )
)

REM -- Fallback: use whatever python is on PATH --
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" (
    echo ERROR: Python not found. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

:found
echo Using Python: %PYTHON%

REM -- Install dependencies --
echo Installing requirements...
"%PYTHON%" -m pip install -r requirements.txt --quiet

REM -- Launch dashboard --
echo Launching Streamlit dashboard...
"%PYTHON%" -m streamlit run app.py --server.port 8501 --server.headless false

pause
