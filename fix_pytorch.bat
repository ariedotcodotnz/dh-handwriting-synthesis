@echo off
REM Automatic PyTorch CUDA Installation Fix Script
REM Detects Python version and installs appropriate PyTorch

echo ============================================================
echo PyTorch CUDA Installation Fix Script
echo ============================================================
echo.

REM Check Python version
echo [1/4] Checking Python version...
python --version
python --version > temp_version.txt 2>&1

REM Extract version info
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   Detected: Python %PYTHON_VERSION%

REM Check if Python 3.11 is available
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set HAS_PY311=1
    echo   ✓ Python 3.11 is available
) else (
    set HAS_PY311=0
    echo   ⚠ Python 3.11 not found
)

echo.
echo [2/4] Analyzing compatibility...

REM Check Python version number
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

echo   Python %PY_MAJOR%.%PY_MINOR% detected

if "%PY_MINOR%"=="8" goto compatible
if "%PY_MINOR%"=="9" goto compatible
if "%PY_MINOR%"=="10" goto compatible
if "%PY_MINOR%"=="11" goto compatible
if "%PY_MINOR%"=="12" goto python312
goto incompatible

:compatible
echo   ✓ Python version compatible with PyTorch CUDA
echo.
echo [3/4] Installing PyTorch with CUDA 12.1...
goto install_pytorch

:python312
echo   ⚠ Python 3.12 detected
echo   PyTorch CUDA 12.1 may not be available via index URL
echo.
echo   Options:
echo   1. Try installing anyway (may work)
echo   2. Create Python 3.11 environment (recommended)
echo   3. Exit and install Python 3.11 manually
echo.
choice /C 123 /M "Select option"
if errorlevel 3 goto manual_install
if errorlevel 2 goto create_py311
if errorlevel 1 goto try_install_312
goto end

:incompatible
echo   ❌ Python %PY_MAJOR%.%PY_MINOR% is not compatible
echo.
echo   PyTorch requires Python 3.8-3.12
echo   Recommended: Python 3.11
echo.
if %HAS_PY311%==1 (
    echo   Good news: Python 3.11 is installed on your system!
    echo   Creating new virtual environment with Python 3.11...
    goto create_py311
) else (
    goto manual_install
)

:create_py311
if %HAS_PY311%==0 (
    echo   ❌ Python 3.11 is not installed
    goto manual_install
)
echo.
echo [3/4] Creating virtual environment with Python 3.11...
py -3.11 -m venv .venv311
if errorlevel 1 (
    echo   ❌ Failed to create virtual environment
    pause
    exit /b 1
)
echo   ✓ Virtual environment created: .venv311
echo.
echo   Activating Python 3.11 environment...
call .venv311\Scripts\activate.bat
echo   ✓ Python 3.11 environment activated
echo.
python --version
goto install_pytorch

:try_install_312
echo.
echo [3/4] Attempting to install PyTorch for Python 3.12...
call .venv\Scripts\activate.bat
goto install_pytorch

:install_pytorch
echo.
echo   Updating pip...
python -m pip install --upgrade pip >nul 2>&1
echo   ✓ Pip updated

echo.
echo   Installing PyTorch CUDA 12.1...
echo   (This may take 2-5 minutes and download ~2GB)
echo.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo   ❌ Installation failed with CUDA 12.1
    echo   Trying CUDA 11.8 instead...
    echo.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo   ❌ Also failed with CUDA 11.8
        echo   Trying CPU version as fallback...
        pip install torch torchvision torchaudio
    )
)

echo.
echo [4/4] Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>nul

if errorlevel 1 (
    echo   ❌ PyTorch not installed correctly
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.

python -c "import torch; cuda_available = torch.cuda.is_available(); print('✓ PyTorch installed successfully'); print('CUDA support:', 'YES' if cuda_available else 'NO - CPU only')"

echo.
if exist .venv311 (
    echo NOTE: You created a new Python 3.11 environment: .venv311
    echo.
    echo To use it in the future:
    echo   .venv311\Scripts\activate
    echo.
)

echo Next steps:
echo   1. Test GPU: python test_gpu.py
echo   2. Train: python train.py --data_path iam_processed.json --batch_size 48 --use_gpu
echo.
echo ============================================================
pause
goto end

:manual_install
echo.
echo ============================================================
echo Manual Installation Required
echo ============================================================
echo.
echo Python 3.11 is not installed on your system.
echo.
echo Please follow these steps:
echo.
echo 1. Download Python 3.11:
echo    https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe
echo.
echo 2. Run the installer:
echo    - Check "Add Python 3.11 to PATH"
echo    - Click "Install Now"
echo.
echo 3. After installation, run this script again
echo.
echo ============================================================
pause
goto end

:end
if exist temp_version.txt del temp_version.txt
