@echo off
REM RTX 4060 Ti GPU Setup Script
REM Run this to install CUDA-enabled PyTorch

echo ============================================
echo RTX 4060 Ti Setup for Handwriting Synthesis
echo ============================================
echo.

echo [Step 1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment
    echo Make sure you're in the project directory
    pause
    exit /b 1
)
echo   ^> Virtual environment activated

echo.
echo [Step 2/4] Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio
echo   ^> Old PyTorch removed

echo.
echo [Step 3/4] Installing CUDA 12.1 PyTorch...
echo   This will download ~2GB, please wait...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo   ^> CUDA PyTorch installed

echo.
echo [Step 4/4] Testing GPU...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Run this to verify GPU setup:
echo   python test_gpu.py
echo.
echo Then start training with:
echo   python train.py --data_path iam_processed.json --batch_size 48 --use_gpu
echo.
echo ============================================
pause
