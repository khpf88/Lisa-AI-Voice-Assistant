@echo off
REM This script automates the installation of Python dependencies for the Lisa project.
REM
REM IMPORTANT: Before running this script, ensure the following are manually installed:
REM 1. Python 3.12 (https://www.python.org/downloads/release/python-3120/)
REM    - Make sure to check "Add Python to PATH" during installation.
REM 2. Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/)
REM    - Select "Desktop development with C++" workload.
REM 3. eSpeak NG (https://github.com/espeak-ng/espeak-ng/releases)


echo Installing Python dependencies for Lisa project...

REM Use py -3.12 to ensure the correct Python version is used
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64;%PATH%




py -3.12 -m pip uninstall -y torch torchvision torchaudio

rem Install the CPU-only version of PyTorch
py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



py -3.12 -m pip uninstall -y pywhispercpp


py -3.12 -m pip uninstall -y llama-cpp-python
set FORCE_CMAKE=1
py -3.12 -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
py -3.12 -m pip install fastapi uvicorn google-generativeai python-dotenv kokoro misaki[en] soundfile numpy vosk pyaudio
py -3.12 -m pip install num2words



if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install one or more Python dependencies.
    echo Please check the output above for details.
    echo Ensure Python 3.12, Visual Studio Build Tools, and eSpeak NG are installed correctly.
) else (
    echo.
    echo All Python dependencies installed successfully!
    echo You can now run the server using: py -3.12 main.py
)

pause