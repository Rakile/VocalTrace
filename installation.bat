@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ============================================================
REM  VocalTrace — one-shot installer (Windows)
REM  Double-click this file, or:  installation.bat
REM  Non-interactive:           installation.bat --yes
REM  Force torch:               installation.bat cuda|cpu [--yes]
REM ============================================================

set "PYTHON_EXE="
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "TORCH_VER=2.8.0"
set "TORCH_INDEX="
set "TORCH_LABEL="
set "TORCH_MODE=auto"
set "NONINTERACTIVE=0"
set "WANT_FLASH="
set "WANT_CLEARVOICE="
set "HAS_NVIDIA=0"
set "FLASH_WHEEL="

REM ---------- parse args ----------
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--yes" set "NONINTERACTIVE=1" & shift & goto :parse_args
if /i "%~1"=="-y" set "NONINTERACTIVE=1" & shift & goto :parse_args
if /i "%~1"=="/yes" set "NONINTERACTIVE=1" & shift & goto :parse_args
if /i "%~1"=="cuda" set "TORCH_MODE=cuda" & shift & goto :parse_args
if /i "%~1"=="cpu" set "TORCH_MODE=cpu" & shift & goto :parse_args
if /i "%~1"=="auto" set "TORCH_MODE=auto" & shift & goto :parse_args
echo [WARN] Unknown argument: %~1
shift
goto :parse_args
:args_done

echo.
echo ============================================================
echo   VocalTrace installer
echo ============================================================
echo.
echo  This will:
echo    1. Find Python 3.12.10+  (required)
echo    2. Create a local .venv
echo    3. Install PyTorch 2.8.0 + project dependencies
echo    4. Optionally install FlashAttention and/or ClearVoice
echo    5. Leave you ready to run:  run.bat
echo.
echo  Notes:
echo    - transformers is pinned to 4.57.x  (5.x breaks Whisper)
echo    - system ffmpeg must be on PATH  (not Conda ffmpeg)
echo    - API keys go in a .env file after install  (optional)
echo.

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "HAS_NVIDIA=1"
    echo  [detect] NVIDIA GPU tools found ^(nvidia-smi^).
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul
) else (
    echo  [detect] No nvidia-smi — will default to CPU PyTorch unless you choose CUDA.
)
echo.

REM ---------- Find Python 3.12.10+ ----------
call :try_python "py -3.12"
if defined PYTHON_EXE goto :have_python
call :try_python "py -3.12-64"
if defined PYTHON_EXE goto :have_python
call :try_python "python3.12"
if defined PYTHON_EXE goto :have_python
call :try_python "python"

:have_python
if not defined PYTHON_EXE (
    echo [ERROR] No suitable Python found.
    echo.
    echo VocalTrace needs Python 3.12.10 or a later 3.12.x patch.
    echo Early 3.12.0-3.12.9 hit: AttributeError '_SixMetaPathImporter' ... '_path'
    echo.
    echo Install from: https://www.python.org/downloads/release/python-31210/
    echo Then re-run this script.  Tip:  py -0p
    echo.
    pause
    exit /b 1
)

echo Using host Python:
"%PYTHON_EXE%" -c "import sys; print(' ', sys.executable); print(' ', sys.version)"
echo.

REM ---------- Interactive choices ----------
if "%NONINTERACTIVE%"=="1" goto :choices_auto

echo ------------------------------------------------------------
echo  Optional components
echo ------------------------------------------------------------
echo.

if /i "%TORCH_MODE%"=="auto" (
    if "%HAS_NVIDIA%"=="1" (
        set "TORCH_MODE=cuda"
        echo  GPU: will install CUDA 12.8 PyTorch ^(recommended for NVIDIA^).
        echo  Press Enter to keep CUDA, or type CPU then Enter for CPU-only.
        set /p "TORCH_OVERRIDE=  [CUDA] / CPU : "
        if /i "!TORCH_OVERRIDE!"=="cpu" set "TORCH_MODE=cpu"
        if /i "!TORCH_OVERRIDE!"=="CPU" set "TORCH_MODE=cpu"
    ) else (
        echo  No NVIDIA detected. Install CPU PyTorch?  [Y]es / [C]UDA anyway
        set /p "TORCH_OVERRIDE=  Choice [Y]: "
        if /i "!TORCH_OVERRIDE!"=="c" set "TORCH_MODE=cuda"
        if /i "!TORCH_OVERRIDE!"=="C" set "TORCH_MODE=cuda"
        if /i "!TORCH_OVERRIDE!"=="cuda" set "TORCH_MODE=cuda"
        if not defined TORCH_MODE set "TORCH_MODE=cpu"
        if /i "!TORCH_MODE!"=="auto" set "TORCH_MODE=cpu"
    )
)
echo.

REM FlashAttention — only meaningful on CUDA + matching prebuilt wheel
call :find_flash_wheel
if /i "%TORCH_MODE%"=="cuda" (
    if defined FLASH_WHEEL (
        echo  FlashAttention2: faster Whisper on NVIDIA GPU. Optional.
        echo  Wheel found: !FLASH_WHEEL!
        echo  Install FlashAttention?  [Y]es / [N]o
        set /p "WANT_FLASH=  Choice [Y]: "
        if "!WANT_FLASH!"=="" set "WANT_FLASH=Y"
    ) else (
        echo  FlashAttention2: no matching .whl in this folder ^(optional speedup^).
        echo  Place flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl here
        echo  and re-run if you want it. Skipping for now.
        set "WANT_FLASH=N"
    )
) else (
    echo  FlashAttention: skipped ^(CPU install / not useful without CUDA^).
    set "WANT_FLASH=N"
)
echo.

echo  ClearVoice: optional AI speech enhancement ^(extra download, can conflict
echo  with pins — installer re-applies requirements after^). Not required.
echo  Install ClearVoice?  [Y]es / [N]o
set /p "WANT_CLEARVOICE=  Choice [N]: "
if "!WANT_CLEARVOICE!"=="" set "WANT_CLEARVOICE=N"
echo.
goto :choices_done

:choices_auto
if /i "%TORCH_MODE%"=="auto" (
    if "%HAS_NVIDIA%"=="1" (set "TORCH_MODE=cuda") else (set "TORCH_MODE=cpu")
)
call :find_flash_wheel
if /i "%TORCH_MODE%"=="cuda" if defined FLASH_WHEEL (set "WANT_FLASH=Y") else (set "WANT_FLASH=N")
if not defined WANT_FLASH set "WANT_FLASH=N"
set "WANT_CLEARVOICE=N"
echo  Non-interactive: torch=%TORCH_MODE%  flash=%WANT_FLASH%  clearvoice=%WANT_CLEARVOICE%
echo.

:choices_done

REM ---------- Create / refresh venv ----------
set "NEED_VENV=0"
if not exist "%VENV_PY%" (
    set "NEED_VENV=1"
) else (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if (sys.version_info.major==3 and sys.version_info.minor==12 and sys.version_info.micro>=10) else 1)" >nul 2>&1
    if errorlevel 1 set "NEED_VENV=1"
)

if "%NEED_VENV%"=="1" (
    if exist "%VENV_DIR%" (
        echo Existing venv is missing or uses wrong Python — recreating...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Creating virtual environment in %VENV_DIR% ...
    )
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo Reusing existing venv: %VENV_DIR%
)

echo Venv Python:
"%VENV_PY%" -c "import sys; print(' ', sys.executable); print(' ', sys.version)"
echo.

REM ---------- Torch ----------
if /i "%TORCH_MODE%"=="cuda" (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    set "TORCH_LABEL=CUDA 12.8 (cu128) — torch==%TORCH_VER%"
) else (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "TORCH_LABEL=CPU — torch==%TORCH_VER%"
)

echo [1/5] Installing PyTorch: %TORCH_LABEL%
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :pip_fail
"%VENV_PY%" -m pip install --upgrade "torch==%TORCH_VER%" "torchaudio==%TORCH_VER%" --index-url "%TORCH_INDEX%"
if errorlevel 1 (
    echo [ERROR] PyTorch install failed.
    pause
    exit /b 1
)

echo.
echo [2/5] Installing VocalTrace requirements...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail
"%VENV_PY%" -m pip install "transformers>=4.57.3,<5.0.0" "pyannote.audio==4.0.3"
if errorlevel 1 goto :pip_fail

REM ---------- FlashAttention ----------
echo.
echo [3/5] Optional FlashAttention...
if /i "%WANT_FLASH%"=="Y" goto :do_flash
if /i "%WANT_FLASH%"=="y" goto :do_flash
if /i "%WANT_FLASH%"=="yes" goto :do_flash
echo   Skipped.
goto :after_flash

:do_flash
if not defined FLASH_WHEEL (
    echo   [WARN] No wheel found — skip.
    goto :after_flash
)
echo   Installing with --no-deps ^(keeps torch %TORCH_VER%^)...
"%VENV_PY%" -m pip install --no-deps "%FLASH_WHEEL%"
if errorlevel 1 (
    echo   [WARN] flash-attn install failed — app still works without it.
) else (
    "%VENV_PY%" -c "import flash_attn; print('  [OK] flash_attn', flash_attn.__version__)" 2>nul
    if errorlevel 1 echo   [WARN] flash_attn import failed — wheel may not match this system.
)
:after_flash

REM ---------- ClearVoice ----------
echo.
echo [4/5] Optional ClearVoice...
if /i "%WANT_CLEARVOICE%"=="Y" goto :do_clearvoice
if /i "%WANT_CLEARVOICE%"=="y" goto :do_clearvoice
if /i "%WANT_CLEARVOICE%"=="yes" goto :do_clearvoice
echo   Skipped.
goto :after_clearvoice

:do_clearvoice
echo   Installing clearvoice ^(may take a while^)...
"%VENV_PY%" -m pip install clearvoice
if errorlevel 1 (
    echo   [WARN] clearvoice install failed — continuing.
) else (
    echo   Re-applying requirements.txt to restore pinned versions...
    "%VENV_PY%" -m pip install -r requirements.txt
    "%VENV_PY%" -m pip install "transformers>=4.57.3,<5.0.0" "pyannote.audio==4.0.3"
    echo   [OK] ClearVoice installed; pins restored.
)
:after_clearvoice

REM ---------- Cleanup torchcodec ----------
echo.
echo [5/5] Cleanup ^(remove broken torchcodec if present^)...
"%VENV_PY%" -m pip uninstall -y torchcodec 2>nul
if errorlevel 1 (echo   torchcodec not present — OK.) else (echo   [OK] torchcodec removed.)

REM ---------- ffmpeg ----------
echo.
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg not found on PATH.
    echo        Install system ffmpeg and add it to PATH, or set VOCALTRACE_FFMPEG.
    echo        Do NOT use Conda ffmpeg on Windows ^(breaks PySide6^).
) else (
    echo [OK] ffmpeg found on PATH.
)

REM ---------- .env template ----------
if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo [OK] Created .env from .env.example — add API keys when ready.
    )
)

REM ---------- write run.bat ----------
(
echo @echo off
echo cd /d "%%~dp0"
echo if not exist ".venv\Scripts\python.exe" ^(
echo   echo Virtual environment missing. Run installation.bat first.
echo   pause
echo   exit /b 1
echo ^)
echo ".venv\Scripts\python.exe" launch.py
echo if errorlevel 1 pause
) > run.bat

echo.
echo ============================================================
echo   Install complete
echo ============================================================
echo.
"%VENV_PY%" -c "import torch; print('  torch         ', torch.__version__, '| cuda:', torch.cuda.is_available()); import transformers; print('  transformers  ', transformers.__version__);
try:
 import flash_attn; print('  flash_attn    ', flash_attn.__version__)
except Exception:
 print('  flash_attn     not installed (optional)')
try:
 import clearvoice; print('  clearvoice     installed')
except Exception:
 print('  clearvoice     not installed (optional)')
"
echo.
echo  Start VocalTrace:
echo      run.bat
echo  or:
echo      .venv\Scripts\python.exe launch.py
echo.
echo  Recommended first run in the app:
echo      Diarization: pyannote/speaker-diarization-3.1
echo      Transcription: auto  or  KBLab/kb-whisper-large
echo.
echo  API keys ^(optional^): edit .env in this folder
echo    HF_TOKEN_TRANSCRIBE=...
echo    OPENAI_API_KEY=...
echo    GEMINI_TRANSCRIBE_ANALYSIS_API_KEY=...
echo.
echo  Re-run installer:  installation.bat
echo  Quiet defaults:    installation.bat --yes
echo ============================================================
echo.

if "%NONINTERACTIVE%"=="1" exit /b 0

set /p "LAUNCH_NOW=Launch VocalTrace now? [Y/n]: "
if /i "%LAUNCH_NOW%"=="n" goto :end_ok
if /i "%LAUNCH_NOW%"=="N" goto :end_ok
if /i "%LAUNCH_NOW%"=="no" goto :end_ok
echo Starting...
"%VENV_PY%" launch.py
if errorlevel 1 pause

:end_ok
exit /b 0

:pip_fail
echo [ERROR] pip install failed.
pause
exit /b 1

REM ---------- helpers ----------
:find_flash_wheel
set "FLASH_WHEEL="
if exist "%~dp0flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl" (
    set "FLASH_WHEEL=%~dp0flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl"
    goto :eof
)
if exist "%~dp0flash_attn*.whl" (
    for %%F in ("%~dp0flash_attn*.whl") do (
        set "FLASH_WHEEL=%%~fF"
        goto :eof
    )
)
if exist "D:\tools\VocalTrace_\flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl" (
    set "FLASH_WHEEL=D:\tools\VocalTrace_\flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl"
)
goto :eof

:try_python
set "CAND=%~1"
if defined PYTHON_EXE goto :eof
for /f "delims=" %%P in ('%CAND% -c "import sys; print(sys.executable)" 2^>nul') do (
    set "CAND_EXE=%%P"
)
if not defined CAND_EXE goto :eof
"%CAND_EXE%" -c "import sys; v=sys.version_info; raise SystemExit(0 if (v.major==3 and v.minor==12 and v.micro>=10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [skip] %CAND% is not Python 3.12.10+:
    "%CAND_EXE%" -c "import sys; print('      ', sys.version)" 2>nul
    set "CAND_EXE="
    goto :eof
)
set "PYTHON_EXE=%CAND_EXE%"
set "CAND_EXE="
goto :eof
