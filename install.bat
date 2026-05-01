@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /D "%~dp0"
chcp 65001 > nul
title G3 EMBED Installer

set "VENV_DIR=%CD%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "FRONTEND_DIR=%CD%\frontend"

where node > nul 2>&1
if errorlevel 1 (
    echo FEHLER: Node.js wurde nicht gefunden. Bitte Node.js inklusive npm installieren.
    pause
    exit /b 1
)

where npm > nul 2>&1
if errorlevel 1 (
    echo FEHLER: npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren.
    pause
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo Erstelle virtuelle Umgebung in "%VENV_DIR%" ...
    py -3.10 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo FEHLER: Konnte keine lokale venv erstellen.
        pause
        exit /b 1
    )
)

echo Aktualisiere pip, setuptools und wheel ...
call "%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo FEHLER: pip-Update fehlgeschlagen.
    pause
    exit /b 1
)

echo Installiere passende PyTorch-Backend-Wheels ...
call "%VENV_PYTHON%" "%CD%\tools\install_torch_backend.py" --python "%VENV_PYTHON%"
if errorlevel 1 (
    echo FEHLER: PyTorch-Backend-Installation fehlgeschlagen.
    pause
    exit /b 1
)

echo Installiere Python-Abhaengigkeiten ...
call "%VENV_PIP%" install -r requirements.txt
if errorlevel 1 (
    echo FEHLER: Installation aus requirements.txt fehlgeschlagen.
    pause
    exit /b 1
)

echo Installiere Frontend-Abhaengigkeiten ...
pushd "%FRONTEND_DIR%"
call npm install
if errorlevel 1 (
    popd
    echo FEHLER: npm install fehlgeschlagen.
    pause
    exit /b 1
)

echo Baue Frontend ...
call npm run build
if errorlevel 1 (
    popd
    echo FEHLER: Frontend-Build fehlgeschlagen.
    pause
    exit /b 1
)
popd

echo.
echo Installation abgeschlossen. Starte danach start.bat.
pause
