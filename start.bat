@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =================================================================
REM == Startskript fuer G3_EMBED
REM == richtet bei Bedarf lokale virtuelle Umgebung .\venv selbst ein
REM =================================================================

cd /D "%~dp0"
chcp 65001 > nul
title G3 EMBED Server Launcher

set "VENV_DIR=%CD%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "REQ_STAMP=%VENV_DIR%\.requirements_installed"
set "FRONTEND_DIR=%CD%\frontend"
set "FRONTEND_NODE_MODULES=%FRONTEND_DIR%\node_modules"
set "FRONTEND_NPM_STAMP=%FRONTEND_DIR%\.node_modules_installed"
set "NEEDS_SETUP=0"
set "NEEDS_PY_DEPS=0"
set "NEEDS_FRONTEND_DEPS=0"

if not exist "%VENV_PYTHON%" (
    set "NEEDS_SETUP=1"
    set "NEEDS_PY_DEPS=1"
)

if "%NEEDS_SETUP%"=="1" (
    echo Lokale venv nicht gefunden. Setup wird gestartet...
    echo Erstelle virtuelle Umgebung in "%VENV_DIR%" ...
    py -3.10 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo FEHLER: Konnte keine lokale venv erstellen.
        pause
        exit /b 1
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
)

if exist "%VENV_PYTHON%" if "%NEEDS_PY_DEPS%"=="0" (
    call "%VENV_PYTHON%" -c "import os, sys; req='requirements.txt'; stamp=r'%REQ_STAMP%'; sys.exit(0 if os.path.exists(stamp) and os.path.getmtime(stamp) >= os.path.getmtime(req) else 1)"
    if errorlevel 1 set "NEEDS_PY_DEPS=1"
)

if "%NEEDS_PY_DEPS%"=="1" (
    echo Installiere/Aktualisiere Python-Abhaengigkeiten ...
    call "%VENV_PIP%" install -r requirements.txt
    if errorlevel 1 (
        echo FEHLER: Installation aus requirements.txt fehlgeschlagen.
        pause
        exit /b 1
    )
    call "%VENV_PYTHON%" -c "from pathlib import Path; Path(r'%REQ_STAMP%').write_text('ok', encoding='utf-8')"
)

if not exist "%FRONTEND_NODE_MODULES%" (
    set "NEEDS_FRONTEND_DEPS=1"
) else (
    powershell -NoProfile -Command "$stamp = '%FRONTEND_NPM_STAMP%'; if (-not (Test-Path $stamp)) { exit 1 }; $stampItem = Get-Item $stamp; $items = @(); foreach ($path in @('frontend\\package.json','frontend\\package-lock.json')) { if (Test-Path $path) { $items += Get-Item $path } }; if ($items | Where-Object { $_.LastWriteTime -gt $stampItem.LastWriteTime } | Select-Object -First 1) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEEDS_FRONTEND_DEPS=1"
)

if "%NEEDS_FRONTEND_DEPS%"=="1" (
    where npm > nul 2>&1
    if errorlevel 1 (
        echo FEHLER: npm wurde nicht gefunden. Bitte Node.js inklusive npm installieren.
        pause
        exit /b 1
    )
    echo Frontend-Abhaengigkeiten fehlen oder sind veraltet. Fuehre npm install aus...
    pushd "%FRONTEND_DIR%"
    call npm install
    if errorlevel 1 (
        popd
        echo FEHLER: npm install fehlgeschlagen.
        pause
        exit /b 1
    )
    > "%FRONTEND_NPM_STAMP%" echo installed
    popd
)

set "NEEDS_FRONTEND_BUILD=0"
if not exist "%FRONTEND_DIR%\dist\index.html" (
    set "NEEDS_FRONTEND_BUILD=1"
) else (
    powershell -NoProfile -Command "$dist = Get-Item 'frontend\\dist\\index.html'; $items = @(); foreach ($path in @('frontend\\index.html','frontend\\package.json','frontend\\package-lock.json','frontend\\tsconfig.app.json','frontend\\tsconfig.json','frontend\\tsconfig.node.json','frontend\\vite.config.ts','frontend\\vite.config.js','frontend\\vite.config.d.ts')) { if (Test-Path $path) { $items += Get-Item $path } }; if (Test-Path 'frontend\\src') { $items += Get-ChildItem 'frontend\\src' -Recurse -File }; if ($items | Where-Object { $_.LastWriteTime -gt $dist.LastWriteTime } | Select-Object -First 1) { exit 1 } else { exit 0 }"
    if errorlevel 1 set "NEEDS_FRONTEND_BUILD=1"
)

if "%NEEDS_FRONTEND_BUILD%"=="1" (
    echo Frontend-Build ist veraltet oder fehlt. Fuehre npm run build aus...
    pushd "%FRONTEND_DIR%"
    call npm run build
    if errorlevel 1 (
        popd
        echo FEHLER: Frontend-Build fehlgeschlagen.
        pause
        exit /b 1
    )
    popd
)

echo Nutze lokale venv unter "%CD%\venv" ...
echo Starte G3_EMBED auf http://127.0.0.1:8777 ...
title G3 EMBED Server
call "%VENV_PYTHON%" -m backend.genesis_embed_server

echo.
echo =================================================================
echo == G3_EMBED wurde beendet.
echo == Das Fenster bleibt offen, damit du den Log lesen kannst.
echo =================================================================
pause
