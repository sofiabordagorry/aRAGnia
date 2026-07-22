@echo off
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "PYTHON_VENV=%VENV%\Scripts\python.exe"

echo Levantando contenedores Docker...
docker compose --env-file "backend\.env" -f "backend\docker-compose.yml" up -d

if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron levantar los contenedores Docker.
    pause
    exit /b 1
)

rem =========================================================
rem Crear el entorno virtual si no existe
rem =========================================================

if not exist "%PYTHON_VENV%" (
    echo.
    echo No se encontro el entorno virtual .venv.
    echo Creando entorno virtual con Python 3.11...

    where py >nul 2>&1

    if not errorlevel 1 (
        py -3.11 -m venv "%VENV%"
    ) else (
        echo No se encontro el comando "py".
        echo Intentando crear el entorno con "python"...
        python -m venv "%VENV%"
    )

    if errorlevel 1 (
        echo.
        echo ERROR: No se pudo crear el entorno virtual.
        echo Verifica que Python 3.11 este instalado.
        pause
        exit /b 1
    )

    echo.
    echo Actualizando pip...
    "%PYTHON_VENV%" -m pip install --upgrade pip

    if errorlevel 1 (
        echo ERROR: No se pudo actualizar pip.
        pause
        exit /b 1
    )

    echo.
    echo Instalando las dependencias del proyecto...
    "%PYTHON_VENV%" -m pip install -e ".\backend[dev]"

    if errorlevel 1 (
        echo.
        echo ERROR: No se pudieron instalar las dependencias.
        pause
        exit /b 1
    )

    echo.
    echo Entorno virtual creado correctamente.
) else (
    echo Entorno virtual encontrado.
)

rem =========================================================
rem Iniciar backend y frontend
rem =========================================================

echo.
echo Iniciando backend...
start "Backend" /D "%~dp0backend" cmd /k ""%PYTHON_VENV%" -m uvicorn institutional_graphrag.api.main:app --reload --port 8000"

echo Iniciando frontend...
start "Frontend" /D "%~dp0frontend\public" cmd /k ""%PYTHON_VENV%" -m http.server 5500"

echo.
echo Proyecto iniciado.
echo Backend: http://localhost:8000
echo Documentacion: http://localhost:8000/docs
echo Frontend: http://localhost:5500
echo.

pause