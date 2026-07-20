#!/usr/bin/env bash

# =========================================================
# Configuración inicial
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR" || {
    echo "ERROR: No se pudo acceder a la carpeta del proyecto."
    exit 1
}

VENV="$SCRIPT_DIR/.venv"
PYTHON_VENV="$VENV/bin/python"

ENV_FILE="$SCRIPT_DIR/backend/.env"
COMPOSE_FILE="$SCRIPT_DIR/backend/docker-compose.yml"
FRONTEND_DIR="$SCRIPT_DIR/frontend/public"

BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"

BACKEND_PID_FILE="$SCRIPT_DIR/backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/frontend.pid"

# =========================================================
# Verificar archivos y programas necesarios
# =========================================================

if ! command -v docker >/dev/null 2>&1; then
    echo
    echo "ERROR: Docker no está instalado o no está disponible."
    echo "Instalá Docker y asegurate de que esté en ejecución."
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo
    echo "ERROR: Docker no está en ejecución."
    echo "Abrí Docker Desktop o iniciá el servicio de Docker."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo
    echo "ERROR: No se encontró el archivo backend/.env."
    echo "Copiá backend/.env.example como backend/.env"
    echo "y configurá las credenciales de Neo4j y PostgreSQL."
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    echo
    echo "ERROR: No se encontró backend/docker-compose.yml."
    exit 1
fi

if [ ! -d "$FRONTEND_DIR" ]; then
    echo
    echo "ERROR: No se encontró la carpeta frontend/public."
    exit 1
fi

# =========================================================
# Levantar contenedores Docker
# =========================================================

echo "Levantando contenedores Docker..."

docker compose \
    --env-file "$ENV_FILE" \
    -f "$COMPOSE_FILE" \
    up -d

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: No se pudieron levantar los contenedores Docker."
    exit 1
fi

# =========================================================
# Crear el entorno virtual si no existe
# =========================================================

if [ ! -x "$PYTHON_VENV" ]; then
    echo
    echo "No se encontró el entorno virtual .venv."
    echo "Creando entorno virtual con Python 3.11..."

    if command -v python3.11 >/dev/null 2>&1; then
        python3.11 -m venv "$VENV"
        VENV_RESULT=$?
    elif command -v python3 >/dev/null 2>&1; then
        echo 'No se encontró el comando "python3.11".'
        echo 'Intentando crear el entorno con "python3"...'
        python3 -m venv "$VENV"
        VENV_RESULT=$?
    elif command -v python >/dev/null 2>&1; then
        echo 'No se encontró el comando "python3".'
        echo 'Intentando crear el entorno con "python"...'
        python -m venv "$VENV"
        VENV_RESULT=$?
    else
        echo
        echo "ERROR: No se encontró Python."
        echo "Verificá que Python 3.11 esté instalado."
        exit 1
    fi

    if [ "$VENV_RESULT" -ne 0 ] || [ ! -x "$PYTHON_VENV" ]; then
        echo
        echo "ERROR: No se pudo crear el entorno virtual."
        echo "Verificá que Python 3.11 y el módulo venv estén instalados."
        echo
        echo "En Ubuntu o Debian puede ser necesario ejecutar:"
        echo "sudo apt install python3.11-venv"
        exit 1
    fi

    echo
    echo "Actualizando pip..."

    "$PYTHON_VENV" -m pip install --upgrade pip

    if [ $? -ne 0 ]; then
        echo
        echo "ERROR: No se pudo actualizar pip."
        exit 1
    fi

    echo
    echo "Instalando las dependencias del proyecto..."

    "$PYTHON_VENV" -m pip install -e "$SCRIPT_DIR/backend[dev]"

    if [ $? -ne 0 ]; then
        echo
        echo "ERROR: No se pudieron instalar las dependencias."
        exit 1
    fi

    echo
    echo "Entorno virtual creado correctamente."
else
    echo "Entorno virtual encontrado."
fi

# =========================================================
# Evitar iniciar procesos duplicados
# =========================================================

if [ -f "$BACKEND_PID_FILE" ]; then
    OLD_BACKEND_PID="$(cat "$BACKEND_PID_FILE" 2>/dev/null)"

    if [ -n "$OLD_BACKEND_PID" ] &&
       kill -0 "$OLD_BACKEND_PID" 2>/dev/null; then
        echo
        echo "ADVERTENCIA: El backend ya parece estar ejecutándose."
        echo "PID: $OLD_BACKEND_PID"
        echo "No se iniciará otro backend."
        BACKEND_ALREADY_RUNNING=true
    else
        rm -f "$BACKEND_PID_FILE"
        BACKEND_ALREADY_RUNNING=false
    fi
else
    BACKEND_ALREADY_RUNNING=false
fi

if [ -f "$FRONTEND_PID_FILE" ]; then
    OLD_FRONTEND_PID="$(cat "$FRONTEND_PID_FILE" 2>/dev/null)"

    if [ -n "$OLD_FRONTEND_PID" ] &&
       kill -0 "$OLD_FRONTEND_PID" 2>/dev/null; then
        echo
        echo "ADVERTENCIA: El frontend ya parece estar ejecutándose."
        echo "PID: $OLD_FRONTEND_PID"
        echo "No se iniciará otro frontend."
        FRONTEND_ALREADY_RUNNING=true
    else
        rm -f "$FRONTEND_PID_FILE"
        FRONTEND_ALREADY_RUNNING=false
    fi
else
    FRONTEND_ALREADY_RUNNING=false
fi

# =========================================================
# Iniciar backend
# =========================================================

if [ "$BACKEND_ALREADY_RUNNING" = false ]; then
    echo
    echo "Iniciando backend..."

    cd "$SCRIPT_DIR/backend" || {
        echo "ERROR: No se pudo acceder a la carpeta backend."
        exit 1
    }

    nohup "$PYTHON_VENV" -m uvicorn \
        institutional_graphrag.api.main:app \
        --reload \
        --port 8000 \
        > "$BACKEND_LOG" 2>&1 &

    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

    cd "$SCRIPT_DIR" || exit 1
else
    BACKEND_PID="$OLD_BACKEND_PID"
fi

# =========================================================
# Iniciar frontend
# =========================================================

if [ "$FRONTEND_ALREADY_RUNNING" = false ]; then
    echo "Iniciando frontend..."

    cd "$FRONTEND_DIR" || {
        echo "ERROR: No se pudo acceder a frontend/public."
        exit 1
    }

    nohup "$PYTHON_VENV" -m http.server 5500 \
        > "$FRONTEND_LOG" 2>&1 &

    FRONTEND_PID=$!
    echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

    cd "$SCRIPT_DIR" || exit 1
else
    FRONTEND_PID="$OLD_FRONTEND_PID"
fi

# =========================================================
# Comprobar que los procesos hayan iniciado
# =========================================================

sleep 2

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo
    echo "ERROR: El backend no pudo iniciarse."
    echo "Revisá el archivo:"
    echo "$BACKEND_LOG"
    rm -f "$BACKEND_PID_FILE"
    exit 1
fi

if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo
    echo "ERROR: El frontend no pudo iniciarse."
    echo "Revisá el archivo:"
    echo "$FRONTEND_LOG"
    rm -f "$FRONTEND_PID_FILE"
    exit 1
fi

# =========================================================
# Información final
# =========================================================

echo
echo "Proyecto iniciado correctamente."
echo
echo "Backend:       http://localhost:8000"
echo "Documentación: http://localhost:8000/docs"
echo "Frontend:      http://localhost:5500"
echo
echo "PID del backend:  $BACKEND_PID"
echo "PID del frontend: $FRONTEND_PID"
echo
echo "Registros:"
echo "  Backend:  $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"
echo
echo "Los servicios seguirán ejecutándose aunque cierres esta terminal."
echo

read -r -p "Presioná Enter para cerrar..."