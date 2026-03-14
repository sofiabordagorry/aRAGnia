#!/bin/bash

# Ir al directorio donde está el script
cd "$(dirname "$0")"

echo "Levantando contenedores Docker..."
docker compose -f backend/docker-compose.yml up -d

# Activar entorno virtual
source .venv/bin/activate

echo "Iniciando backend..."
(
  cd backend
  uvicorn institutional_graphrag.api.main:app --reload --port 8000
) &

echo "Iniciando frontend..."
(
  cd frontend/public
  python -m http.server 5500
) &

wait
