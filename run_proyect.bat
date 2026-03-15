@echo off
cd /d %~dp0

echo Levantando contenedores Docker...
docker compose -f backend\docker-compose.yml up -d

call .venv\Scripts\activate

echo Iniciando backend...
start cmd /k "cd backend && uvicorn institutional_graphrag.api.main:app --reload --port 8000"

echo Iniciando frontend...
start cmd /k "cd frontend/public && python -m http.server 5500"

pause