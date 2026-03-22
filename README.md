# Institutional GraphRAG

Proyecto de grado de Ingeniería en Computación (FIng - Udelar, 2025-2026).

Este repositorio implementa un pipeline de GraphRAG para documentos de proyectos institucionales. El objetivo es poder procesar informes/propuestas, construir conocimiento estructurado y hacer consultas sobre la evolución de los proyectos.

## Stack del proyecto

- Backend: FastAPI + Python
- Frontend: HTML/CSS/JS estático
- Base de grafo: Neo4j
- Base vectorial: Qdrant
- Base relacional: PostgreSQL
- LLM local/opcional: Ollama

## Estructura general

- `backend/`: API, lógica de ingestión/retrieval, tests y scripts
- `frontend/`: interfaz web estática
- `data/`: corpus, outputs intermedios (docling, chunks, embeddings, etc.)
- `run_proyect.bat`: arranque rápido en Windows
- `run_proyect.sh`: arranque rápido en Linux/macOS

## Requisitos

- Python 3.11
- Docker + Docker Compose

## Configuración inicial

1. Clonar el repo y ubicarse en la raíz del proyecto.
2. Crear entorno virtual:

```bash
python -m venv .venv
```

3. Activar entorno virtual:

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

4. Instalar dependencias del backend:

```bash
cd backend
pip install -e ".[dev]"
cd ..
```

5. Instalar dependencias del frontend (Prettier):

```bash
cd frontend
npm install
cd ..
```

6. Configurar variables de entorno:

- Copiar `backend/.env.example` a `backend/.env`.
- Completar al menos:
	- `NEO4J_USER`
	- `NEO4J_PASSWORD`
	- `POSTGRES_USER`
	- `POSTGRES_PASSWORD`
	- `FING_TOKEN` (si se va a descargar corpus)
	- `GROQ_API_KEY` (si se va a usar Groq)

## Levantar el proyecto (modo recomendado)

Desde la raíz, usar el script `run_proyect`:

Windows:

```powershell
.\run_proyect.bat
```

Linux/macOS:

```bash
chmod +x run_proyect.sh
./run_proyect.sh
```

El script realiza lo siguiente:

1. Levanta contenedores con `backend/docker-compose.yml`.
2. Inicia el backend con `uvicorn` en puerto 8000.
3. Inicia el frontend estático en puerto 5500.

Accesos útiles:

- Frontend: http://localhost:5500/
- API (health): http://localhost:8000/health
- Docs FastAPI: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474/browser/
- Qdrant: http://localhost:6333/

## Proveedor de modelo (LLM)

El backend soporta dos opciones principales:

- Ollama (local): usa `OLLAMA_BASE_URL` (por defecto `http://localhost:11434`).
- Groq (API): requiere `GROQ_API_KEY` en `backend/.env`.

Si se levanta el proyecto con Docker Compose, el servicio `ollama-init` descarga automáticamente modelos base (`llama3.2:3b` y `qwen2.5:3b-instruct`).

Para verificar modelos en Ollama:

```bash
cd backend
docker compose exec ollama ollama list
```

## Pipeline de datos (scripts)

Todos estos comandos se ejecutan desde `backend/` con el entorno virtual activo.

1. Descargar corpus (requiere `FING_TOKEN` en `.env`):

- La lista de URLs está en `data/downloads_list.txt`.
- Si quieren cambiar qué PDFs se descargan, editen ese archivo (agregar/quitar URLs).
- Los archivos descargados se guardan en `data/corpus/`.

```bash
python .\scripts\download_corpus.py
```

2. Convertir PDFs con Docling:

Un archivo:

```bash
python .\scripts\docling_manual.py corpus\13.pdf
```

Corpus completo:

```bash
python .\scripts\docling_manual.py corpus
```

3. Generar chunks:

```bash
python .\scripts\chunk_corpus.py
```

4. Generar embeddings:

```bash
python .\scripts\embed_chunks.py
```

5. Ejecutar extracción end-to-end:

```bash
python .\scripts\extract_end_to_end.py
```

Outputs esperados en `data/`: `docling/`, `chunks/`, `embeddings/`, `entities_relations/`.

## Checks y formato

### Backend

Desde `backend/`:

```bash
./scripts/format.sh
./scripts/check.sh
```

`check.sh` ejecuta:

- ruff
- black --check
- mypy
- pytest con cobertura

### Frontend

Desde `frontend/`:

Linux/macOS:

```bash
./scripts/format.sh
./scripts/check.sh
```

Windows:

```bat
scripts\format.bat
scripts\check.bat
```

## Notas rápidas

- En el primer arranque, `ollama-init` puede demorar porque descarga modelos.