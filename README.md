# Institutional GraphRAG

Proyecto de grado de Ingeniería en Computación (FIng – Udelar) 2025-2026.

Este repositorio contiene el código de un pipeline incremental de GraphRAG para trabajar con informes de proyectos institucionales y permitir consultas longitudinales sobre los mismos.

## Requisitos

- Python 3.11
- pip

## Instalación

Se recomienda usar un entorno virtual.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv/Scripts/activate
```

### Backend

```bash
cd backend
pip install -e ".[dev]"
```

### Frontend

```bash
cd frontend
npm init -y
npm config set strict-ssl false
npm install -D prettier
```

## Ejecución

### Formatear código

#### Backend

ubicacion /backend/

```bash
./scripts/format.sh
```

#### Frontend

ubicacion /frontend/

```bash
./scripts/format.sh
```

### Ejecutar tests y verificaciones

#### Backend

ubicacion /backend/

```bash
./scripts/check.sh
```

#### Frontend

ubicacion /frontend/

```bash
./scripts/check.sh
```

Este comando ejecuta:

- Linter (ruff)
- Verificación de formato (black)
- Type checking (mypy)
- Tests con cobertura (pytest)

### Descargar corpus de PDFs

El proyecto incluye un script (en la carpeta backend) para descargar los PDF de la nube institucional a partir de una lista de URLs, las cuales corresponden a pares de documentos (Informe Final y Propuesta de Postulación). Además, se incorporaron tres documentos adicionales que contienen resultados o conclusiones de varios proyectos correspondientes a un año determinado.

- La lista de URLs se encuentra en data/downloads_list.txt.

- Los PDFs descargados se guardan en la carpeta data/corpus.

- El acceso a la nube se realiza mediante un enlace público de Nextcloud, cuyo token debe configurarse a través de una variable de entorno.

#### Configuración del token

Antes de ejecutar el script, es necesario crear un archivo `.env` en la carpeta backend del proyecto con una variable de entorno llamada **FING_TOKEN**

#### Ejecutar el script de descarga

```bash
python .\scripts\download_corpus.py
```

### Convertir PDFs en Documentos Estructurados Utilizando Dockling

Ubicarse en la ruta /institutional-graphrag/backend

#### Procesar un PDF individual:

```bash
python .\scripts\docling_manual.py corpus\13.pdf
```

#### Procesar todo el corpus:

```bash
python .\scripts\docling_manual.py corpus
```

Los archivos Json se generan en /institutional-graphrag/data/docling.

### Generar chunks usando los archivos Json

Se transforman los Json de /institutional-graphrag/data/docling en chunks ubicados en
/institutional-graphrag/data/chunks.

```bash
python .\scripts\chunk_corpus.py
```

### Crear embeddings de los chunks

Se transforman los Json de /institutional-graphrag/data/chunks en chunks ubicados en
/institutional-graphrag/data/embeddings.

Para Windows: antes correr en consola el siguiente comando para poder usar cuda (en GPUs NVIDIA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Para MAC: antes correr el siguiente comando para prevenir fallos en caso de funcionalidades aún no implementadas para MPS

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Luego:

```bash
python .\scripts\embed_chunks.py
```

### Levantar servicios con Docker Compose

Desde `backend/docker-compose.yml` se puede levantar Neo4j, Qdrant y Ollama juntos:

```bash
cd backend
docker compose up -d
```

### Elegir provedor del modelo (llm)

Se configura el api/router_rag.py

#### Ollama

- Si usás Docker Compose, el servicio queda expuesto en `http://localhost:11434`.
- Después de levantar el compose, descargá dentro del contenedor el modelo a utilizar:

```bash
cd backend
docker compose exec ollama ollama pull llama3.2:3b
```

- Si no usás Docker, también podés seguir instalando Ollama nativo desde ollama.com.
- La URL del servicio se puede configurar con `OLLAMA_BASE_URL` en `backend/.env`.

#### Groq

Para pruebas, no es gratis pero va rapido.
obtener una key en la aplicacion https://console.groq.com/keys y guardarla en el .env "GROQ_API_KEY"

#### Local

- Es la mas lenta

### Levantar Servidor

- Desde la carpeta raiz ejecutar el archivo run_proyect
- el frontend se encuentra en http://localhost:5500/
- neo4j se encuentra en http://localhost:7474/browser/
