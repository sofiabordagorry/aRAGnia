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

El proyecto incluye un script (en la carpeta backend)  para descargar los PDF de la nube institucional a partir de una lista de URLs, las cuales corresponden a pares de documentos (Informe Final y Propuesta de Postulación). Además, se incorporaron tres documentos adicionales que contienen resultados o conclusiones de varios proyectos correspondientes a un año determinado.

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

Procesar un PDF individual:

#### Procesar un PDF individual:

```bash
python .\scripts\docling_manual.py corpus\13.pdf
```

#### Procesar todo el corpus:

```bash
python .\scripts\docling_manual.py corpus
```

Los archivos JSON se generan en /institutional-graphrag/data/docling.

### Generar chunks usando los archivos JSON

Se transforman los Json de /institutional-graphrag/data/docling en chunks ubicados en
/institutional-graphrag/data/chunks.

```bash
python .\scripts\chunk_corpus.py
```

### Levantar Servidor backend

1. Ubicarse en carpeta

```text
   institutional-graphrag/backend/src
```

2. Una vez realizada la instalación de dependencias mencionada anteriormente, ejecutá:

```bash
uvicorn institutional_graphrag.api.main:app --reload --port 8000
```

3. El backend quedará disponible en:

```text
http://localhost:8000
```

### Levantar Frontend

cd institutional-graphrag/frontend/public
python -m http.server 5500
