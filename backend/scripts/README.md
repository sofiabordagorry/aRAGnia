# Scripts del backend

Guía rápida de los scripts en esta carpeta.

## Antes de correrlos

1. Pararse en `backend/`.
2. Activar el entorno virtual.
3. Tener `backend/.env` configurado (si el script usa servicios o token).

Ejemplo (Windows PowerShell):

```powershell
cd backend
..\.venv\Scripts\Activate.ps1
```

## Lista de scripts

- `download_corpus.py`
  - Descarga PDFs/tablas usando la lista de `data/downloads_list.txt`.
  - Requiere `FING_TOKEN` en `.env`.
  - Uso: `python scripts/download_corpus.py`

- `docling_manual.py`
  - Convierte un archivo o una carpeta a JSON estructurado con Docling.
  - Uso (archivo): `python scripts/docling_manual.py corpus/13.pdf`
  - Uso (carpeta): `python scripts/docling_manual.py corpus`

- `chunk_corpus.py`
  - Genera chunks desde `data/docling/`.
  - Uso: `python scripts/chunk_corpus.py`

- `embed_chunks.py`
  - Genera embeddings desde `data/chunks/` y los guarda en `data/embeddings/`.
  - Uso: `python scripts/embed_chunks.py`

- `run_extraction.py`
  - Ejecuta extracción de entidades/relaciones y postprocesado.
  - Uso básico: `python scripts/run_extraction.py`
  - Opciones útiles:
    - `--max-docs N`
    - `--no-llm-researchers`
    - `--no-llm-topics`
    - `--enable_researcher_consolidation`

- `extract_end_to_end.py`
  - Corre un flujo de ingest end-to-end para un input configurado dentro del script.
  - Uso: `python scripts/extract_end_to_end.py`
  - Opciones:
    - `--debug`
    - `--enable_researcher_consolidation`

- `add_projects.py`
  - Ingresa varios proyectos predefinidos (paths hardcodeados en el script).
  - Uso: `python scripts/add_projects.py`

- `load_graph.py`
  - Carga entidades/relaciones en Neo4j desde `data/entities_relations/entity_documents.json`.
  - Uso: `python scripts/load_graph.py`

- `export_graph.py`
  - Exporta el grafo desde Neo4j a `data/entities_relations/export_graph.json`.
  - Uso: `python scripts/export_graph.py`

- `load_database.py`
  - Sube embeddings/metadata a Qdrant (colección demo).
  - Uso: `python scripts/load_database.py`

- `basic_knn_query.py`
  - Prueba KNN local con embeddings guardados en disco.
  - Uso: `python scripts/basic_knn_query.py`

- `demo_vector_store.py`
  - Demo de chunking + embeddings + carga en Qdrant + búsqueda.
  - Uso: `python scripts/demo_vector_store.py`

- `format.sh`
  - Formatea código del backend (black + orden de imports con ruff).
  - Uso: `./scripts/format.sh`

- `check.sh`
  - Corre chequeos del backend (ruff, black --check, mypy, pytest con cobertura).
  - Uso: `./scripts/check.sh`

## Orden recomendado del pipeline para probar GraphRAG

1. `python scripts/download_corpus.py`
2. `python scripts/docling_manual.py corpus`
3. `python scripts/chunk_corpus.py`
4. `python scripts/run_extraction.py`
5. `python scripts/load_graph.py`
