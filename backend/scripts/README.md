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
  - Salida:
    - PDFs en `data/corpus/`
    - Tablas extraídas en `data/tables/` (parquet)
  - Uso: `python scripts/download_corpus.py`

- `docling_manual.py`
  - Convierte un archivo o una carpeta a JSON estructurado con Docling.
  - Salida: JSONs en `data/docling/`.
  - Uso (archivo): `python scripts/docling_manual.py corpus/13.pdf`
  - Uso (carpeta): `python scripts/docling_manual.py corpus`

- `chunk_corpus.py`
  - Genera chunks desde `data/docling/`.
  - Salida: archivos `*_chunks.json` en `data/chunks/`.
  - Uso: `python scripts/chunk_corpus.py`

- `embed_chunks.py`
  - Genera embeddings desde `data/chunks/` y los guarda en `data/embeddings/`.
  - Salida:
    - Embeddings en `data/embeddings/*.npy`
    - Metadata por documento en `data/embeddings/*_metadata.json`
  - Uso: `python scripts/embed_chunks.py`

- `run_extraction.py`
  - Ejecuta extracción de entidades/relaciones y postprocesado.
  - Salida: `data/entities_relations/entity_documents.json` (y luego se postprocesa sobre ese archivo).
  - Uso básico: `python scripts/run_extraction.py`
  - Opciones útiles:
    - `--max-docs N`
    - `--no-llm-researchers`
    - `--no-llm-topics`
    - `--enable_researcher_consolidation`

- `extract_end_to_end.py`
  - Corre un flujo de ingest end-to-end para un input configurado dentro del script.
  - Salida principal: entidades/relaciones consolidadas en `data/entities_relations/entity_extraction_web_<id>.json`.
  - Si se usa `--debug`, también deja artefactos intermedios en `data/corpus/`, `data/docling/`, `data/chunks/` y `data/embeddings/`.
  - Uso: `python scripts/extract_end_to_end.py`
  - Opciones:
    - `--debug`
    - `--enable_researcher_consolidation`

- `add_projects.py`
  - Ingresa varios proyectos predefinidos (paths hardcodeados en el script).
  - Salida: igual que `extract_end_to_end.py` (crea/actualiza `data/entities_relations/entity_extraction_web_<id>.json` y, al estar en modo debug, conserva artefactos intermedios en `data/`).
  - Uso: `python scripts/add_projects.py`

- `load_graph.py`
  - Carga entidades/relaciones en Neo4j desde `data/entities_relations/entity_documents.json`.
  - Salida: nodos y relaciones en Neo4j (no genera archivo local).
  - Uso: `python scripts/load_graph.py`

- `export_graph.py`
  - Exporta el grafo desde Neo4j a `data/entities_relations/export_graph.json`.
  - Salida: `data/entities_relations/export_graph.json`.
  - Uso: `python scripts/export_graph.py`

- `load_database.py`
  - Sube embeddings/metadata a Qdrant (colección demo).
  - Salida: vectores y payload en Qdrant (`demo_collection`, no genera archivo local).
  - Uso: `python scripts/load_database.py`

- `basic_knn_query.py`
  - Prueba KNN local con embeddings guardados en disco.
  - Salida: no genera archivos; imprime top-k resultados en consola usando `data/embeddings/`.
  - Uso: `python scripts/basic_knn_query.py`

- `question_to_answer.py`
  - Lee preguntas desde un archivo (`.txt`, `.json`, `.jsonl`) y consulta solo GraphRAG.
  - Reutiliza la lógica del endpoint interno de GraphRAG para evitar duplicar pipeline.
  - Salida:
    - Reporte completo en `data/results/query_batch_report.json`.
    - Resumen en `data/results/query_batch_answers.jsonl` con `id`, `question`, `answer`, `ok`.
  - Opcional: `--include-debug` para guardar `cypher_query`, `chunks`, `chunk_to_entities`, `entities_extracted` y `relationships_extracted`.
  - Uso:
    - `python scripts/question_to_answer.py --questions-file data/results/preguntas.txt`
    - `python scripts/question_to_answer.py --questions-file data/results/preguntas.json --include-debug`

- `demo_vector_store.py`
  - Demo de chunking + embeddings + carga en Qdrant + búsqueda.
  - Salida: carga de ejemplo en Qdrant (`demo_collection`) y resultados de búsqueda en consola (no escribe archivos).
  - Uso: `python scripts/demo_vector_store.py`

- `format.sh`
  - Formatea código del backend (black + orden de imports con ruff).
  - Salida: modifica archivos Python del backend en el lugar.
  - Uso: `./scripts/format.sh`

- `check.sh`
  - Corre chequeos del backend (ruff, black --check, mypy, pytest con cobertura).
  - Salida: reporte en consola y código de salida del proceso (no genera artefactos del pipeline de datos).
  - Uso: `./scripts/check.sh`

## Orden recomendado del pipeline para probar GraphRAG

1. `python scripts/download_corpus.py`
2. `python scripts/docling_manual.py corpus`
3. `python scripts/chunk_corpus.py`
4. `python scripts/run_extraction.py`
5. `python scripts/load_graph.py`
