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

- `table_validator.py`
  - Valida y limpia la estructura de una tabla CSV y detecta filas con cantidad incorrecta de columnas.
  - Entrada:
    - CSV de entrada por par�metro posicional obligatorio `csv_path`.
  - Salida:
    - Reporte en consola con filas inv�lidas (si existen).
    - CSV limpio en `data/tables/` con el mismo nombre del archivo de entrada.
  - Uso: `python scripts/table_validator.py <ruta/al/archivo.csv>`

- `prepare_corpus.py`
  - Valida la estructura de carpetas de proyectos, renombra archivos y los guarda en `data/corpus/`.
  - Entrada:
    - Carpeta raíz.
    - Formato esperado (proyectos):
    - `proyectos_2018/`
    - `  informes_propuestas_XXX/`
    - `    63/` (id_formulario)
    - `      informe_XXX/`
    - `        archivo.pdf` o `archivo.odt`
    - `      propuesta_XXX/`
    - `        archivo.pdf` o `archivo.odt`
  - Salida:
    - Archivos renombrados en `data/corpus/`.
  - Uso: `python scripts/prepare_corpus.py <ruta/a/carpeta>`

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
  - Opcion útil:
    - `--max-project N`

- `extract_end_to_end.py`
  - Corre un flujo de ingest end-to-end para un input configurado dentro del script.
  - Salida principal: entidades/relaciones consolidadas en `data/entities_relations/entity_extraction_web_<id>.json`.
  - Si se usa `--debug`, también deja artefactos intermedios en `data/corpus/`, `data/docling/`, `data/chunks/` y `data/embeddings/`.
  - Uso: `python scripts/extract_end_to_end.py`
  - Opciones:
    - `--debug`
    - `--no-headings`

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

1. `python scripts/table_validator.py <ruta/al/archivo.csv>`
2. `python scripts/prepare_corpus.py <ruta/a/carpeta>`
3. `python scripts/docling_manual.py corpus`
4. `python scripts/chunk_corpus.py`
5. `python scripts/run_extraction.py`
6. `python scripts/load_graph.py`
