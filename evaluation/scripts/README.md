# Scripts de evaluación

Guía rápida de los scripts en esta carpeta.

## Lista de scripts

- `question_to_answer.py`
  - Lee preguntas desde un archivo (`.txt`, `.json`, `.jsonl`) y consulta solo GraphRAG.
  - Reutiliza la lógica del endpoint interno de GraphRAG para evitar duplicar pipeline.
  - Salida:
    - Reporte completo en `evaluation/ground_truth/question_to_answer/query_batch_report.json`.
    - Resumen en `evaluation/ground_truth/question_to_answer/query_batch_answers.jsonl` con `id`, `question`, `answer`, `ok`.
  - Opcional: `--include-debug` para guardar `cypher_query`, `chunks`, `chunk_to_entities`, `entities_extracted` y `relationships_extracted`.
  - Uso:
    - `python evaluation/scripts/question_to_answer.py --questions-file data/results/preguntas.txt`
    - `python evaluation/scripts/question_to_answer.py --questions-file data/results/preguntas.json --include-debug`

- `evaluate_kg.py`
  - Compara un grafo proporcionado contra el ground truth (evaluation/ground_truth/extraction/ground_truth_kg.json).
  - Genera un reporte HTML y json lado a lado de entidades y relaciones.
  - También valida la estructura del grafo contra el esquema definido.
  - Salida:
    - `Reporte comparativo en evaluation/ground_truth/results_knowledge_graph/report_complete.html`
    - `Datos renderizados en evaluation/ground_truth/results_knowledge_graph/report_render_data.json`
    - `Reporte de validación estructural en evaluation/ground_truth/results_knowledge_graph/report_structure_graph.json`
    - `Reporte HTML de validación estructural en evaluation/ground_truth/results_knowledge_graph/report_structure_graph.html`
  - Uso:
    - `python evaluation/scripts/evaluate_kg.py evaluation/llm_extraction/Qwen_Qwen2.5-Coder-14B-Instruct/entity_documents.json`

- `evaluate_all_llms.py`
  - Evalúa todos los modelos en `evaluation/llm_extraction/` contra el ground truth de extracción.
  - Calcula Precision, Recall y F1 macro-promediados por proyecto, tanto global como por tipo de entidad y tipo de relación.
  - Valida cada grafo contra el esquema definido.
  - Requiere `matplotlib` (`pip install matplotlib`).
  - Salida:
    - Resumen JSON en `evaluation/results/extraction/llm_comparison_summary.json`
    - Reporte HTML en `evaluation/results/extraction/llm_comparison_report.html`
    - Gráficas PNG en `evaluation/results/extraction/images/`
  - Uso:
    - `python evaluation/scripts/evaluate_all_llms.py`

- `evaluate_cypher.py`
  - Lee preguntas desde un archivo JSON.
  - Genera automáticamente queries Cypher y ejecuta las consultas en Neo4j.
  - Recupera chunks y contexto asociado a cada pregunta.
  - Incluye logs de progreso y manejo de errores por pregunta.

  - Salida:
    - Reporte completo en `evaluation/ground_truth/question_to_cypher/query_batch_report.json`
    - Respuestas resumidas en `evaluation/ground_truth/question_to_cypher/query_batch_answers.jsonl`

  - Uso:
    - `python evaluation/scripts/evaluate_cypher.py evaluation/ground_truth/datasetQA_GT.json`

- `generate_gt_answers.py`
  - Genera respuestas en lenguaje natural a partir de un subgrafo recuperado.
  - Toma un JSON cualquiera cuyos items tengan los campos `pregunta` (pregunta en lenguaje natural) y `retrieved_subgraph` (subgrafo recuperado del grafo) y, para cada uno, genera/completa el campo `answer`.
  - Reutiliza la misma lógica de generación de respuestas del pipeline GraphRAG (`answer_llm_client`), sin necesidad de conectarse a Neo4j. Funciona con Ollama o HuggingFace según la variable de entorno `LLM_BACKEND`.
  - Los mensajes centinela (`fuera de alcance`, `no se encontró información`) se resuelven directamente, sin invocar al LLM.

  - Entrada:
    - Un archivo JSON (lista de objetos, o un objeto con la clave `questions`) donde cada item tenga al menos `pregunta` y `retrieved_subgraph` (por ejemplo, la salida de `result_cypher_GT.py`).

  - Salida:
    - Escribe en `<nombre_entrada>_answers.json` (no modifica el archivo de entrada), agregando/completando el campo `answer` en cada item.
    - Por defecto solo completa las respuestas vacías; usar `--overwrite` para regenerarlas todas.

  - Uso:
    - `python evaluation/scripts/generate_gt_answers.py preguntas.json`
    - `python evaluation/scripts/generate_gt_answers.py preguntas.json --overwrite`

- `evaluate_generation.py`
  - Evalúa la **generación** de respuestas en lenguaje natural sobre el GT de validación (`datasetQA_GT.json`): responde qué combinación de LLM y prompt genera mejores respuestas y mide los tiempos de espera.
  - Para cada combinación de modelo × variante de prompt:
    - Genera la respuesta desde `pregunta` + `retrieved_subgraph`, midiendo la latencia de cada generación.
    - Juzga la respuesta contra el `answer` de referencia con LLM-as-a-judge vía API de Anthropic (correctitud factual, completitud, fidelidad; escala 1-5 normalizada a 0-1).
    - Los items centinela (fuera de alcance / sin info) se evalúan de forma determinística (exact-match), sin invocar al juez.
  - Configuración: editar las constantes `MODELS` (lista de `{display, backend, model}`) y `PROMPT_VARIANTS` en el script, o pasar `--models-file`.
  - Requiere `ANTHROPIC_API_KEY` (en el entorno o en `backend/.env`) salvo que se use `--no-judge`. Requiere `matplotlib`.
  - Salida:
    - Resumen agregado en `evaluation/results/generation/generation_comparison_summary.json`
    - Detalle por pregunta en `evaluation/results/generation/generation_details.json`
    - Gráficas PNG en `evaluation/results/generation/images/`
    - Reporte HTML en `evaluation/results/generation/generation_report.html`
  - Uso:
    - `python evaluation/scripts/evaluate_generation.py`
    - `python evaluation/scripts/evaluate_generation.py --max-questions 3` (smoke test)
    - `python evaluation/scripts/evaluate_generation.py --no-judge` (solo genera y mide latencias)
    - `python evaluation/scripts/evaluate_generation.py --judge-model claude-opus-4-8`
    
- `evaluate_topic_classification.py`
  - Ayuda a **definir los parámetros de la clasificación de tópicos** (issue #234) probando combinaciones de parámetros y comparando contra el GT de tópicos (`ground_truth_kg.json`, relación `TIENE_TOPICO`).
  - Parámetros (cada uno es una **lista**; se evalúan **todas las combinaciones** = producto cartesiano):
    - `CHUNKING_THRESHOLDS`: umbral de la etapa de chunking (score mínimo por chunk).
    - `COVERAGE_THRESHOLDS` / `CONFIDENCE_THRESHOLDS`: los dos umbrales de la etapa final (agregación por proyecto).
    - `CHUNK_SIZES`: tamaño del chunk; al variarlo **re-chunkea** desde `data/docling/` y **re-corre BERT** (es el caso más caro). No re-corre el parseo de Docling: el `DoclingDocument` no depende del tamaño de chunk.
  - Para dejar un parámetro fijo, se le pone una lista de un solo valor; para comparar varios, varios. Así se pueden barrer **uno o varios parámetros a la vez** (ej. una grilla chunking × coverage).
  - Optimización: BERT depende solo de `chunk_size`, así que corre **una vez por tamaño distinto** (cachea las predicciones crudas en `evaluation/results/topic_classification/cache/`) y todas las combinaciones de umbrales se calculan en Python sobre esas predicciones.
  - Compara por **nombre normalizado del tópico** (no por id) para ser robusto a que el GT de tópicos está pendiente de actualización con los valores de los profes.
  - Las corridas son **acumulativas**: cada vez que se prueban nuevas combinaciones se mergean en el reporte (por firma de parámetros), se re-etiquetan según qué parámetros varían y se regeneran las gráficas (poner `FRESH = True` o pasar `--fresh` para empezar de cero).
  - **Configuración**: se edita el bloque de CONSTANTES en MAYÚSCULA al principio del archivo (`CHUNKING_THRESHOLDS`, `COVERAGE_THRESHOLDS`, `CONFIDENCE_THRESHOLDS`, `CHUNK_SIZES`, etc.) y se corre el script sin argumentos. Opcionalmente hay flags que sobrescriben cada lista para una corrida puntual.
  - Requiere `matplotlib` y que el paquete `institutional_graphrag` esté instalado (mismo entorno que el pipeline). No usa Neo4j.
  - Salida (en `evaluation/results/topic_classification/`):
    - Resumen JSON en `topic_params_summary.json`
    - Detalle por proyecto en `topic_params_details.json`
    - Gráficas PNG en `images/`
    - Reporte HTML en `topic_params_report.html` (tablas + gráficas, mismo formato que los demás reportes)
  - Uso:
    - Editar las constantes y correr: `python evaluation/scripts/evaluate_topic_classification.py`
    - Grilla por flags (chunking × coverage): `python evaluation/scripts/evaluate_topic_classification.py --chunking-thresholds 0.04,0.06,0.08 --coverage-thresholds 0.4,0.6`
    - Tamaño de chunk (re-chunkea + re-corre BERT): `python evaluation/scripts/evaluate_topic_classification.py --chunk-sizes 256,384,512`

- `result_cypher_GT.py`
  - Ejecuta queries Cypher almacenadas en un archivo JSON.
  - Recupera el subgrafo asociado a cada consulta y lo guarda en el mismo JSON.
  - Genera contexto agregado a partir de los resultados recuperados.

  - Entrada:
    - `datasetQA_GT.json`

  - Salida:
    - Actualiza el mismo archivo JSON agregando datos en el campo `retrieved_subgraph`.

  - Uso:
    - `python evaluation/scripts/result_cypher_GT.py`
