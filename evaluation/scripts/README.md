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
    - `python evaluation/scripts/evaluate_kg.py evaluation/llm_extraction/Qwen_Qwen2.5-coder-14B-Instruct/entity_documents.json`

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

- `result_cypher_GT.py`
  - Ejecuta queries Cypher almacenadas en un archivo JSON.
  - Recupera el subgrafo asociado a cada consulta y lo guarda en el mismo JSON.
  - Genera contexto agregado a partir de los resultados recuperados.

  - Entrada:
    - `datasetQA_GT.json`

  - Salida:
    - Actualiza el mismo archivo JSON agregando datos en el campo `retrieved_subgraph`.

  - Uso:
    - `python evaluation/scripts/result_cypher_GT.py evaluation/ground_truth/datasetQA_GT.json`
