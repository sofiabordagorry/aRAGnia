# Scripts de evaluación

Guía rápida de los scripts en esta carpeta.

## Lista de scripts

- `question_to_answer.py`
  - Lee preguntas desde un archivo (`.txt`, `.json`, `.jsonl`) y consulta solo GraphRAG.
  - Reutiliza la lógica del endpoint interno de GraphRAG para evitar duplicar pipeline.
  - Salida:
    - Reporte completo en `evaluation/question_to_answer/query_batch_report.json`.
    - Resumen en `evaluation/question_to_answer/query_batch_answers.jsonl` con `id`, `question`, `answer`, `ok`.
  - Opcional: `--include-debug` para guardar `cypher_query`, `chunks`, `chunk_to_entities`, `entities_extracted` y `relationships_extracted`.
  - Uso:
    - `python evaluation/scripts/question_to_answer.py --questions-file data/results/preguntas.txt`
    - `python evaluation/scripts/question_to_answer.py --questions-file data/results/preguntas.json --include-debug`

- `evaluate_kg.py`
  - Compara un grafo proporcionado contra el ground truth (evaluation/ground_truth/extraction/ground_truth_kg.json).
  - Genera un reporte HTML y json lado a lado de entidades y relaciones.
  - También valida la estructura del grafo contra el esquema definido.
  - Salida:
    - Reporte comparativo en evaluation/results_knowledge_graph/report_complete.html
    - Datos renderizados en evaluation/results_knowledge_graph/report_render_data.json
    - Reporte de validación estructural en evaluation/results_knowledge_graph/report_structure_graph.json
    - Reporte HTML de validación estructural en evaluation/results_knowledge_graph/report_structure_graph.html
  - Uso:
    - `python evaluation/scripts/evaluate_kg.py --graph-file evaluation/llm_extraction/Qwen_Qwen2.5-coder-14B-Instruct/export_graph.json`

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
