# Scripts de evaluación

Guía rápida de los scripts en esta carpeta.

## Lista de scripts

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