from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from institutional_graphrag.api.router_graphrag import QueryRequest, graphrag_query


REPORT_OUTPUT_PATH = Path("../evaluation/question_to_answer/query_batch_report.json")
ANSWERS_OUTPUT_PATH = Path("../evaluation/question_to_answer/query_batch_answers.json")


def _extract_question_text(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("pregunta")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".json":
        raise ValueError("Formato no soportado. Este script solo acepta .json")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        questions_data = payload.get("questions", [])
    elif isinstance(payload, list):
        questions_data = payload
    else:
        raise ValueError(
            "JSON de preguntas inválido: se esperaba una lista o un objeto con 'questions'."
        )

    items: list[dict[str, Any]] = []

    for idx, obj in enumerate(questions_data, 1):
        if not isinstance(obj, dict):
            continue

        question = _extract_question_text(obj)
        if not question:
            continue

        qid = obj.get("id", idx)
        categoria = obj.get("categoria", "")

        items.append(
            {
                "id": qid,
                "categoria": categoria,
                "question": question,
                "raw": obj,
            }
        )

    return items


def _build_retrieved_subgraph(response_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunks": response_data.get("chunks", []),
        "evidence_entities": response_data.get("evidence_entities", {}),
        "chunk_to_entities": response_data.get("chunk_to_entities", {}),
    }


def run_batch(
    *,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    results: list[dict[str, Any]] = []

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        started = time.perf_counter()

        response_data: dict[str, Any] = {}
        error_text = ""
        status_code: int | None = None

        try:
            response = graphrag_query(QueryRequest(query=question))
            response_data = response.model_dump()
            status_code = 200

        except HTTPException as exc:
            status_code = int(exc.status_code)
            error_text = str(exc.detail)

        except Exception as exc:
            status_code = 500
            error_text = str(exc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        ok = bool(response_data) and not error_text

        results.append(
            {
                "index": idx,
                "id": item["id"],
                "categoria": item["categoria"],
                "pregunta": question,
                "cypher_query": response_data.get("cypher_query", ""),
                "retrieved_subgraph": (
                    _build_retrieved_subgraph(response_data)
                    if response_data
                    else ""
                ),
                "answer": response_data.get("answer", ""),
                "ok": ok,
                "status_code": status_code,
                "latency_ms": elapsed_ms,
                "error": error_text,
                "raw_question": item["raw"],
            }
        )

    return results


def write_answers_json(path: Path, results: list[dict[str, Any]]) -> None:
    answers: list[dict[str, Any]] = []

    for row in results:
        answers.append(
            {
                "id": row["id"],
                "categoria": row["categoria"],
                "pregunta": row["pregunta"],
                "cypher_query": row["cypher_query"],
                "retrieved_subgraph": row["retrieved_subgraph"],
                "answer": row["answer"],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(answers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lee preguntas desde un JSON y consulta GraphRAG."
    )

    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Ruta al archivo de preguntas .json.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions_file: Path = args.questions_file

    if not questions_file.exists():
        raise FileNotFoundError(f"No existe el archivo de preguntas: {questions_file}")

    questions = load_questions(questions_file)

    if not questions:
        raise ValueError("No se encontraron preguntas válidas en el archivo.")

    results = run_batch(questions=questions)

    ok_count = sum(1 for row in results if row["ok"])
    failed_count = len(results) - ok_count

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_file": str(questions_file),
            "system": "graphrag",
            "execution_mode": "in-process-router",
            "total_questions": len(results),
            "ok_count": ok_count,
            "failed_count": failed_count,
        },
        "results": results,
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_answers_json(ANSWERS_OUTPUT_PATH, results)

    print(f"Preguntas procesadas: {len(results)}")
    print(f"OK: {ok_count} | Fallidas: {failed_count}")
    print(f"Reporte completo: {REPORT_OUTPUT_PATH}")
    print(f"Respuestas finales: {ANSWERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()