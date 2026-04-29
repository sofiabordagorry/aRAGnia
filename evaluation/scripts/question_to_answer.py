from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from institutional_graphrag.api.router_graphrag import QueryRequest, graphrag_query


REPORT_OUTPUT_PATH = Path("../ground_truth/question_to_answer/query_batch_report.json")
ANSWERS_OUTPUT_PATH = Path("../ground_truth/question_to_answer/query_batch_answers.jsonl")


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
        raise ValueError("JSON de preguntas invalido: se esperaba lista u objeto con 'questions'.")

    items = []
    for idx, obj in enumerate(questions_data, 1):
        if not isinstance(obj, dict):
            continue

        text = _extract_question_text(obj)
        if not text:
            continue

        qid = obj.get("id") or idx

        items.append(
            {
                "id": qid,
                "question": text,
                "raw": obj,
            }
        )

    return items


def _build_debug_payload(response_data: dict[str, Any]) -> dict[str, Any]:
    chunk_to_entities = response_data.get("chunk_to_entities") or {}

    unique_entities: set[tuple[str, str]] = set()
    relationships: list[dict[str, Any]] = []

    for chunk_id, entities in chunk_to_entities.items():
        if not isinstance(entities, list):
            continue

        for item in entities:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue

            entity_id, entity_label = str(item[0]), str(item[1])
            unique_entities.add((entity_id, entity_label))
            relationships.append(
                {
                    "type": "CHUNK_EVIDENCE_FOR",
                    "source_chunk_id": str(chunk_id),
                    "target_entity_id": entity_id,
                    "target_entity_label": entity_label,
                }
            )

    entities_list = [
        {"id": entity_id, "label": entity_label}
        for entity_id, entity_label in sorted(unique_entities, key=lambda x: (x[1], x[0]))
    ]

    return {
        "cypher_query": response_data.get("cypher_query", ""),
        "retrieved_subgraph": {
            "chunk_to_entities": chunk_to_entities,
            "chunks": response_data.get("chunks", []),
        },
        "chunks": response_data.get("chunks", []),
        "chunk_to_entities": chunk_to_entities,
        "entities_extracted": entities_list,
        "relationships_extracted": relationships,
    }


def run_batch(
    *,
    questions: list[dict[str, Any]],
    include_debug: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        qid = item["id"]
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
        except Exception as exc:  # noqa: BLE001
            status_code = 500
            error_text = str(exc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        answer = response_data.get("answer", "") if response_data else ""
        debug_payload = _build_debug_payload(response_data) if include_debug else {}

        results.append(
            {
                "index": idx,
                "id": qid,
                "question": question,
                "ok": bool(response_data) and not error_text,
                "status_code": status_code,
                "latency_ms": elapsed_ms,
                "answer": answer,
                "debug": debug_payload,
                "error": error_text,
                "raw_question": item["raw"],
            }
        )

    return results


def write_answers_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for row in results:
        out = {
            "id": row["id"],
            "pregunta": row["question"],
            "answer": row["answer"],
            "ok": row["ok"],
        }

        debug = row.get("debug") if isinstance(row.get("debug"), dict) else {}
        if debug:
            out["cypher_query"] = debug.get("cypher_query", "")
            out["retrieved_subgraph"] = debug.get("retrieved_subgraph", "")

        lines.append(
            json.dumps(out, ensure_ascii=False)
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lee preguntas desde un archivo y consulta GraphRAG. "
            "Guarda respuestas y, opcionalmente, metadata para debug."
        )
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Ruta al archivo de preguntas .json.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help=(
            "Incluye metadata de debug por pregunta: cypher_query, chunks, "
            "chunk_to_entities, entidades y relaciones extraidas."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions_file: Path = args.questions_file
    if not questions_file.exists():
        raise FileNotFoundError(f"No existe el archivo de preguntas: {questions_file}")

    questions = load_questions(questions_file)
    if not questions:
        raise ValueError("No se encontraron preguntas validas en el archivo.")

    results = run_batch(
        questions=questions,
        include_debug=args.include_debug,
    )

    ok_count = sum(1 for r in results if r["ok"])
    failed_count = len(results) - ok_count

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_file": str(questions_file),
            "system": "graphrag",
            "execution_mode": "in-process-router",
            "debug_enabled": bool(args.include_debug),
            "total_questions": len(results),
            "ok_count": ok_count,
            "failed_count": failed_count,
        },
        "results": results,
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ANSWERS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_answers_jsonl(ANSWERS_OUTPUT_PATH, results)

    print(f"Preguntas procesadas: {len(results)}")
    print(f"OK: {ok_count} | Fallidas: {failed_count}")
    print(f"Reporte completo: {REPORT_OUTPUT_PATH}")
    print(f"Respuestas resumidas: {ANSWERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
