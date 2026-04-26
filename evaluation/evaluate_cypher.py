from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever


REPORT_OUTPUT_PATH = Path("../evaluation/question_to_cypher/query_batch_report.json")
ANSWERS_OUTPUT_PATH = Path("../evaluation/question_to_cypher/query_batch_answers.jsonl")


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, list):
        return [to_jsonable(item) for item in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(item) for item in obj]

    if isinstance(obj, set):
        return [to_jsonable(item) for item in obj]

    if isinstance(obj, dict):
        jsonable_dict: Dict[str, Any] = {}
        for key, value in obj.items():
            key_str = " | ".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
            jsonable_dict[key_str] = to_jsonable(value)
        return jsonable_dict

    return str(obj)


def serialize_records(records: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []

    for record in records:
        row: dict[str, Any] = {}

        try:
            items = record.items()
        except Exception:
            serialized.append({"value": str(record)})
            continue

        for key, value in items:
            try:
                row[str(key)] = to_jsonable(value)
            except Exception:
                row[str(key)] = str(value)

        serialized.append(row)

    return serialized


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
        raise ValueError("JSON inválido: se esperaba una lista o un objeto con 'questions'.")

    items: list[dict[str, Any]] = []

    for idx, obj in enumerate(questions_data, 1):
        if not isinstance(obj, dict):
            continue

        question = _extract_question_text(obj)
        if not question:
            continue

        items.append(
            {
                "id": obj.get("id", idx),
                "categoria": obj.get("categoria", ""),
                "question": question,
                "raw": obj,
            }
        )

    return items


def _build_debug_payload_from_records(
    *,
    cypher_query: str,
    records: list[Any],
    chunks: Any,
    evidence_entities: Any,
    chunk_to_entities: Any,
) -> dict[str, Any]:

    return {
        "cypher_query": cypher_query,
        "retrieved_subgraph": {
            "chunks": to_jsonable(chunks),
            "evidence_entities": to_jsonable(evidence_entities),
            "chunk_to_entities": to_jsonable(chunk_to_entities),
        },
        "chunks": to_jsonable(chunks),
        "evidence_entities": to_jsonable(evidence_entities),
        "chunk_to_entities": to_jsonable(chunk_to_entities),
    }


def run_batch(*, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    retriever = GraphRAGRetriever(
        neo4j_uri=neo4j_uri,
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        temperature=0.3,
        max_tokens=1024,
    )

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        started = time.perf_counter()

        error_text = ""
        status_code = 200
        ok = True

        cypher_query = ""
        records: list[Any] = []
        answer = ""
        chunks: Any = []
        evidence_entities: Any = {}
        chunk_to_entities: Any = {}
        context = ""
        try:
            invalid_result, records, cypher_query = retriever.generate_cypher_query_result(
                user_query=question
            )

            if invalid_result:
                answer = getattr(invalid_result, "answer", "") or ""

            if records:
                chunks, evidence_entities, chunk_to_entities = (
                    retriever.extract_chunks_and_entities_from_results(records)
                )
                context = retriever.build_entity_context(evidence_entities=evidence_entities, chunk_to_entities=chunk_to_entities)

            if not records and not cypher_query:
                ok = False
                status_code = 204
                error_text = "No se generó Cypher ni se recuperaron records"

            if not cypher_query:
                cypher_query = getattr(invalid_result, "cypher_query", "") or ""

        except Exception as exc:
            ok = False
            status_code = 500
            error_text = str(exc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        results.append(
            {
                "index": idx,
                "id": item["id"],
                "question": question,
                "ok": ok,
                "status_code": status_code,
                "latency_ms": elapsed_ms,
                "cypher_query": cypher_query,
                "cypher_result": context,
                "chunks": chunks,
                "answer": answer,
                "error": error_text,
                "raw_question": item["raw"],
            }
        )

    return results


def write_answers_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for row in results:
        out = {
            "id": row["id"],
            "pregunta": row["question"],
            "cypher_result": row["cypher_result"],
            "ok": row["ok"],
        }

        lines.append(json.dumps(to_jsonable(out), ensure_ascii=False))

    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
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
            "execution_mode": "direct-retriever",
            "total_questions": len(results),
            "ok_count": ok_count,
            "failed_count": failed_count,
        },
        "results": results,
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_answers_jsonl(ANSWERS_OUTPUT_PATH, results)

    print(f"Preguntas procesadas: {len(results)}")
    print(f"OK: {ok_count} | Fallidas: {failed_count}")
    print(f"Reporte completo: {REPORT_OUTPUT_PATH}")
    print(f"Respuestas resumidas: {ANSWERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()