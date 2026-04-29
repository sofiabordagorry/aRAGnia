from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever

REPORT_OUTPUT_PATH = (
    Path(__file__).parents[1]
    / "ground_truth"
    / "question_to_cypher"
    / "query_batch_report.json"
)
ANSWERS_OUTPUT_PATH = (
    Path(__file__).parents[1]
    / "ground_truth"
    / "question_to_cypher"
    / "query_batch_answers.jsonl"
)
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {
            " | ".join(map(str, k)) if isinstance(k, tuple) else str(k): to_jsonable(v)
            for k, v in obj.items()
        }

    return str(obj)


def load_questions(path: Path) -> list[dict[str, Any]]:
    log(f"Leyendo archivo: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError("Formato no soportado. Este script solo acepta .json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("questions", []) if isinstance(payload, dict) else payload

    if not isinstance(data, list):
        raise ValueError("JSON inválido: se esperaba una lista o un objeto con 'questions'.")

    questions = []

    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue

        question = item.get("pregunta", "")

        if not isinstance(question, str) or not question.strip():
            continue

        questions.append(
            {
                "id": item.get("id", idx),
                "categoria": item.get("categoria", ""),
                "question": question.strip(),
                "raw": item,
            }
        )

    log(f"Preguntas válidas encontradas: {len(questions)}")
    return questions


def build_retriever() -> GraphRAGRetriever:
    host = os.getenv("HOST", "localhost")
    port = os.getenv("NEO4J_BOLT_PORT", "7687")
    uri = f"bolt://{host}:{port}"

    log(f"Conectando a Neo4j en {uri}")

    return GraphRAGRetriever(
        neo4j_uri=uri,
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        temperature=0.3,
        max_tokens=1024,
    )


def run_batch(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retriever = build_retriever()
    results = []

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        started = time.perf_counter()

        log(f"Procesando {idx}/{len(questions)} | ID: {item['id']}")

        result = {
            "index": idx,
            "id": item["id"],
            "question": question,
            "ok": True,
            "status_code": 200,
            "latency_ms": 0,
            "cypher_query": "",
            "cypher_result": "",
            "chunks": [],
            "answer": "",
            "error": "",
            "rawQA_GT": item["raw"],
        }

        try:
            invalid_result, records, cypher_query = retriever.generate_cypher_query_result(
                user_query=question
            )

            result["cypher_query"] = cypher_query or getattr(invalid_result, "cypher_query", "") or ""
            result["answer"] = getattr(invalid_result, "answer", "") if invalid_result else ""

            chunks, _, _ = retriever.extract_chunks_and_entities_from_results(records)
            result["chunks"] = chunks

            if not chunks and records:
                result["cypher_result"] = retriever._build_aggregation_context(records)

            if not records and not result["cypher_query"]:
                result["ok"] = False
                result["status_code"] = 204
                result["error"] = "No se generó Cypher ni se recuperaron records"

        except Exception as exc:
            result["ok"] = False
            result["status_code"] = 500
            result["error"] = str(exc)

        result["latency_ms"] = int((time.perf_counter() - started) * 1000)

        status = "OK" if result["ok"] else "ERROR"
        log(f"{status} {idx}/{len(questions)} | {result['latency_ms']} ms")

        if result["error"]:
            log(f"Error: {result['error']}")

        results.append(result)

    return results


def write_json(path: Path, data: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(data), ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def write_answers_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(
            {
                "id": row["id"],
                "pregunta": row["question"],
                "cypher_result": row["cypher_result"],
                "ok": row["ok"],
            },
            ensure_ascii=False,
        )
        for row in results
    ]

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lee preguntas desde un JSON y consulta GraphRAG."
    )

    parser.add_argument(
        "questions_file",
        type=Path,
        help="Ruta al archivo JSON de preguntas.",
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

    log("Iniciando evaluación")
    results = run_batch(questions)

    ok_count = sum(row["ok"] for row in results)
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

    log("Guardando archivos de salida")

    write_json(REPORT_OUTPUT_PATH, report, indent=2)
    write_answers_jsonl(ANSWERS_OUTPUT_PATH, results)

    log(f"Preguntas procesadas: {len(results)}")
    log(f"OK: {ok_count} | Fallidas: {failed_count}")
    log(f"Reporte completo: {REPORT_OUTPUT_PATH}")
    log(f"Respuestas resumidas: {ANSWERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()