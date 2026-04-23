from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv
from institutional_graphrag.retrieval.graph_retriever import (
    GraphRAGRetriever,
    GraphRAGResult,
)

load_dotenv()


# =========================================================
# SERIALIZACION
# =========================================================

def to_jsonable(obj: Any) -> Any:
    """
    Convierte objetos no serializables por json.dump en estructuras
    compatibles con JSON.
    """
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
            if isinstance(key, tuple):
                key_str = " | ".join(str(part) for part in key)
            else:
                key_str = str(key)
            jsonable_dict[key_str] = to_jsonable(value)
        return jsonable_dict

    if hasattr(obj, "__dict__"):
        return {
            key: to_jsonable(value)
            for key, value in vars(obj).items()
        }

    return str(obj)


# =========================================================
# FUNCION PRINCIPAL
# =========================================================

def mi_funcion(query: str) -> Dict[str, Any]:
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

    if not query or not query.strip():
        raise ValueError("Query vacía")

    invalid_result: GraphRAGResult | None = None
    cypher_query: str = ""
    records: List[Any] = []

    invalid_result, records, cypher_query = retriever.generate_cypher_query_result(
        user_query=query
    )

    if not records:
        return {
            "answer": invalid_result.answer if invalid_result else "",
            "cypher": cypher_query,
            "chunks": [],
            "evidence_entities": {},
            "chunk_to_entities": {},
        }

    chunks, evidence_entities, chunk_to_entities = (
        retriever.extract_chunks_and_entities_from_results(records)
    )

    return {
        "answer": invalid_result.answer if invalid_result else "",
        "cypher": cypher_query,
        "chunks": to_jsonable(chunks),
        "evidence_entities": to_jsonable(evidence_entities),
        "chunk_to_entities": to_jsonable(chunk_to_entities),
    }


# =========================================================
# PATHS
# =========================================================

INPUT_PATH = (
    Path(__file__).resolve().parent
    / "inputs"
    / "questions.json"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "query_results.json"
)


# =========================================================
# HELPERS
# =========================================================

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_questions_from_json(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "preguntas" not in data:
        raise ValueError("El JSON debe tener una clave 'preguntas'")

    questions = data["preguntas"]

    if not isinstance(questions, list):
        raise ValueError("'preguntas' debe ser una lista")

    for i, q in enumerate(questions):
        if not isinstance(q, str):
            raise ValueError(f"La pregunta en la posición {i} no es un string")

    return questions


def normalize_response(question: str, response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return {
            "question": question,
            "cypher": response.get("cypher"),
            "result": to_jsonable(response),
            "raw_response": to_jsonable(response),
        }

    return {
        "question": question,
        "cypher": None,
        "result": to_jsonable(response),
        "raw_response": to_jsonable(response),
    }


def process_questions(questions: List[str]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for question in questions:
        try:
            response = mi_funcion(question)
            normalized = normalize_response(question, response)
            normalized["status"] = "success"
            results.append(normalized)

        except Exception as e:
            results.append(
                {
                    "question": question,
                    "cypher": None,
                    "result": None,
                    "raw_response": None,
                    "status": "error",
                    "error": str(e),
                }
            )

    return {
        "generated_at": datetime.now().isoformat(),
        "total_questions": len(questions),
        "results": results,
    }


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    questions = load_questions_from_json(INPUT_PATH)
    output = process_questions(questions)
    save_json(output, OUTPUT_PATH)
    print(f"Resultados guardados en: {OUTPUT_PATH}")