"""Genera respuestas en lenguaje natural a partir de un subgrafo recuperado.

Toma un JSON cualquiera cuyos items tengan los campos `question` (pregunta en
lenguaje natural) y `retrieved_subgraph` (subgrafo recuperado del grafo) y, para
cada uno, genera el campo `answer` en lenguaje natural.

Reutiliza la misma lógica de generación de respuestas del pipeline de GraphRAG
(`answer_llm_client`), SIN necesidad de conectarse a Neo4j. El backend se elige
con la variable de entorno LLM_BACKEND (ollama | huggingface).

Entrada:
    - Un archivo JSON donde cada item tenga al menos: question, retrieved_subgraph.

Salida:
    - Agrega/completa el campo `answer` en cada item.
    - Escribe en <nombre_entrada>_answers.json (no modifica el archivo de entrada).

Uso:
    python evaluation/scripts/generate_gt_answers.py preguntas.json
    python evaluation/scripts/generate_gt_answers.py preguntas.json --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from institutional_graphrag.llm.llm_provider import get_llm_client

OUT_OF_SCOPE = "La consulta solicitada está fuera del alcance del esquema actual del grafo."
NO_INFO = "No se encontró ningún elemento que cumpla con los criterios de la consulta."


def _is_count_context(subgraph: str) -> bool:
    lowered = subgraph.lower()
    return any(key in lowered for key in ("cantidad", "frecuencia", "total", "count"))


def build_messages(question: str, subgraph: str) -> list[dict[str, str]]:
    """Construye los mensajes para el LLM a partir de la pregunta y el subgrafo.
    Replica los prompts usados en `GraphRAGRetriever.generate_result`.
    """
    if _is_count_context(subgraph):
        system = (
            "Respondé en español de forma DIRECTA y NUMÉRICA. Si los resultados "
            "muestran un número, respondé ese número exacto. No digas 'no se puede "
            "determinar' si el número está ahí."
        )
        user = (
            f"PREGUNTA: {question}\n\nRESULTADOS DEL GRAFO:\n{subgraph}\n\n"
            "RESPONDE con el número exacto que aparece en los resultados e incluí "
            "todos los valores mostrados."
        )
    else:
        system = (
            "Eres un asistente de investigación académica. Respondé en ESPAÑOL "
            "basándote EXACTAMENTE en los resultados mostrados. Incluí TODOS los "
            "resultados sin omitir ninguno. No inventes información ni agregues "
            "interpretaciones. Comenzá con una frase introductoria y luego la lista "
            "completa de resultados."
        )
        user = (
            f"PREGUNTA: {question}\n\nRESULTADOS DEL GRAFO:\n{subgraph}\n\n"
            "IMPORTANTE: Los resultados de arriba contienen la respuesta. Usalos "
            "TODOS. No digas que no hay información si los resultados muestran datos."
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _resolve_answer_model() -> str | None:
    """Modelo de respuestas según el backend (LLM_BACKEND=ollama|huggingface)."""
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "huggingface":
        return os.getenv("HF_MODEL")
    return os.getenv("OLLAMA_MODEL_ANSWER")


def generate_answer(client: Any, question: str, subgraph: str) -> str:
    """Genera la respuesta en lenguaje natural para una pregunta."""
    subgraph = (subgraph or "").strip()

    if not subgraph or subgraph == NO_INFO:
        return NO_INFO
    if subgraph == OUT_OF_SCOPE:
        return OUT_OF_SCOPE

    messages = build_messages(question, subgraph)
    answer = client.generate(messages=messages, temperature=0.1, max_tokens=2048)
    return answer.strip()


def load_items(path: Path) -> list[dict[str, Any]]:
    """Carga la lista de items desde una lista JSON o un objeto con 'questions'."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        items = payload.get("questions", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("JSON inválido: se esperaba una lista o un objeto con 'questions'.")

    if not isinstance(items, list):
        raise ValueError("JSON inválido: 'questions' debe ser una lista.")

    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera respuestas en lenguaje natural a partir de la pregunta y el "
            "subgrafo recuperado, completando el campo 'answer' de cada item."
        )
    )
    parser.add_argument(
        "questions_file",
        type=Path,
        help="Ruta al archivo JSON con items que tengan 'question' y 'retrieved_subgraph'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenera también las respuestas que ya estén completas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path: Path = args.questions_file
    out_path: Path = in_path.with_name(f"{in_path.stem}_answers{in_path.suffix}")

    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {in_path}")

    items = load_items(in_path)
    client = get_llm_client(model=_resolve_answer_model())

    generated = 0
    skipped = 0
    for item in items:
        if not isinstance(item, dict):
            continue

        question = (item.get("pregunta") or "").strip()
        if not question:
            continue

        if item.get("answer") and not args.overwrite:
            skipped += 1
            continue

        qid = item.get("id", "?")
        print(f"Generando respuesta para pregunta {qid}...", flush=True)
        item["answer"] = generate_answer(client, question, item.get("retrieved_subgraph", ""))
        generated += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Respuestas generadas: {generated} | Omitidas (ya existían): {skipped}")
    print(f"Archivo escrito: {out_path}")


if __name__ == "__main__":
    main()
