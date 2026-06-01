"""Genera respuestas en lenguaje natural para el ground truth de QA.

Toma cada pregunta en lenguaje natural y el subgrafo recuperado del grafo
(`retrieved_subgraph`) y genera una respuesta en lenguaje natural, reutilizando
la misma lógica de generación de respuestas del pipeline de GraphRAG
(`answer_llm_client`), SIN necesidad de conectarse a Neo4j.

Entrada:
    - evaluation/ground_truth/datasetQA_GT.json
      (campos: id, categoria, pregunta, cypher_query, retrieved_subgraph, answer)

Salida:
    - Actualiza el mismo archivo completando el campo `answer`.
      Este archivo sirve como GT QA (pregunta + answer) y como GT CypherQA
      (pregunta + cypher_query + retrieved_subgraph + answer).

Uso:
    python evaluation/scripts/generate_gt_answers.py
    python evaluation/scripts/generate_gt_answers.py --overwrite
    python evaluation/scripts/generate_gt_answers.py --questions-file otro.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from institutional_graphrag.llm.llm_provider import get_llm_client

import os

GT_PATH = Path(__file__).parents[1] / "ground_truth" / "datasetQA_GT.json"

OUT_OF_SCOPE = "La consulta solicitada está fuera del alcance del esquema actual del grafo."
NO_INFO = "No se encontró información relevante en el grafo para responder esta pregunta."


def _is_count_context(subgrafo: str) -> bool:
    lowered = subgrafo.lower()
    return any(key in lowered for key in ("cantidad", "frecuencia", "total", "count"))


def build_messages(pregunta: str, subgrafo: str) -> list[dict[str, str]]:
    """Construye los mensajes para el LLM a partir de la pregunta y el subgrafo.

    Replica los prompts usados en `GraphRAGRetriever.generate_result` para que las
    respuestas del ground truth sean consistentes con las del sistema en producción.
    """
    if _is_count_context(subgrafo):
        system = (
            "Respondé en español de forma DIRECTA y NUMÉRICA. Si los resultados "
            "muestran un número, respondé ese número exacto. No digas 'no se puede "
            "determinar' si el número está ahí."
        )
        user = (
            f"PREGUNTA: {pregunta}\n\nRESULTADOS DEL GRAFO:\n{subgrafo}\n\n"
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
            f"PREGUNTA: {pregunta}\n\nRESULTADOS DEL GRAFO:\n{subgrafo}\n\n"
            "IMPORTANTE: Los resultados de arriba contienen la respuesta. Usalos "
            "TODOS. No digas que no hay información si los resultados muestran datos."
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(client: Any, pregunta: str, subgrafo: str) -> str:
    """Genera la respuesta en lenguaje natural para una pregunta."""
    subgrafo = (subgrafo or "").strip()

    if not subgrafo or subgrafo == NO_INFO:
        return NO_INFO
    if subgrafo == OUT_OF_SCOPE:
        return OUT_OF_SCOPE

    messages = build_messages(pregunta, subgrafo)
    answer = client.generate(messages=messages, temperature=0.1, max_tokens=2048)
    return answer.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera respuestas en lenguaje natural para el ground truth de QA a "
            "partir de la pregunta y el subgrafo recuperado."
        )
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=GT_PATH,
        help=f"Ruta al archivo JSON del ground truth (default: {GT_PATH}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenera también las respuestas que ya estén completas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path: Path = args.questions_file

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("El JSON debe ser una lista de preguntas.")

    client = get_llm_client(model=os.getenv("OLLAMA_MODEL_ANSWER"))

    generated = 0
    skipped = 0
    for item in data:
        if not isinstance(item, dict):
            continue

        pregunta = (item.get("pregunta") or "").strip()
        if not pregunta:
            continue

        if item.get("answer") and not args.overwrite:
            skipped += 1
            continue

        qid = item.get("id")
        print(f"Generando respuesta para pregunta {qid}...", flush=True)
        item["answer"] = generate_answer(client, pregunta, item.get("retrieved_subgraph", ""))
        generated += 1

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Respuestas generadas: {generated} | Omitidas (ya existían): {skipped}")
    print(f"Archivo actualizado: {path}")


if __name__ == "__main__":
    main()
