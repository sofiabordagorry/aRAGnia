# scripts/run_queries_and_save_subgraphs.py

import json
from pathlib import Path
from typing import Any
import os

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever
# ajustá este import según dónde tengas realmente execute_cypher_query


GT_PATH = (
    Path(__file__).parents[1]
    / "ground_truth"
    / "datasetQA_GT.json"
)


def main() -> None:
    if not GT_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {GT_PATH}")

    with GT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El JSON debe ser una lista de preguntas.")

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

    for item in data:
        question_id = item.get("id")
        query = item.get("cypher_query")

        print(f"Ejecutando pregunta {question_id}...")
        if not query:
            print(f"No hay query")
            item["retrieved_subgraph"] = "La consulta solicitada está fuera del alcance del esquema actual del grafo."
            continue

        try:
            records = retriever.execute_cypher_query(query)
            if records:
                context = retriever._build_aggregation_context(records)
            else:
                context = "No se encontró información relevante en el grafo para responder esta pregunta."


            item["retrieved_subgraph"] = context

        except Exception as e:
            print(f"Error en pregunta {question_id}: {e}")
            item["retrieved_subgraph"] = "No se encontró información relevante en el grafo para responder esta pregunta."

    retriever.close()
    with GT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON actualizado en: {GT_PATH}")


if __name__ == "__main__":
    main()