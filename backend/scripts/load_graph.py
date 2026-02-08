import os
from pathlib import Path

from dotenv import load_dotenv

from institutional_graphrag.graph.builder import GraphBuilder, load_graph_json

load_dotenv()
JSON_PATH = Path(__file__).parents[2] / "data" / "entities_relations" / "entity_documents.json"


def main():
    entities, relationships, _ = load_graph_json(JSON_PATH)
    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")

    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    grafo = GraphBuilder(
        neo4j_uri,
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD"),
    )
    grafo.ingest(entities=entities, relationships=relationships)


if __name__ == "__main__":
    main()
