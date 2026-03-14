import os
from pathlib import Path

from dotenv import load_dotenv

from institutional_graphrag.graph.builder import GraphBuilder, load_graph_json

env_path = Path(__file__).parents[1] / ".env"
load_dotenv(env_path)
JSON_PATH = Path(__file__).parents[2] / "data" / "entities_relations" / "entity_documents.json"


def main():
    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"
    entities, relationships, _ = load_graph_json(JSON_PATH, neo4j_uri, neo4j_user, neo4j_password, enable_non_equal_name_unification=False)

    graph = GraphBuilder(
        neo4j_uri,
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD"),
    )
    graph.clear_graph()
    graph.ingest(entities=entities, relationships=relationships)


if __name__ == "__main__":
    main()
