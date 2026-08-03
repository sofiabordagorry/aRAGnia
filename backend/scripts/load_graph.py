import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from institutional_graphrag.graph.builder import GraphBuilder
from institutional_graphrag.graph.graph_loader import load_graph_json

env_path = Path(__file__).parents[1] / ".env"
load_dotenv(env_path)
DEFAULT_JSON_PATH = (
    Path(__file__).parents[2] / "data" / "entities_relations" / "entity_documents.json"
)


def main(json_path: Path):
    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"
    print(f"{json_path}")
    entities, relationships = load_graph_json(json_path, neo4j_uri, neo4j_user, neo4j_password)

    graph = GraphBuilder(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
    )
    graph.clear_graph()
    graph.ingest(entities=entities, relationships=relationships)
    graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cargar un grafo JSON en Neo4j")
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help=f"JSON de entidades/relaciones a cargar (default: {DEFAULT_JSON_PATH})",
    )

    args = parser.parse_args()

    main(args.json_path)
