from institutional_graphrag.graph.graph import load_graph_json
from institutional_graphrag.graph.graph import GraphBuilder
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()
JSON_PATH = Path(__file__).parents[2] / "data" / "entities_relations" / "entity_documents.json"

def main():
    entities, relationships = load_graph_json(JSON_PATH)
    neo4j_host = os.getenv("NEO4J_HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")

    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    grafo = GraphBuilder(
        neo4j_uri,
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD"),
    )
    grafo.backend.clear_graph()
    grafo.ingest(entities=entities, relationships=relationships)


if __name__ == "__main__":
    main()
