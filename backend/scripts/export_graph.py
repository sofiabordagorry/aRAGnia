import os
from pathlib import Path

from dotenv import load_dotenv
from institutional_graphrag.graph.builder import GraphBuilder

env_path = Path(__file__).parents[1] / ".env"
load_dotenv(env_path)
INPUT_PATH = Path(__file__).parents[2] / "data" / "entities_relations" / "entity_documents.json"
OUTPUT_PATH = Path(__file__).parents[2] / "data" / "entities_relations" / "export_graph.json"

def main():
    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"
    graph = GraphBuilder(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
    )
    graph.export_graph(output_json_path=OUTPUT_PATH)




if __name__ == "__main__":
    main()
