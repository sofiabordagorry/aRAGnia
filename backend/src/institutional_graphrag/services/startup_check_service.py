import os
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from institutional_graphrag.graph.builder import GraphBuilder
from institutional_graphrag.graph.graph_loader import load_graph_json
from institutional_graphrag.ingest.persist_embeddings import persist_all_embeddings_and_metadata
from institutional_graphrag.retrieval.vector_store import VectorStore
from institutional_graphrag.storage.database import create_tables

env_path = Path(__file__).parents[3] / ".env"
load_dotenv(env_path)
JSON_PATH = Path(__file__).parents[4] / "data" / "entities_relations" / "export_graph.json"
FALLBACK_JSON_PATH = (
    Path(__file__).parents[4] / "data" / "entities_relations" / "entity_documents.json"
)
if not JSON_PATH.exists():
    JSON_PATH = FALLBACK_JSON_PATH


def wait_for_postgres(max_retries: int = 15, delay: int = 2) -> None:
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", 5432))
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "admin123")
    dbname = os.getenv("POSTGRES_DB", "app_db")

    for _ in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname,
            )
            conn.close()
            return
        except psycopg2.OperationalError:
            time.sleep(delay)

    raise RuntimeError("Postgres no estuvo listo a tiempo")


def wait_for_neo4j(max_retries: int = 15, delay: int = 2) -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    for _ in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                session.run("RETURN 1").single()
            driver.close()
            return
        except Exception:
            time.sleep(delay)

    raise RuntimeError("Neo4j no estuvo listo a tiempo")


def wait_for_qdrant(max_retries: int = 15, delay: int = 2) -> None:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_HTTP_PORT", 6333))

    for _ in range(max_retries):
        try:
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            return
        except Exception:
            time.sleep(delay)

    raise RuntimeError("Qdrant no estuvo listo a tiempo")


def check_postgres_not_empty() -> None:
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", 5432))
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "admin123")
    dbname = os.getenv("POSTGRES_DB", "app_db")

    expected_tables = [
        "queries",
        "chunks",
        "graphrag_chunks",
        "graphrag_chunk_entities",
    ]

    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=dbname,
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public';
        """)
    existing_tables_before = {row[0] for row in cur.fetchall()}

    missing_tables = [t for t in expected_tables if t not in existing_tables_before]

    if missing_tables:
        cur.close()
        conn.close()

        create_tables()

        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=dbname,
        )
        cur = conn.cursor()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
            """)

    cur.close()
    conn.close()

    return


def check_neo4j_not_empty() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        node_result = session.run("MATCH (n) RETURN count(n) AS c").single()
        rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()

        if node_result is None or rel_result is None:
            raise RuntimeError("No se pudieron obtener métricas de Neo4j.")

        node_count = int(node_result["c"])
        rel_count = int(rel_result["c"])

    driver.close()

    neo4j_status = {
        "exists": True,
        "node_count": node_count,
        "rel_count": rel_count,
        "is_empty": node_count == 0 and rel_count == 0,
    }

    print("Estado Neo4j:", neo4j_status)
    if neo4j_status["is_empty"]:
        try:
            entities, relationships = load_graph_json(JSON_PATH, uri, user, password)
            graph = GraphBuilder(
                uri,
                user,
                password,
            )
            graph.clear_graph()
            graph.ingest(entities=entities, relationships=relationships)
            graph.close()
        except Exception as e:
            print("No se pudo crear el grafo:", e)


def check_qdrant_not_empty() -> None:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_HTTP_PORT", 6333))

    client = QdrantClient(host=host, port=port)
    collections_response = client.get_collections()
    collections = collections_response.collections

    collection_info = []
    total_points = 0

    for col in collections:
        info = client.get_collection(col.name)
        points = info.points_count or 0
        total_points += points
        collection_info.append(
            {
                "name": col.name,
                "points_count": points,
            }
        )

    qdrant_status = {
        "exists": True,
        "collections": collection_info,
        "is_empty": total_points == 0,
    }
    print("Estado Qdrant:", qdrant_status)

    if qdrant_status["is_empty"]:
        store = VectorStore(collection_name="demo_collection", embedding_dim=1024)  # E5-large-v2
        persist_all_embeddings_and_metadata(store)


def ensure_backends_ready() -> None:
    print("Esperando Postgres...")
    wait_for_postgres()

    print("Esperando Neo4j...")
    wait_for_neo4j()

    print("Esperando Qdrant...")
    wait_for_qdrant()

    check_postgres_not_empty()
    check_neo4j_not_empty()
    check_qdrant_not_empty()
