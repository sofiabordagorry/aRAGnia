from psycopg2.extras import RealDictCursor

from .database import get_connection


def insert_query(type, query_text, response, cypher_query=None):
    """
    Insert a query into the queries table and return its ID.

    Args:
        type: 'rag' or 'graph_rag'
        query_text: user query text
        response: generated response
        cypher_query: cypher query executed in Neo4j (optional)
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO queries (type, query_text, cypher_query, response)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """,
        (type, query_text, cypher_query, response),
    )
    query_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return query_id


def insert_chunks(query_id, chunks_list):
    conn = get_connection()
    cur = conn.cursor()

    for chunk in chunks_list:
        cur.execute(
            """
            INSERT INTO chunks (query_id, chunk_id, chunk_text, score)
            VALUES (%s, %s, %s, %s);
            """,
            (
                query_id,
                chunk["id"],
                chunk["text"],
                chunk["score"],
            ),
        )

    conn.commit()
    cur.close()
    conn.close()


def insert_graphrag_chunks(query_id, chunks):
    conn = get_connection()
    cur = conn.cursor()

    inserted_chunk_ids = {}

    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            chunk_text = chunk.get("chunk_text") or chunk.get("text")
        else:
            if len(chunk) < 2:
                raise ValueError(f"Chunk inválido: {chunk}")
            chunk_id = chunk[0]
            chunk_text = chunk[1]

        cur.execute(
            """
            INSERT INTO graphrag_chunks (query_id, chunk_id, chunk_text)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (query_id, chunk_id, chunk_text),
        )
        graphrag_chunk_row_id = cur.fetchone()[0]
        inserted_chunk_ids[chunk_id] = graphrag_chunk_row_id

    conn.commit()
    cur.close()
    conn.close()

    return inserted_chunk_ids


def insert_graphrag_chunk_entities(query_id, chunk_to_entities):
    """
    chunk_to_entities:
        Dict[str, List[tuple[str, str]]]
    """
    conn = get_connection()
    cur = conn.cursor()

    for chunk_id, entities in chunk_to_entities.items():
        for entity_id, entity_label in entities:
            cur.execute(
                """
                INSERT INTO graphrag_chunk_entities (
                    query_id,
                    chunk_id,
                    entity_id,
                    entity_label
                )
                VALUES (%s, %s, %s, %s);
                """,
                (query_id, chunk_id, entity_id, entity_label),
            )

    conn.commit()
    cur.close()
    conn.close()


def get_queries_with_chunks():
    """
    Return all queries with their associated chunks.
    Supports both:
    - RAG -> chunks
    - GraphRAG -> graphrag_chunks + graphrag_chunk_entities
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Traer todas las queries
    cur.execute("""
        SELECT
            id,
            type,
            query_text,
            cypher_query,
            response,
            created_at
        FROM queries
        ORDER BY id;
    """)
    query_rows = cur.fetchall()

    queries = {}
    for row in query_rows:
        qid = row["id"]
        queries[qid] = {
            "id": qid,
            "type": row["type"],
            "query_text": row["query_text"],
            "cypher_query": row["cypher_query"],
            "response": row["response"],
            "created_at": row["created_at"],
            "chunks": [],
        }

    # Traer chunks de RAG
    cur.execute("""
        SELECT
            id,
            query_id,
            chunk_id,
            chunk_text,
            score
        FROM chunks
        ORDER BY query_id, id;
    """)
    rag_rows = cur.fetchall()

    for row in rag_rows:
        qid = row["query_id"]
        if qid in queries:
            queries[qid]["chunks"].append(
                {
                    "id": row["id"],
                    "chunk_id": row["chunk_id"],
                    "chunk_text": row["chunk_text"],
                    "score": row["score"],
                }
            )

    # Traer chunks de GraphRAG
    cur.execute("""
        SELECT
            id,
            query_id,
            chunk_id,
            chunk_text
        FROM graphrag_chunks
        ORDER BY query_id, id;
    """)
    graphrag_chunk_rows = cur.fetchall()

    # Agrupar chunks GraphRAG por query
    graphrag_chunks_by_query = {}
    graphrag_chunks_by_chunk_id = {}

    for row in graphrag_chunk_rows:
        qid = row["query_id"]

        chunk_obj = {
            "id": row["id"],
            "chunk_id": row["chunk_id"],
            "chunk_text": row["chunk_text"],
            "score": None,
            "entities": [],
        }

        graphrag_chunks_by_query.setdefault(qid, []).append(chunk_obj)
        graphrag_chunks_by_chunk_id[(qid, row["chunk_id"])] = chunk_obj

    # Traer entidades de GraphRAG
    cur.execute("""
        SELECT
            query_id,
            chunk_id,
            entity_id,
            entity_label
        FROM graphrag_chunk_entities
        ORDER BY query_id, chunk_id, id;
    """)
    entity_rows = cur.fetchall()

    for row in entity_rows:
        key = (row["query_id"], row["chunk_id"])
        chunk_obj = graphrag_chunks_by_chunk_id.get(key)

        if chunk_obj is not None:
            chunk_obj["entities"].append(
                {"entity_id": row["entity_id"], "entity_label": row["entity_label"]}
            )

    # Agregar chunks GraphRAG a sus queries
    for qid, chunk_list in graphrag_chunks_by_query.items():
        if qid in queries:
            queries[qid]["chunks"].extend(chunk_list)

    cur.close()
    conn.close()

    return list(queries.values())


def delete_query_by_id(query_id: int) -> bool:
    """
    Delete a query by id.
    If foreign keys use ON DELETE CASCADE, associated chunks/entities
    are deleted automatically.
    Returns True if the query existed.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM queries WHERE id = %s RETURNING id;", (query_id,))
    deleted = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()

    return deleted is not None


def delete_all_queries() -> int:
    """
    Delete all queries.
    If foreign keys use ON DELETE CASCADE, associated chunks/entities
    are deleted automatically.
    Returns the number of deleted queries.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM queries RETURNING id;")
    deleted_rows = cur.fetchall()

    conn.commit()
    cur.close()
    conn.close()

    return len(deleted_rows)
