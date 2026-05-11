import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever
from institutional_graphrag.storage.queries import (
    insert_graphrag_chunk_entities,
    insert_graphrag_chunks,
    insert_query,
)

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    chunk_to_entities: dict
    chunks: list[dict] = []
    cypher_query: str = ""


@router.post("/query", response_model=QueryResponse)
def graphrag_query(payload: QueryRequest, request: Request):
    """Endpoint GraphRAG que usa solo grafo (sin embeddings)."""
    logger.info(f"[GraphRAG] Request recibido: '{payload.query}'")
    try:
        neo4j_host = os.getenv("HOST", "localhost")
        neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
        neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

        retriever = GraphRAGRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            temperature=0.3,
            max_tokens=1024,
            fewshot_store=request.app.state.fewshot_store
        )

        result = retriever.query(payload.query)

        retriever.close()
        logger.info("Conexión a Neo4j cerrada")
        chunks = [
            {"id": chunk.chunk_id, "text": chunk.text, "page": chunk.page}
            for chunk in result.chunks
        ]

        logger.info(
            f"[GraphRAG] Respuesta generada: {len(chunks)} chunks, {len(result.answer)} caracteres"
        )
        query_id = insert_query("graphrag", payload.query, result.answer, result.cypher_query)
        insert_graphrag_chunks(query_id, chunks)
        insert_graphrag_chunk_entities(query_id, result.chunk_to_entities)
        return QueryResponse(
            answer=result.answer,
            chunk_to_entities=result.chunk_to_entities,
            chunks=chunks,
            cypher_query=result.cypher_query,
        )

    except ValueError as e:
        logger.error(f"[GraphRAG] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[GraphRAG] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en GraphRAG: {str(e)}")
