from fastapi import APIRouter
from pydantic import BaseModel

from institutional_graphrag.rag.generate import RAG
from institutional_graphrag.storage.queries import insert_chunks, insert_query

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[dict] = []


@router.post("/query", response_model=QueryResponse)
def rag_query(payload: QueryRequest):
    # llm provider = [groq, ollama, local]
    rag = RAG(top_k=3, llm_provider="ollama")
    query = payload.query
    result = rag.generate(query)
    chunks = [
        {
            "id": ch.semantic_id,
            "text": ch.text,
            "score": ch.score,
        }
        for ch in result.contexts
    ]

    query_id = insert_query("rag", query, result.answer)
    insert_chunks(query_id, chunks)
    return QueryResponse(answer=result.answer, chunks=chunks)
