from fastapi import APIRouter
from pydantic import BaseModel

# Ej: from institutional_graphrag.ingest.graphrag import graphrag_answer

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[dict] = []


@router.post("/query", response_model=QueryResponse)
def graphrag_query(payload: QueryRequest):
    # Remplazar lo Hardcodeado por llamada a la funcion
    fake_chunks = [{"id": 1, "text": "chunk ejemplo", "score": 0.92}]
    fake_answer = f"Respuesta GraphRAG para: {payload.query}"
    return QueryResponse(answer=fake_answer, chunks=fake_chunks)
