from fastapi import APIRouter
from pydantic import BaseModel
from institutional_graphrag.rag.generate import Rag


router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[dict] = []


@router.post("/query", response_model=QueryResponse)
def rag_query(payload: QueryRequest):
    # llm provider = [groq, ollama, local]
    rag = Rag(top_k=3, llm_provider="ollama")
    result = rag.generate(payload.query)
    chunks = [
        {
            "id": ch.semantic_id,
            "text": ch.text,
            "score": ch.score,
        }
        for ch in result.contexts
    ]
    return QueryResponse(answer=result.answer, chunks=chunks)
