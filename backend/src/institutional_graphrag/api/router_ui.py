# routes_ui.py
from typing import Literal, Optional, List
from typing_extensions import TypedDict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from institutional_graphrag.storage.queries import get_queries_with_chunks, delete_all_queries, delete_query_by_id

router = APIRouter()

class ChunkItem(TypedDict):
    id: int
    chunk: str
    score: Optional[float]  # Puede ser None si no hay score

class HistoryItem(TypedDict):
    id: int
    type: Literal["rag", "graphrag"]
    query_text: str
    response: str
    created_at: str  
    chunks: List[ChunkItem]  



@router.get("/history", response_model=List[HistoryItem])
def get_history():
    history = get_queries_with_chunks()
    return history

@router.delete("/history")
def clear_history():
    delete_all_queries()
    return {"ok": True}


@router.delete("/history/item")
def delete_history_item(id: int = Query(...)):
    is_deleted = delete_query_by_id(id)
    if not is_deleted:
        return JSONResponse(status_code=404, content={"detail": "Item not found"})

    return {"ok": True}