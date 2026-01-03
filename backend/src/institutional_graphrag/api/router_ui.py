# routes_ui.py
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import time

router = APIRouter()

# Eliminar al tener funcionalidad 
def now_ms() -> int:
    return int(time.time() * 1000)

MOCK_HISTORY = [
    {
        "id": 101,
        "q": "¿Qué es RAG y en qué se diferencia de un buscador clásico?",
        "answer": "RAG combina recuperación (retrieval) con generación (LLM). Primero busca chunks relevantes y luego genera usando ese contexto.",
        "mode": "rag",
        "ts": now_ms() - 1000 * 60 * 60 * 4,
    },
    {
        "id": 102,
        "q": "Dame un ejemplo de cómo formatear chunks en la respuesta",
        "answer": "Podés devolver `chunks: [{title, score, text}]` y el front los renderiza en tarjetas.",
        "mode": "rag",
        "ts": now_ms() - 1000 * 60 * 60 * 2,
    },
    {
        "id": 103,
        "q": "En GraphRAG, ¿cómo se expande el subgrafo?",
        "answer": "Se parte de entidades/tuplas relevantes y se expande por vecinos (k-hops), luego se filtra por relevancia con la query.",
        "mode": "graphrag",
        "ts": now_ms() - 1000 * 60 * 45,
    },
    {
        "id": 104,
        "q": "¿Qué endpoint usa el front según el modo?",
        "answer": "Si GraphRAG está ON llama a `/graphrag/query`; si está OFF llama a `/rag/query`.",
        "mode": "graphrag",
        "ts": now_ms() - 1000 * 60 * 10,
    },
    {
        "id": 105,
        "q": "¿Cómo manejar errores HTTP en fetch?",
        "answer": "Chequeá `res.ok`, si falla leé `res.text()` y tirá un Error con status + body.",
        "mode": "rag",
        "ts": now_ms() - 1000 * 60 * 2,
    },
]

@router.get("/history")
def get_history(limit: int = 50):
    # Remplazar lo Hardcodeado por llamada a la funcion
    items = sorted(MOCK_HISTORY, key=lambda x: x["ts"], reverse=True)
    limit = max(1, min(limit, 200))
    return items[:limit]

@router.delete("/history")
def clear_history():
    # Remplazar lo Hardcodeado por llamada a la funcion
    MOCK_HISTORY.clear()
    return {"ok": True}

@router.delete("/history/item")
def delete_history_item(id: int = Query(...)):
    # Remplazar lo Hardcodeado por llamada a la funcion
    before = len(MOCK_HISTORY)
    MOCK_HISTORY[:] = [x for x in MOCK_HISTORY if x.get("id") != id]
    after = len(MOCK_HISTORY)
    if before == after:
        return JSONResponse(status_code=404, content={"detail": "Item not found"})

    return {"ok": True}
