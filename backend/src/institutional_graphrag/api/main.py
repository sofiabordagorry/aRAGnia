from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from institutional_graphrag.retrieval.fewshot_store import FewShotStore
from institutional_graphrag.services.startup_check_service import ensure_backends_ready

from .router_graphrag import router as graphrag_router
from .router_rag import router as rag_router
from .router_ui import router as ui_router

CORPUS_PATH = Path(__file__).parents[4] / "data" / "corpus"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_backends_ready()

    # Inicializar el embedder y store solo una vez por sesión
    app.state.fewshot_store = FewShotStore()
    yield

    # Borrar de la memoria el embedder + store
    app.state.fewshot_store.close()


app = FastAPI(title="Institutional GraphRAG API", lifespan=lifespan)

app.mount("/pdfs", StaticFiles(directory=CORPUS_PATH), name="pdfs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/rag", tags=["rag"], include_in_schema=False)
app.include_router(graphrag_router, prefix="/graphrag", tags=["graphrag"])
app.include_router(ui_router, prefix="/ui", tags=["ui"])


@app.get("/health")
def health():
    return {"status": "ok"}
