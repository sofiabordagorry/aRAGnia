from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from institutional_graphrag.services.startup_check_service import ensure_backends_ready

from .router_graphrag import router as graphrag_router
from .router_rag import router as rag_router
from .router_ui import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_backends_ready()
    yield


app = FastAPI(title="Institutional GraphRAG API", lifespan=lifespan)

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
