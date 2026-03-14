from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from institutional_graphrag.services.docker_service import manage_docker_services

from .router_graphrag import router as graphrag_router
from .router_rag import router as rag_router
from .router_ui import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    with manage_docker_services():
        yield


app = FastAPI(title="Institutional GraphRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(graphrag_router, prefix="/graphrag", tags=["graphrag"])
app.include_router(ui_router, prefix="/ui", tags=["ui"])


@app.get("/health")
def health():
    return {"status": "ok"}
