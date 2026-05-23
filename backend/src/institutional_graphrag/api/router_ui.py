from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import BaseModel
from typing_extensions import TypedDict

from institutional_graphrag.graph.builder import GraphBuilder
from institutional_graphrag.services.ingest_service import IngestService
from institutional_graphrag.storage.queries import (
    delete_all_queries,
    delete_query_by_id,
    get_queries_with_chunks,
)

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
BACKEND_DIR = Path(__file__).resolve().parents[4] / "backend"
ENV_PATH: Optional[Path] = Path(BACKEND_DIR / ".env")

router = APIRouter()
logger = logging.getLogger(__name__)


class ChunkEntityItem(TypedDict):
    entity_id: str
    entity_label: str


class ChunkItem(TypedDict, total=False):
    id: int | str
    chunk_id: str
    chunk_text: str
    chunk_page: int
    score: Optional[float]
    entities: List[ChunkEntityItem]


class HistoryItem(TypedDict, total=False):
    id: int
    type: Literal["rag", "graphrag"]
    query_text: str
    cypher_query: Optional[str]
    response: str
    created_at: datetime
    chunks: List[ChunkItem]


class GraphNodeResponse(BaseModel):
    id: str
    label: str
    display: str
    degree: int


class GraphEdgeResponse(BaseModel):
    source: str
    target: str
    type: str
    properties: dict[str, Any] = {}


class GraphSummaryResponse(BaseModel):
    node_count: int
    edge_count: int


class GraphSnapshotResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    summary: GraphSummaryResponse


class GraphEntityItemResponse(BaseModel):
    id: str
    label: str
    display: str


class GraphEntityCatalogSummaryResponse(BaseModel):
    result_count: int


class GraphEntityCatalogResponse(BaseModel):
    entities: list[GraphEntityItemResponse]
    summary: GraphEntityCatalogSummaryResponse


class DeleteEntityRequest(BaseModel):
    entity_id: str


UPLOAD_JOBS: Dict[str, dict] = {}
UPLOAD_JOBS_LOCK = threading.Lock()


def _set_job(job_key: str, **fields: object) -> None:
    with UPLOAD_JOBS_LOCK:
        job = UPLOAD_JOBS.setdefault(job_key, {})
        job.update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with UPLOAD_JOBS_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _get_latest_job() -> Optional[dict]:
    with UPLOAD_JOBS_LOCK:
        if not UPLOAD_JOBS:
            return None

        latest = max(
            UPLOAD_JOBS.values(),
            key=lambda job: job.get("created_at", ""),
        )
        return dict(latest)


def _build_neo4j_uri() -> str:
    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    return f"bolt://{neo4j_host}:{neo4j_port}"


def _get_graph_reader() -> GraphBuilder:
    return GraphBuilder(
        _build_neo4j_uri(),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password"),
    )


def _run_ingest_job_from_uploads(
    job_id: str,
    folder_data: List[tuple],
    csv_data: Optional[tuple],
) -> None:
    try:
        total_files = len(folder_data)
        _set_job(
            job_id,
            status="running",
            started_at=datetime.utcnow().isoformat(),
            message="Carga de archivos en curso...",
            finished=False,
            progress={"current": 0, "total": total_files},
        )

        def _on_progress(current: int, total: int) -> None:
            _set_job(job_id, progress={"current": current, "total": total})

        service = IngestService(
            data_dir=DATA_DIR,
            env_path=ENV_PATH,
            keep_debug_artifacts=False,
        )

        csv_bytes = csv_data[1] if csv_data else None
        csv_filename = csv_data[0] if csv_data else None

        result = asyncio.run(
            service.ingest_from_uploads(
                folder_files=folder_data,
                csv_bytes=csv_bytes,
                csv_filename=csv_filename,
                progress_callback=_on_progress,
            )
        )

        errors = result.get("errors", []) if isinstance(result, dict) else []
        message = "Carga finalizada con errores." if errors else "Carga completada correctamente."

        _set_job(
            job_id,
            status="success",
            finished=True,
            finished_at=datetime.utcnow().isoformat(),
            message=message,
            result=result,
        )

    except Exception as e:
        _set_job(
            job_id,
            status="error",
            finished=True,
            finished_at=datetime.utcnow().isoformat(),
            message=f"Error durante la carga: {e}",
            error=str(e),
            traceback=traceback.format_exc(),
        )


@router.get("/history", response_model=List[HistoryItem])
def get_history():
    raw_history = get_queries_with_chunks()

    normalized: List[HistoryItem] = []

    for item in raw_history:
        raw_type = item.get("type", "")
        normalized_type: Literal["rag", "graphrag"] = (
            "graphrag" if raw_type in ("graph_rag", "graphrag") else "rag"
        )

        raw_chunks = item.get("chunks", []) or []
        normalized_chunks: List[ChunkItem] = []

        for chunk in raw_chunks:
            normalized_chunk: ChunkItem = {
                "id": chunk.get("id", ""),
                "chunk_text": (
                    chunk.get("chunk_text") or chunk.get("chunk") or chunk.get("text") or ""
                ),
                "chunk_page": (chunk.get("chunk_page") or 0),
                "score": chunk.get("score"),
            }

            if chunk.get("chunk_id") is not None:
                normalized_chunk["chunk_id"] = chunk["chunk_id"]

            raw_entities = chunk.get("entities", []) or []
            normalized_entities: List[ChunkEntityItem] = []

            for entity in raw_entities:
                normalized_entities.append(
                    {
                        "entity_id": entity.get("entity_id", ""),
                        "entity_label": entity.get("entity_label", ""),
                    }
                )

            if normalized_entities:
                normalized_chunk["entities"] = normalized_entities

            normalized_chunks.append(normalized_chunk)

        normalized_item: HistoryItem = {
            "id": item["id"],
            "type": normalized_type,
            "query_text": item.get("query_text", ""),
            "cypher_query": item.get("cypher_query"),
            "response": item.get("response", ""),
            "created_at": item["created_at"],
            "chunks": normalized_chunks,
        }

        normalized.append(normalized_item)

    return normalized


@router.get("/graph", response_model=GraphSnapshotResponse)
def get_graph_snapshot(
    node_limit: int = Query(default=160, ge=1, le=500),
    relationship_limit: int = Query(default=320, ge=1, le=1200),
):
    reader: Optional[GraphBuilder] = None
    try:
        reader = _get_graph_reader()
        return reader.fetch_graph_snapshot(
            node_limit=node_limit,
            relationship_limit=relationship_limit,
        )
    except Exception as e:
        logger.error("No se pudo obtener el snapshot del grafo", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error leyendo grafo: {e}")
    finally:
        if reader is not None:
            reader.close()


@router.get("/graph/entities", response_model=GraphEntityCatalogResponse)
def get_graph_entities(
    search: str = Query(default=""),
    entity_label: str = Query(default=""),
    limit: int = Query(default=30, ge=1, le=100),
):
    reader: Optional[GraphBuilder] = None
    try:
        reader = _get_graph_reader()
        return reader.fetch_entities_catalog(
            search=search,
            entity_label=entity_label or None,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("No se pudo obtener el catálogo de entidades", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error leyendo entidades: {e}")
    finally:
        if reader is not None:
            reader.close()


@router.get("/graph/neighborhood", response_model=GraphSnapshotResponse)
def get_graph_neighborhood(
    entity_id: str = Query(..., min_length=1),
    relationship_limit: int = Query(default=320, ge=1, le=1200),
):
    reader: Optional[GraphBuilder] = None
    try:
        reader = _get_graph_reader()
        return reader.fetch_graph_neighborhood(
            entity_id=entity_id,
            relationship_limit=relationship_limit,
        )
    except Exception as e:
        logger.error("No se pudo obtener la vecindad de la entidad", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error leyendo vecindad: {e}")
    finally:
        if reader is not None:
            reader.close()


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


@router.post("/upload")
async def upload_files(request: Request):
    """
    Recibe los archivos de las carpetas arrastradas desde el frontend.
    Los paths relativos completos vienen en file_paths (campo Form paralelo a files).

    Se parsea el form manualmente para poder subir los límites por defecto de
    starlette (1 MB por parte, 1000 archivos, 1000 campos), que son muy bajos
    para cargas de proyectos con cientos de PDFs.
    """
    latest = _get_latest_job()
    if latest and latest.get("status") == "running":
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "detail": "Ya hay una carga en ejecución.",
                "job_id": latest.get("job_id"),
                "status": latest.get("status"),
                "message": latest.get("message"),
            },
        )

    form = await request.form(
        max_files=100_000,
        max_fields=100_000,
        max_part_size=500 * 1024 * 1024,
    )

    files = form.getlist("files")
    file_paths = form.getlist("file_paths")
    csv_file = form.get("csv_file")

    folder_data: List[tuple] = []
    for i, f in enumerate(files):
        if not hasattr(f, "read"):
            continue
        data = await f.read()
        filename = str(file_paths[i]) if i < len(file_paths) else (getattr(f, "filename", "") or "")
        if not data or not filename:
            continue

        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for zip_info in zf.infolist():
                        if zip_info.is_dir():
                            continue
                        inner_path = zip_info.filename
                        # Ignorar archivos de metadata de macOS y archivos ocultos
                        if inner_path.startswith("__MACOSX/"):
                            continue
                        if any(part.startswith(".") for part in inner_path.split("/")):
                            continue
                        with zf.open(zip_info) as fp:
                            inner_data = fp.read()
                        if inner_data:
                            folder_data.append((inner_path, inner_data))
            except zipfile.BadZipFile:
                logger.error("ZIP inválido: %r", filename)
        else:
            folder_data.append((filename, data))

    csv_data: Optional[tuple] = None
    if isinstance(csv_file, StarletteUploadFile) and csv_file.filename:
        csv_bytes = await csv_file.read()
        if csv_bytes:
            csv_data = (csv_file.filename, csv_bytes)

    if not folder_data and csv_data is None:
        raise HTTPException(
            status_code=400,
            detail="Se requiere al menos una carpeta o un CSV.",
        )

    job_id = str(uuid.uuid4())

    _set_job(
        job_id,
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        started_at=None,
        finished_at=None,
        finished=False,
        message="Carga en cola...",
        result=None,
    )

    thread = threading.Thread(
        target=_run_ingest_job_from_uploads,
        args=(job_id, folder_data, csv_data),
        daemon=True,
    )
    thread.start()

    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "message": "Carga iniciada en segundo plano.",
    }


@router.get("/csv/status")
def get_csv_status():
    """Indica si ya existe un CSV de proyectos en el sistema."""
    has_csv = any("proyectos" in p.name.lower() for p in (DATA_DIR / "tables").glob("*.csv"))
    return {"has_proyectos_csv": has_csv}


@router.get("/upload/status")
def get_upload_status(job_id: Optional[str] = Query(default=None)):
    """
    Devuelve el estado de una carga:
    - si se pasa job_id, devuelve ese
    - si no, devuelve el último job
    """
    job = _get_job(job_id) if job_id else _get_latest_job()

    if job is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "detail": "No hay cargas registradas."},
        )

    return {"ok": True, **job}


@router.delete("/graph/entity")
def delete_graph_entity(payload: DeleteEntityRequest):
    entity_id = payload.entity_id.strip()
    if not entity_id:
        raise HTTPException(status_code=400, detail="entity_id vacío")
    reader = None
    try:
        reader = _get_graph_reader()
        reader.delete_graph_entity(entity_id=entity_id)
        return {"ok": True, "message": "Entidad eliminada correctamente."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("No se pudo eliminar la entidad", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error eliminando entidad: {e}")
    finally:
        if reader is not None:
            reader.close()
