"""Institutional GraphRAG - Pipeline incremental de procesamiento de documentos y GraphRAG."""

__version__ = "0.2.0"

from .config import EMBED_MODEL_ID
from .ingest.chunker import (
    chunk_document,
    get_native_chunker,
)
try:
    from .ingest.docling_parser import parse_corpus, parse_single_document
except Exception:
    pass
from .storage.database import create_tables, get_connection
from .storage.queries import (
    delete_all_queries,
    delete_query_by_id,
    get_queries_with_chunks,
    insert_chunks,
    insert_query,
)

__all__ = [
    "EMBED_MODEL_ID",
    "get_native_chunker",
    "chunk_document",
    "parse_corpus",
    "parse_single_document",
    "create_tables",
    "get_connection",
    "delete_all_queries",
    "delete_query_by_id",
    "get_queries_with_chunks",
    "insert_chunks",
    "insert_query",
]
