"""Institutional GraphRAG - Pipeline incremental de procesamiento de documentos y GraphRAG."""

__version__ = "0.2.0"

from .queries import insert_query, insert_chunks, delete_all_queries, delete_query_by_id, get_queries_with_chunks
from .database import create_tables, get_connection
from .config import EMBED_MODEL_ID
from .ingest.chunker import (
    chunk_document,
    get_native_chunker,
)
from .ingest.docling_parser import parse_corpus, parse_single_document

__all__ = [
    "EMBED_MODEL_ID",
    "get_native_chunker",
    "chunk_document",
    "parse_corpus",
    "parse_single_document",
]
