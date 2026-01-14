"""Institutional GraphRAG - Pipeline incremental de procesamiento de documentos y GraphRAG."""

__version__ = "0.2.0"

from .config import EMBED_MODEL_ID

from .ingest.chunker import (
    get_native_chunker,
    chunk_document,
)
from .ingest.docling_parser import parse_corpus, parse_single_document

__all__ = [
    "EMBED_MODEL_ID",
    "get_native_chunker",
    "chunk_document",
    "parse_corpus",
    "parse_single_document",
]
