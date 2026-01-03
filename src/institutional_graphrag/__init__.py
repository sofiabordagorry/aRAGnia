"""Institutional GraphRAG - Pipeline incremental de procesamiento de documentos y GraphRAG."""

__version__ = "0.1.0"

from .ingest.chunker import (
    SectionBasedChunker,
    chunk_documents,
    chunk_loaded_doc,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from .ingest.load_pdf import load_document, load_corpus, LoadedDoc
from .ingest.docling_parse import parse_corpus, load_parsed_document

__all__ = [
    "SectionBasedChunker",
    "chunk_documents",
    "chunk_loaded_doc",
    "DEFAULT_MAX_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "load_document",
    "load_corpus",
    "LoadedDoc",
    "parse_corpus",
    "load_parsed_document",
]
