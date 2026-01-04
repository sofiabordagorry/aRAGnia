"""Institutional GraphRAG - Pipeline incremental de procesamiento de documentos y GraphRAG."""

__version__ = "0.1.0"

from .ingest.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    SectionBasedChunker,
    chunk_documents,
    chunk_loaded_doc,
)
from .ingest.docling_parse import load_parsed_document, parse_corpus
from .ingest.load_pdf import LoadedDoc, load_corpus, load_document

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
