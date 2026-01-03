"""
Tests para el módulo de chunking basado en secciones.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from institutional_graphrag.ingest.chunker import (
    SectionBasedChunker,
    chunk_documents,
    chunk_loaded_doc,
)
from institutional_graphrag.ingest.load_pdf import load_document


class TestSectionBasedChunker:
    """Tests para SectionBasedChunker."""

    def test_chunker_initialization(self):
        """Verifica que el chunker se inicializa correctamente."""
        chunker = SectionBasedChunker(max_chunk_size=1000, chunk_overlap=100)
        assert chunker.max_chunk_size == 1000
        assert chunker.chunk_overlap == 100
        assert chunker.text_splitter is not None

    def test_chunk_small_sections(self):
        """Verifica chunking de secciones pequeñas sin split."""
        docs = [
            Document(
                page_content="Contenido sección 1",
                metadata={
                    "dl_meta": {"headings": ["Sección 1"]},
                    "source": "test.pdf",
                },
            ),
            Document(
                page_content="Contenido sección 2",
                metadata={
                    "dl_meta": {"headings": ["Sección 2"]},
                    "source": "test.pdf",
                },
            ),
        ]

        chunker = SectionBasedChunker(max_chunk_size=500)
        chunks = chunker.chunk_documents(docs)

        # Debe haber 2 chunks, uno por sección
        assert len(chunks) == 2
        assert chunks[0].metadata.get("section_heading") == "Sección 1"
        assert chunks[1].metadata.get("section_heading") == "Sección 2"
        assert chunks[0].metadata.get("chunk_method") == "section"

    def test_chunk_same_section_grouped(self):
        """Verifica que documentos con el mismo heading se agrupan."""
        docs = [
            Document(
                page_content="Parte 1 de la introducción",
                metadata={
                    "dl_meta": {"headings": ["Introducción"]},
                    "source": "test.pdf",
                },
            ),
            Document(
                page_content="Parte 2 de la introducción",
                metadata={
                    "dl_meta": {"headings": ["Introducción"]},
                    "source": "test.pdf",
                },
            ),
        ]

        chunker = SectionBasedChunker(max_chunk_size=500)
        chunks = chunker.chunk_documents(docs)

        # Debe haber 1 chunk que agrupa ambos documentos
        assert len(chunks) == 1
        assert "Parte 1" in chunks[0].page_content
        assert "Parte 2" in chunks[0].page_content
        assert chunks[0].metadata.get("section_heading") == "Introducción"
        assert chunks[0].metadata.get("section_doc_count") == 2

    def test_chunk_large_section_split(self):
        """Verifica que secciones grandes se dividen con fallback."""
        # Crear contenido grande
        large_content = "A" * 2500

        docs = [
            Document(
                page_content=large_content,
                metadata={
                    "dl_meta": {"headings": ["Sección Grande"]},
                    "source": "test.pdf",
                },
            ),
        ]

        chunker = SectionBasedChunker(max_chunk_size=1000, chunk_overlap=100)
        chunks = chunker.chunk_documents(docs)

        # Debe dividirse en múltiples chunks
        assert len(chunks) > 1
        assert all(c.metadata.get("chunk_method") == "section_split" for c in chunks)
        assert all(c.metadata.get("section_heading") == "Sección Grande" for c in chunks)
        
        # Verificar que los chunks tienen índices
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("section_chunk_index") == i
            assert chunk.metadata.get("section_chunk_total") == len(chunks)

    def test_chunk_no_heading(self):
        """Verifica manejo de documentos sin heading."""
        docs = [
            Document(
                page_content="Contenido sin heading",
                metadata={"source": "test.pdf"},
            ),
        ]

        chunker = SectionBasedChunker(max_chunk_size=500)
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 1
        assert chunks[0].metadata.get("section_heading") is None

    def test_chunk_mixed_sections(self):
        """Verifica chunking con secciones mixtas (con y sin heading)."""
        docs = [
            Document(
                page_content="Contenido con heading",
                metadata={
                    "dl_meta": {"headings": ["Sección A"]},
                    "source": "test.pdf",
                },
            ),
            Document(
                page_content="Contenido sin heading",
                metadata={"source": "test.pdf"},
            ),
        ]

        chunker = SectionBasedChunker(max_chunk_size=500)
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 2
        assert chunks[0].metadata.get("section_heading") == "Sección A"
        assert chunks[1].metadata.get("section_heading") is None


class TestChunkFunctions:
    """Tests para funciones de conveniencia."""

    def test_chunk_documents_function(self):
        """Verifica la función chunk_documents."""
        docs = [
            Document(
                page_content="Test content",
                metadata={
                    "dl_meta": {"headings": ["Test"]},
                    "source": "test.pdf",
                },
            ),
        ]

        chunks = chunk_documents(docs, max_chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].metadata.get("section_heading") == "Test"