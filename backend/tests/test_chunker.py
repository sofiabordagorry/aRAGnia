"""
Tests para el módulo de chunking basado en secciones.
"""
import pytest
import json

from pathlib import Path

from docling_core.types.doc import DoclingDocument
from institutional_graphrag.ingest.chunker import (
    chunk_document,
    get_native_chunker
)

# Definir la ruta a los datos procesados por Docling
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

class TestNativeDoclingChunker:
    """Tests para la nueva implementación basada en Docling Native HybridChunker."""

    @pytest.fixture
    def sample_doc(self):
        """Extrae un documento real de data/docling para los tests."""
        docling_dir = DATA_DIR / "docling"
        
        # Buscar archivos JSON en el directorio de docling
        json_files = sorted(list(docling_dir.glob("*.json")))
        
        if not json_files:
            pytest.skip(f"No se encontraron archivos JSON en {docling_dir}. Ejecuta el parsing primero.")
        
        # Tomar el primer documento disponible para el test
        with open(json_files[0], "r", encoding="utf-8") as f:
            doc_dict = json.load(f)
            
        # Reconstruir el objeto DoclingDocument
        return DoclingDocument.model_validate(doc_dict)

    def test_get_native_chunker_initialization(self):
        """Verifica que el chunker se inicializa con el tokenizer correcto."""
        chunker = get_native_chunker()
        assert chunker.max_tokens == 512
        assert hasattr(chunker, "tokenizer")

    def test_chunk_document_structure(self, sample_doc):
        """Verifica que el output tenga la estructura de diccionario requerida."""
        chunker = get_native_chunker(max_tokens=512)
        chunks = chunk_document(doc=sample_doc, chunker=chunker)

        assert len(chunks) > 0
        first_chunk = chunks[0]

        # Verificar campos
        assert "chunk_id" in first_chunk
        assert first_chunk["chunk_id"] == f"{sample_doc.name}#chunk0"
        assert "text" in first_chunk
        assert isinstance(first_chunk["metadata"], dict)

    def test_metadata_extraction(self, sample_doc):
        """Verifica la extracción de procedencia (páginas) y tipos de elementos."""
        chunker = get_native_chunker()
        chunks = chunk_document(doc=sample_doc, chunker=chunker)
        meta = chunks[0]["metadata"]

        # Verifica los campos que agregaste manualmente
        assert meta["parent_doc"] == sample_doc.name
        assert len(meta["page_numbers"]) > 0
        assert "element_type" in meta
        assert "token_count" in meta
        assert meta["token_count"] > 0

    def test_contextualization_logic(self, sample_doc):
        """Verifica que contextualize() incluya jerarquía en el texto del chunk."""

        chunker = get_native_chunker()
        chunks = chunk_document(doc=sample_doc, chunker=chunker)

        chunk_with_headings = next((c for c in chunks if c["metadata"]["headings"]), None)

        if chunk_with_headings:
            # El texto debe contener al menos uno de los encabezados padres
            heading = chunk_with_headings["metadata"]["headings"][0]
            assert heading in chunk_with_headings["text"]
        else:
            pytest.skip("El documento seleccionado no tiene encabezados para probar la jerarquía.")
