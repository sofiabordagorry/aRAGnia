"""
Tests para el módulo de chunking basado en secciones.
"""

import json
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker

# Definir la ruta a los datos procesados por Docling
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class TestNativeDoclingChunker:
    """Tests para la nueva implementación basada en Docling Native HybridChunker."""

    @pytest.fixture
    def sample_doc(self):
        """Extrae un documento real de data/docling para los tests."""
        docling_dir = DATA_DIR / "docling"

        # Buscar archivos JSON que no sean dummy.json (que está vacío)
        json_files = sorted([f for f in docling_dir.glob("*.json") if f.stem != "dummy"])

        if not json_files:
            pytest.skip(
                f"No se encontraron archivos JSON con contenido en {docling_dir}. Ejecuta el parsing primero."
            )

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
        chunker = get_native_chunker()
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

    def test_heading_extraction_logic(self, sample_doc):
        """Verifica que los encabezados se extraigan correctamente en la metadata."""

        chunker = get_native_chunker()
        chunks = chunk_document(doc=sample_doc, chunker=chunker)

        chunk_with_headings = next((c for c in chunks if c["metadata"]["headings"]), None)

        if chunk_with_headings:
            # Los encabezados deben extraerse correctamente como una lista en la metadata
            headings = chunk_with_headings["metadata"]["headings"]
            assert isinstance(headings, list)
            assert len(headings) > 0
            
            # Verificamos que el texto exista independientemente de los encabezados
            assert isinstance(chunk_with_headings["text"], str)
            assert len(chunk_with_headings["text"]) > 0
        else:
            pytest.skip("El documento seleccionado no tiene encabezados para probar la jerarquía.")

    def test_verify_reconstruction_fidelity(self, sample_doc):
        """
        Verifica que la reconstrucción manual del texto (headings + raw_text) 
        no pierda información ni duplique texto en comparación con contextualize() de Docling.
        """
        chunker = get_native_chunker()
        
        # Necesitamos iterar sobre los chunks originales de Docling para usar contextualize()
        chunk_iter = chunker.chunk(dl_doc=sample_doc)
        
        def smart_reconstruct(raw_text: str, headings: list) -> str:
            clean_text = raw_text.strip()
            if not headings:
                return clean_text
            
            last_heading = headings[-1].strip()
            
            # Evitar duplicación si el chunk ES el encabezado
            if clean_text == last_heading:
                return "\n".join(headings)
            else:
                # Usar \n\n para separar la jerarquía del cuerpo del texto
                return "\n".join(headings) + "\n" + clean_text

        diffs_found = 0

        for chunk in chunk_iter:
            # 1. Obtener el string contextualizado nativo
            native_context = chunker.contextualize(chunk)
            
            # 2. Obtener los componentes crudos
            raw_text = chunk.text
            headings = chunk.meta.headings if chunk.meta and chunk.meta.headings else []
            
            # 3. Aplicar nuestra lógica
            my_reconstructed_text = smart_reconstruct(raw_text, headings)
            
            # 4. Comparar (ignorando espacios en blanco al inicio/final)
            if native_context.strip() != my_reconstructed_text.strip():
                diffs_found += 1
                print("\n--- DIFF DETECTED in Chunk ---")
                print(f"RAW TEXT: {raw_text[:50]}...")
                print(f"HEADINGS: {headings}")
                print("--- NATIVE (Docling) ---")
                print(repr(native_context.strip()))
                print("--- MINE (Reconstructed) ---")
                print(repr(my_reconstructed_text.strip()))
                print("-" * 40)
        
        # El test fallará si encuentra diferencias, permitiéndote ver los prints en la consola
        assert diffs_found == 0, f"Se encontraron {diffs_found} diferencias entre Docling nativo y la reconstrucción."
