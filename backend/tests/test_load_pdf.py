"""Tests simplificados para carga de documentos."""

from pathlib import Path

import pytest

from institutional_graphrag.ingest.load_pdf import (
    DocumentLoadError,
    LoadedDoc,
    iter_document_paths,
    load_document,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def test_iter_document_paths_encuentra_archivos(tmp_path):
    """Debe encontrar archivos de formatos soportados."""
    pdf = tmp_path / "doc.pdf"
    txt = tmp_path / "doc.txt"
    no_soportado = tmp_path / "doc.xyz"

    pdf.write_text("pdf")
    txt.write_text("txt")
    no_soportado.write_text("xyz")

    result = list(iter_document_paths(tmp_path))

    assert len(result) == 2
    assert pdf in result
    assert txt in result


def test_load_document_validaciones():
    """Debe validar archivo y formato."""
    with pytest.raises(FileNotFoundError, match="Documento no encontrado"):
        load_document(Path("/archivo_inexistente.pdf"))


def test_load_document_formato_invalido(tmp_path):
    """Debe rechazar formatos no soportados."""
    archivo = tmp_path / "doc.xyz"
    archivo.write_text("contenido")

    with pytest.raises(DocumentLoadError, match="Formato de archivo no soportado"):
        load_document(archivo)


def test_load_document_pdf_real():
    """Debe cargar un PDF real del corpus."""
    corpus_dir = DATA_DIR / Path("corpus")

    if not corpus_dir.exists():
        pytest.skip("Directorio data/corpus no existe")

    pdfs = list(corpus_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No hay PDFs en data/corpus")

    result = load_document(pdfs[0])

    assert isinstance(result, LoadedDoc)
    assert result.documents is not None
    assert len(result.documents) > 0
    assert all(hasattr(doc, "page_content") for doc in result.documents)
