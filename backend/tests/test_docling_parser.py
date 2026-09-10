"""Tests para docling_parse.py"""

import shutil
from pathlib import Path

import pytest

from aragnia.ingest.docling_parser import (
    DocumentAlreadyProcessed,
    parse_corpus,
    parse_single_document,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def test_parse_corpus_directorio_no_existe():
    """Debe fallar si el directorio no existe."""
    with pytest.raises(FileNotFoundError, match="Directorio de corpus no encontrado"):
        parse_corpus(corpus_dir=Path("/directorio_inexistente"))


def test_parse_corpus_sin_archivos(tmp_path):
    """Debe retornar iterable vacío si no hay archivos."""
    result = list(parse_corpus(tmp_path))

    assert result == []


@pytest.mark.skipif(
    not (DATA_DIR / "corpus").exists(),
    reason="Requiere data/corpus",
)
def test_parse_single_document_real_pdf(tmp_path, monkeypatch):
    """
    Prueba el parseo de un solo documento copiado a un directorio temporal.
    """

    # Redirigir el directorio de salida a uno temporal
    # Esto asegura que exists_docling() siempre devuelva False durante el test
    test_output_dir = tmp_path / "test_output"
    test_output_dir.mkdir()
    monkeypatch.setattr(
        "aragnia.ingest.docling_parser.DEFAULT_DOCLING_DIR", test_output_dir
    )

    corpus_dir = DATA_DIR / "corpus"
    pdf_files = sorted(list(corpus_dir.glob("*.pdf")))

    if not pdf_files:
        pytest.skip("No se encontraron archivos PDF en data/corpus para realizar el test.")

    pdf_path = pdf_files[0]
    test_pdf_path = tmp_path / pdf_path.name
    shutil.copy(pdf_path, test_pdf_path)

    doc_dict = parse_single_document(test_pdf_path)

    # Validaciones de estructura basadas en el schema de Docling
    assert isinstance(doc_dict, dict)
    assert doc_dict["schema_name"] == "DoclingDocument"
    assert doc_dict["name"] == test_pdf_path.stem

    # Verificar metadatos de origen
    assert "origin" in doc_dict
    assert doc_dict["origin"]["filename"] == test_pdf_path.name


@pytest.mark.skipif(
    not (DATA_DIR / "corpus").exists(),
    reason="Requiere data/corpus",
)
def test_parse_corpus_subset_integration(tmp_path, monkeypatch):
    """
    Test de integración procesando solo los primeros 2 documentos.
    Usa monkeypatch para evitar conflictos con archivos ya procesados.
    """
    corpus_dir = DATA_DIR / "corpus"
    pdf_files = sorted(list(corpus_dir.glob("*.pdf")))

    if not pdf_files:
        pytest.skip("No hay PDFs en data/corpus")

    # Redirigir el directorio de salida a uno temporal
    # Esto asegura que exists_docling() siempre devuelva False durante el test
    test_output_dir = tmp_path / "test_docling_output"
    test_output_dir.mkdir()
    monkeypatch.setattr(
        "aragnia.ingest.docling_parser.DEFAULT_DOCLING_DIR", test_output_dir
    )

    # Crear el corpus temporal para el test
    test_subset_dir = tmp_path / "subset_corpus"
    test_subset_dir.mkdir()

    # Tomamos los primeros 2 archivos
    files_to_test = pdf_files[:2]
    for pdf in files_to_test:
        shutil.copy(pdf, test_subset_dir / pdf.name)

    results = list(parse_corpus(test_subset_dir, recursive=False, skip_errors=True))

    assert len(results) == len(files_to_test)
    assert results[0]["name"] == files_to_test[0].stem


def test_exists_docling_exception(tmp_path, monkeypatch):
    """Verifica que salte la excepción si el JSON ya existe."""
    from aragnia.ingest.docling_parser import is_already_processed

    test_file = Path("test_doc.pdf")
    # Mockear el directorio de salida para que apunte a un temporal
    monkeypatch.setattr(
        "aragnia.ingest.docling_parser.DEFAULT_DOCLING_DIR", tmp_path
    )

    # Crear el "json_twin"
    json_twin = tmp_path / "test_doc.json"
    json_twin.touch()

    with pytest.raises(DocumentAlreadyProcessed) as excinfo:
        is_already_processed(test_file)

    # Verificamos que el mensaje del error sea el esperado (sin los decoradores de la clase)
    assert "ya había sido convertido" in str(excinfo.value)
