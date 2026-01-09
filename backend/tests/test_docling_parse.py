"""Tests para docling_parse.py"""

import json
import shutil
from pathlib import Path

import pytest

from institutional_graphrag.ingest.docling_parse import (
    load_parsed_document,
    parse_corpus,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def test_parse_corpus_directorio_no_existe():
    """Debe fallar si el directorio no existe."""
    with pytest.raises(FileNotFoundError, match="Directorio de corpus no encontrado"):
        parse_corpus(Path("/directorio_inexistente"))


def test_parse_corpus_sin_archivos(tmp_path):
    """Debe retornar lista vacía si no hay archivos."""
    output_dir = tmp_path / "output"
    result = parse_corpus(tmp_path, output_dir)

    assert result == []


def test_load_parsed_document_no_existe():
    """Debe fallar si el archivo JSON no existe."""
    with pytest.raises(FileNotFoundError, match="Archivo JSON no encontrado"):
        load_parsed_document(Path("/archivo_inexistente.json"))


def test_load_parsed_document_exitoso(tmp_path):
    """Debe cargar un documento JSON válido."""
    data = {
        "source": "test.pdf",
        "num_documents": 1,
        "documents": [{"page_content": "test", "metadata": {}}],
    }

    archivo = tmp_path / "test.json"
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = load_parsed_document(archivo)

    assert result == data
    assert result["num_documents"] == 1


@pytest.mark.skipif(
    not (DATA_DIR / "corpus").exists(),
    reason="Requiere data/corpus",
)
def test_parse_corpus_real(tmp_path):
    """Test de integración procesando un solo PDF."""
    corpus_dir = DATA_DIR / Path("corpus")

    # Buscar un PDF en el corpus
    pdf_files = list(corpus_dir.glob("**/*.pdf"))
    if not pdf_files:
        pytest.skip("No hay PDFs en data/corpus")

    # Crear directorio temporal y copiar solo un PDF
    test_corpus = tmp_path / "corpus"
    test_corpus.mkdir()
    test_pdf = test_corpus / pdf_files[0].name
    shutil.copy(pdf_files[0], test_pdf)

    # Directorio de output temporal
    output_dir = tmp_path / "output"

    # Procesar solo ese PDF
    output_files = parse_corpus(test_corpus, output_dir, skip_errors=True)

    # Verificar
    assert len(output_files) == 1
    output_path = output_files[0]

    assert output_path.exists()
    assert output_path.suffix == ".json"

    # Verificar contenido
    data = load_parsed_document(output_path)
    assert "source" in data
    assert "num_documents" in data
    assert "documents" in data
    assert len(data["documents"]) > 0

    # tmp_path se limpia automáticamente al finalizar el test
