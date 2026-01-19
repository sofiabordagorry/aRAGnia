"""
Tests para extraction/ie.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult
from institutional_graphrag.graph.schema import Anio, Chunk, Documento, Proyecto, Relationship

# -------------------------
# Helpers
# -------------------------


def write_chunks_file(path: Path, source: str, chunks: list[dict]) -> None:
    data = {"source": source, "chunks": chunks}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_result(res: ExtractionResult) -> dict:
    """
    Comparación semántica: ignora orden.
    """
    return {
        "entities": sorted(
            [e.to_dict() for e in res.entities],
            key=lambda x: (x["label"], x["id"]),
        ),
        "relationships": sorted(
            [r.to_dict() for r in res.relationships],
            key=lambda x: (x["type"], x["source_id"], x["target_id"]),
        ),
        "errors": sorted(
            res.errors,
            key=lambda x: (x.get("type", ""), x.get("message", "")),
        ),
    }


@pytest.fixture()
def extractor(tmp_path: Path) -> EntityExtractor:
    """
    Crea un extractor pero apuntando todo a tmp_path.
    """
    ex = EntityExtractor()
    ex.data_dir = tmp_path
    ex.documents_dir = tmp_path / "corpus"
    ex.chunks_dir = tmp_path / "chunks"
    ex.table_dir = tmp_path / "tables"
    ex.input_dir = tmp_path / "entities_relations"
    return ex


# -------------------------
# extract_documents
# -------------------------


def test_extract_documents_missing_folder(extractor: EntityExtractor):
    extractor.extract_documents()
    assert any(e["type"] == "MissingFolder" for e in extractor.res.errors)
    assert extractor.res.entities == []


def test_extract_documents_invalid_filename_is_error(extractor: EntityExtractor):
    extractor.documents_dir.mkdir(parents=True)
    extractor.table_dir.mkdir(parents=True)  # ✅ requerido

    # nombre inválido (no matchea patrón)
    (extractor.documents_dir / "cualquiercosa.pdf").write_text("x", encoding="utf-8")

    extractor.extract_documents()

    assert any(e["type"] == "Document Invalid" for e in extractor.res.errors)
    assert len([e for e in extractor.res.entities if e.label == "Documento"]) == 0


def test_extract_documents_valid_creates_documento(extractor: EntityExtractor):
    extractor.documents_dir.mkdir(parents=True)
    extractor.table_dir.mkdir(parents=True)  # ✅ requerido

    # nombre válido según tu regex: group_year_docid_kind
    (extractor.documents_dir / "gi_2010_152_informe.pdf").write_text("x", encoding="utf-8")

    extractor.extract_documents()

    docs = [e for e in extractor.res.entities if e.label == "Documento"]
    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, Documento)
    assert d.value["base_name"] == "gi_2010_152_informe"
    assert d.value["is_group"].lower() == "gi"
    assert d.value["year_publisher"] == "2010"
    assert d.value["sub_id"] == "152"
    assert d.value["type"].lower() == "informe"


# -------------------------
# _build_doc_indexes
# -------------------------


def test_build_doc_indexes(extractor: EntityExtractor):
    extractor.res.entities.append(
        Documento(
            id="doc1",
            value={
                "base_name": "gi_2010_152_informe",
                "is_group": "gi",
                "year_publisher": "2010",
                "sub_id": "152",
                "type": "informe",
            },
        )
    )
    extractor._build_doc_indexes()

    assert "doc1" in extractor.doc_by_id
    assert "gi_2010_152_informe" in extractor.doc_by_basename
    assert ("gi", "2010") in extractor.docs_by_group_year
    assert len(extractor.docs_by_group_year[("gi", "2010")]) == 1


# -------------------------
# _read_json
# -------------------------


def test_read_json_invalid_adds_error(extractor: EntityExtractor, tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")

    data = extractor._read_json(p)

    assert data is None
    assert any(e["type"] == "InvalidJson" for e in extractor.res.errors)


# -------------------------
# extract_chunks
# -------------------------


def test_extract_chunks_missing_folder(extractor: EntityExtractor):
    extractor.extract_chunks()
    assert any(e["type"] == "MissingFolder" for e in extractor.res.errors)


def test_extract_chunks_links_document_and_chunks(extractor: EntityExtractor):
    # preparar dirs
    extractor.chunks_dir.mkdir(parents=True)
    extractor.documents_dir.mkdir(parents=True)

    # crear un documento y construir índices
    extractor.res.entities.append(
        Documento(
            id="doc1",
            value={
                "base_name": "gi_2010_152_informe",
                "is_group": "gi",
                "year_publisher": "2010",
                "sub_id": "152",
                "type": "informe",
            },
        )
    )
    extractor._build_doc_indexes()

    # chunks file
    chunks = [
        {
            "chunk_id": "gi_2010_152_informe_chunk0",
            "text": "hola",
            "metadata": {"headings": ["Titulo X"]},
        },
        {"chunk_id": "gi_2010_152_informe_chunk1", "text": "mundo", "metadata": {}},
    ]
    write_chunks_file(
        extractor.chunks_dir / "gi_2010_152_informe_chunks.json",
        source="C:/tmp/gi_2010_152_informe.pdf",
        chunks=chunks,
    )

    extractor.extract_chunks()

    # creó 2 Chunk entities
    chunks_entities = [e for e in extractor.res.entities if e.label == "Chunk"]
    assert len(chunks_entities) == 2
    assert all(isinstance(e, Chunk) for e in chunks_entities)

    # relaciones: PRIMER_CHUNK, SIGUIENTE_CHUNK, DE_DOCUMENTO por cada chunk
    rel_types = [r.type for r in extractor.res.relationships]
    assert "PRIMER_CHUNK" in rel_types
    assert "SIGUIENTE_CHUNK" in rel_types
    assert rel_types.count("DE_DOCUMENTO") == 2


def test_extract_chunks_missing_document_for_chunks_adds_error(extractor: EntityExtractor):
    extractor.chunks_dir.mkdir(parents=True)

    # no hay Documento en índices
    extractor._build_doc_indexes()

    write_chunks_file(
        extractor.chunks_dir / "gi_2010_152_informe_chunks.json",
        source="C:/tmp/gi_2010_152_informe.pdf",
        chunks=[{"chunk_id": "x", "text": "t", "metadata": {}}],
    )

    extractor.extract_chunks()
    assert any(e["type"] == "MissingDocumentForChunks" for e in extractor.res.errors)


# -------------------------
# _expand_rows_by_id_mapping
# -------------------------


def test_expand_rows_by_id_mapping_duplicates_rows(extractor: EntityExtractor):
    df = pd.DataFrame(
        {
            "ID": ["152", "999"],
            "TITULO": ["Proyecto A", "Otro"],
        }
    )

    # sub_id 152 mapea a dos documentos distintos => duplica fila
    id_to_doc = [("152", "docA"), ("152", "docB"), ("999", "docC")]
    out = extractor._expand_rows_by_id_mapping(df, id_to_doc)

    assert len(out) == 3
    assert set(out["ID"].tolist()) == {"docA", "docB", "docC"}


# -------------------------
# _search_title
# -------------------------


def test_search_title_returns_heading_as_best_grade_6_when_title_none(extractor: EntityExtractor):
    extractor.chunks_dir.mkdir(parents=True)

    p = extractor.chunks_dir / "x_chunks.json"
    write_chunks_file(
        p,
        source="C:/tmp/x.pdf",
        chunks=[{"chunk_id": "c0", "text": "abc", "metadata": {"headings": ["Mi Titulo"]}}],
    )

    best = extractor._search_title(p, None)
    assert best is not None
    assert best["best_grade"] == 6
    assert best["candidate_title"] == "Mi Titulo"


def test_search_title_prefers_best_grade_1_over_6(extractor: EntityExtractor):
    extractor.chunks_dir.mkdir(parents=True)

    p = extractor.chunks_dir / "x_chunks.json"
    write_chunks_file(
        p,
        source="C:/tmp/x.pdf",
        chunks=[
            {
                "chunk_id": "c0",
                "text": "Titulo: ABC proyecto",
                "metadata": {"headings": ["Heading malo"]},
            },
        ],
    )

    best = extractor._search_title(p, "ABC")
    assert best is not None
    assert best["best_grade"] == 1
    assert "ABC" in best["candidate_title"]


# -------------------------
# save/load roundtrip
# -------------------------


def test_save_and_load_roundtrip_semantic_equal(extractor: EntityExtractor):
    # armar un res mínimo
    extractor.res.entities.append(Documento(id="doc1", value={"base_name": "a"}))
    extractor.res.entities.append(Proyecto(id="p1", value="Titulo"))
    extractor.res.entities.append(Anio(id="y1", value="2010"))
    extractor.res.relationships.append(
        Relationship(type="INICIO_EN", source_id="p1", target_id="y1", properties={})
    )

    filename = "Entity_documents.json"
    extractor.save_in_file(filename)

    loaded = extractor.load_from_json(filename)
    assert loaded is not None

    assert normalize_result(extractor.res) == normalize_result(loaded)


# -------------------------
# integración chica: run() con fs fake
# -------------------------


def test_run_integration_minimal(tmp_path: Path):
    ex = EntityExtractor()
    ex.data_dir = tmp_path
    ex.documents_dir = tmp_path / "corpus"
    ex.chunks_dir = tmp_path / "chunks"
    ex.table_dir = tmp_path / "tables"
    ex.input_dir = tmp_path / "entities_relations"

    ex.documents_dir.mkdir()
    ex.chunks_dir.mkdir()
    ex.table_dir.mkdir()

    # 1 doc válido
    (ex.documents_dir / "gi_2010_152_informe.pdf").write_text("x", encoding="utf-8")

    # 1 chunks file para ese doc
    write_chunks_file(
        ex.chunks_dir / "gi_2010_152_informe_chunks.json",
        source="C:/tmp/gi_2010_152_informe.pdf",
        chunks=[
            {
                "chunk_id": "gi_2010_152_informe_chunk0",
                "text": "Titulo: Proyecto X",
                "metadata": {"headings": ["Proyecto X"]},
            },
        ],
    )

    # sin parquet => _associate_tables_with_documents devuelve []
    res = ex.run()

    # Debe haber al menos 1 Documento y 1 Chunk
    assert any(e.label == "Documento" for e in res.entities)
    assert any(e.label == "Chunk" for e in res.entities)
