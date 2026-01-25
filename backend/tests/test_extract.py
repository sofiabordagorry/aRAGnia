"""
Tests para extraction/ie.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult
from institutional_graphrag.extraction.llm_extractor import (
    LLMEntityExtractor,
    LLMExtractionResult,
    ResearcherMention,
    TopicMention,
    create_entities_and_relationships_from_llm_extraction,
    create_topics_from_llm_extraction,
)
from institutional_graphrag.graph.schema import (
    Anio,
    Chunk,
    Documento,
    Investigador,
    Proyecto,
    Relationship,
    Topico,
)

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


# -------------------------
# extract_researchers_llm y extract_topics_llm (con mock)
# -------------------------


def test_extract_researchers_and_topics_llm_integration(
    extractor: EntityExtractor, tmp_path: Path, monkeypatch
):
    """Test deduplicación: mismo investigador en mismo proyecto = 1 entidad, en proyectos diferentes = 2 entidades."""
    extractor.chunks_dir.mkdir(parents=True)
    extractor.documents_dir.mkdir(parents=True)

    # Crear dos proyectos diferentes
    # Proyecto 1: dos documentos
    doc1 = Documento(
        id="doc1",
        value={
            "base_name": "gi_2010_152_informe",
            "is_group": "gi",
            "year_publisher": "2010",
            "sub_id": "152",
            "type": "informe",
        },
    )
    doc2 = Documento(
        id="doc2",
        value={
            "base_name": "gi_2010_152_propuesta",
            "is_group": "gi",
            "year_publisher": "2010",
            "sub_id": "152",
            "type": "propuesta",
        },
    )
    # Proyecto 2: un documento
    doc3 = Documento(
        id="doc3",
        value={
            "base_name": "gi_2010_391_informe",
            "is_group": "gi",
            "year_publisher": "2010",
            "sub_id": "391",
            "type": "informe",
        },
    )
    extractor.res.entities.extend([doc1, doc2, doc3])
    extractor._build_doc_indexes()

    proyecto1 = Proyecto(id="gi_2010_152", value="Proyecto 152")
    proyecto2 = Proyecto(id="gi_2010_391", value="Proyecto 391")
    extractor.res.entities.extend([proyecto1, proyecto2])
    extractor.res.relationships.extend(
        [
            Relationship(
                type="ES_DESCRITO_POR", source_id="gi_2010_152", target_id="doc1", properties={}
            ),
            Relationship(
                type="ES_DESCRITO_POR", source_id="gi_2010_152", target_id="doc2", properties={}
            ),
            Relationship(
                type="ES_DESCRITO_POR", source_id="gi_2010_391", target_id="doc3", properties={}
            ),
        ]
    )

    # Los 3 documentos mencionan "Juan Pérez"
    for base_name in ["gi_2010_152_informe", "gi_2010_152_propuesta", "gi_2010_391_informe"]:
        chunks = [
            {
                "chunk_id": f"{base_name}_chunk0",
                "text": "Juan Pérez investiga machine learning.",
                "metadata": {},
            },
        ]
        write_chunks_file(
            extractor.chunks_dir / f"{base_name}_chunks.json",
            source=f"C:/tmp/{base_name}.pdf",
            chunks=chunks,
        )

        # Crear entidades Chunk y relaciones DE_DOCUMENTO necesarias para agregación
        doc_id = {
            "gi_2010_152_informe": "doc1",
            "gi_2010_152_propuesta": "doc2",
            "gi_2010_391_informe": "doc3",
        }[base_name]
        chunk_id = f"{base_name}_chunk0"
        extractor.res.entities.append(Chunk(id=chunk_id, value={}))
        extractor.res.relationships.append(
            Relationship(type="DE_DOCUMENTO", source_id=chunk_id, target_id=doc_id, properties={})
        )

    # Mock para investigadores
    def mock_extract_researchers(chunks_list, max_chunks=None):
        chunk_id = chunks_list[0].get("chunk_id")
        return LLMExtractionResult(
            researchers=[
                ResearcherMention(
                    name="Juan Pérez", evidence="Juan Pérez investiga", chunk_id=chunk_id
                )
            ],
            topics=[],
            errors=[],
        )

    # Mock para tópicos
    def mock_extract_topics(chunks_list, max_chunks=None):
        chunk_id = chunks_list[0].get("chunk_id")
        return LLMExtractionResult(
            researchers=[],
            topics=[
                TopicMention(
                    topic="Machine Learning", evidence="machine learning", chunk_id=chunk_id
                )
            ],
            errors=[],
        )

    monkeypatch.setattr(
        "institutional_graphrag.extraction.ie.LLMEntityExtractor.extract_researchers_from_chunks",
        lambda self, chunks, max_chunks=None: mock_extract_researchers(chunks, max_chunks),
    )
    monkeypatch.setattr(
        "institutional_graphrag.extraction.ie.LLMEntityExtractor.extract_topics_from_chunks",
        lambda self, chunks, max_chunks=None: mock_extract_topics(chunks, max_chunks),
    )

    # Ejecutar ambas extracciones
    extractor.extract_researchers_llm()
    extractor.extract_topics_llm()

    # Verificar deduplicación:
    # - Investigadores: mismo nombre en diferentes proyectos = entidades distintas
    # - Mismo proyecto (doc1 y doc2): 1 investigador
    # - Proyecto diferente (doc3): otro investigador
    # Total: 2 investigadores
    investigadores = [e for e in extractor.res.entities if e.label == "Investigador"]
    assert len(investigadores) == 2, "Debe haber 2 investigadores (uno por proyecto)"

    # - Tópicos: mismo tópico en diferentes proyectos = misma entidad
    # "Machine Learning" es siempre el mismo concepto
    # Total: 1 tópico compartido entre ambos proyectos
    topicos = [e for e in extractor.res.entities if e.label == "Topico"]
    assert len(topicos) == 1, "Debe haber 1 tópico (compartido entre proyectos)"

    # Verificar que cada investigador participa en SU proyecto
    participo_rels = [r for r in extractor.res.relationships if r.type == "PARTICIPO_EN"]
    assert len(participo_rels) == 2
    project_ids_from_rels = {r.target_id for r in participo_rels}
    assert project_ids_from_rels == {"gi_2010_152", "gi_2010_391"}

    # Verificar relaciones TIENE_TOPICO:
    # Solo proyecto->topico (basado en agregación de chunks)
    tiene_topico_rels = [r for r in extractor.res.relationships if r.type == "TIENE_TOPICO"]
    assert (
        len(tiene_topico_rels) == 2
    ), f"Debe haber 2 relaciones TIENE_TOPICO (proyecto->topico), encontradas: {len(tiene_topico_rels)}"
    assert all(r.source_id in {"gi_2010_152", "gi_2010_391"} for r in tiene_topico_rels)

    # Verificar mention_count en propiedades
    proj1_rel = next(r for r in tiene_topico_rels if r.source_id == "gi_2010_152")
    proj2_rel = next(r for r in tiene_topico_rels if r.source_id == "gi_2010_391")
    assert (
        proj1_rel.properties.get("mention_count") == 2
    ), "Proyecto 1 tiene 2 menciones del tópico (2 chunks)"
    assert (
        proj2_rel.properties.get("mention_count") == 1
    ), "Proyecto 2 tiene 1 mención del tópico (1 chunk)"

    # Verificar evidencias: 3 chunks mencionan investigadores (2 en proyecto1, 1 en proyecto2)
    evidencia_inv = [
        r
        for r in extractor.res.relationships
        if r.type == "EVIDENCIA_DE" and r.target_id in {inv.id for inv in investigadores}
    ]
    assert len(evidencia_inv) == 3

    # Verificar evidencias de tópicos: 3 chunks mencionan el mismo tópico
    evidencia_top = [
        r
        for r in extractor.res.relationships
        if r.type == "EVIDENCIA_DE" and r.target_id in {top.id for top in topicos}
    ]
    assert len(evidencia_top) == 3
