"""
Tests para extraction/ie.py (alineados al ie.py actual)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import institutional_graphrag.extraction.ie as ie_mod
from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult
from institutional_graphrag.extraction.llm_extractor import (
    LLMExtractionResult,
    ResearcherMention,
    TopicMention,
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
    """Comparación semántica: ignora orden."""
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
def extractor(tmp_path: Path, monkeypatch) -> EntityExtractor:
    """
    Crea un extractor pero apuntando todo a tmp_path.
    IMPORTANTÍSIMO: patch de DATA_DIR del módulo ie.py porque run() lo usa.
    """
    monkeypatch.setattr(ie_mod, "DATA_DIR", tmp_path)

    ex = EntityExtractor()
    ex.data_dir = tmp_path
    ex.documents_dir = tmp_path / "corpus"
    ex.chunks_dir = tmp_path / "chunks"
    ex.table_dir = tmp_path / "tables"
    ex.input_dir = tmp_path / "entities_relations"

    # las carpetas que el extractor espera (cuando corresponda)
    return ex


# -------------------------
# _extract_documents
# -------------------------


def test_extract_documents_missing_folder(extractor: EntityExtractor):
    extractor._extract_documents()
    assert any(e["type"] == "MissingFolder" for e in extractor.res.errors)
    assert extractor.res.entities == []


def test_extract_documents_invalid_filename_is_error(extractor: EntityExtractor):
    extractor.documents_dir.mkdir(parents=True)
    extractor.table_dir.mkdir(parents=True)  # requerido por ensure_dir

    # nombre inválido (no matchea patrón)
    (extractor.documents_dir / "cualquiercosa.pdf").write_text("x", encoding="utf-8")

    extractor._extract_documents()

    # El type exacto depende del StaticExtractor; chequeamos robusto:
    assert extractor.res.errors, "Debe registrar al menos 1 error"
    assert any(
        "cualquiercosa" in (e.get("message", "") or "")
        or "pattern" in (e.get("message", "") or "").lower()
        for e in extractor.res.errors
    )
    assert len([e for e in extractor.res.entities if e.label == "Documento"]) == 0


def test_extract_documents_valid_creates_documento(extractor: EntityExtractor):
    extractor.documents_dir.mkdir(parents=True)
    extractor.table_dir.mkdir(parents=True)

    (extractor.documents_dir / "gi_2010_152_informe.pdf").write_text("x", encoding="utf-8")

    extractor._extract_documents()

    docs = [e for e in extractor.res.entities if e.label == "Documento"]
    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, Documento)
    assert d.value["base_name"] == "gi_2010_152_informe"
    assert str(d.value["is_group"]).lower() == "gi"
    assert d.value["year_publisher"] == "2010"
    assert d.value["sub_id"] == "152"
    assert str(d.value["type"]).lower() == "informe"


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
# _extract_chunks
# -------------------------


def test_extract_chunks_missing_folder(extractor: EntityExtractor):
    extractor._extract_chunks()
    assert any(e["type"] == "MissingFolder" for e in extractor.res.errors)


def test_extract_chunks_links_document_and_chunks(extractor: EntityExtractor):
    extractor.chunks_dir.mkdir(parents=True)
    extractor.documents_dir.mkdir(parents=True)
    extractor.table_dir.mkdir(parents=True)

    # crear un documento e indexarlo
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

    extractor._extract_chunks()

    chunks_entities = [e for e in extractor.res.entities if e.label == "Chunk"]
    assert len(chunks_entities) == 2
    assert all(isinstance(e, Chunk) for e in chunks_entities)

    rel_types = [r.type for r in extractor.res.relationships]
    assert "PRIMER_CHUNK" in rel_types
    assert "SIGUIENTE_CHUNK" in rel_types
    assert rel_types.count("DE_DOCUMENTO") == 2


def test_extract_chunks_missing_document_for_chunks_adds_error(extractor: EntityExtractor):
    extractor.chunks_dir.mkdir(parents=True)

    extractor._build_doc_indexes()  # doc_by_basename vacío

    write_chunks_file(
        extractor.chunks_dir / "gi_2010_152_informe_chunks.json",
        source="C:/tmp/gi_2010_152_informe.pdf",
        chunks=[{"chunk_id": "x", "text": "t", "metadata": {}}],
    )

    extractor._extract_chunks()
    # Esto lo genera StaticExtractor; chequeo por tipo si coincide, sino por mensaje:
    assert extractor.res.errors
    assert any(
        e.get("type") == "MissingDocumentForChunks"
        or "MissingDocumentForChunks" in (e.get("message", "") or "")
        for e in extractor.res.errors
    )


# -------------------------
# save/load roundtrip
# -------------------------


def test_save_and_load_roundtrip_semantic_equal(extractor: EntityExtractor):
    extractor.input_dir.mkdir(parents=True, exist_ok=True)

    extractor.res.entities.append(Documento(id="doc1", value={"base_name": "a"}))
    extractor.res.entities.append(Proyecto(id="p1", value="Titulo"))
    extractor.res.entities.append(Anio(id="y1", value="2010"))

    # add_relationship ahora recibe LISTA
    extractor.add_relationship(
        [Relationship(type="INICIO_EN", source_id="p1", target_id="y1", properties={})]
    )

    filename = "entity_documents.json"
    extractor.save_in_file(filename)

    loaded = extractor.load_from_json(filename)
    assert loaded is not None

    assert normalize_result(extractor.res) == normalize_result(loaded)


# -------------------------
# integración chica: run() con fs real (sin LLM)
# -------------------------


def test_run_integration_minimal(tmp_path: Path, monkeypatch):
    # patch global DATA_DIR del módulo
    monkeypatch.setattr(ie_mod, "DATA_DIR", tmp_path)

    ex = EntityExtractor()
    ex.data_dir = tmp_path
    ex.documents_dir = tmp_path / "corpus"
    ex.chunks_dir = tmp_path / "chunks"
    ex.table_dir = tmp_path / "tables"
    ex.input_dir = tmp_path / "entities_relations"

    ex.documents_dir.mkdir()
    ex.chunks_dir.mkdir()
    ex.table_dir.mkdir()
    ex.input_dir.mkdir()

    (ex.documents_dir / "gi_2010_152_informe.pdf").write_text("x", encoding="utf-8")

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

    # correr SIN LLM para no depender de ollama en tests
    res = ex.run(llm_researchers=False, llm_topics=False)

    assert any(e.label == "Documento" for e in res.entities)
    assert any(e.label == "Chunk" for e in res.entities)


# -------------------------
# LLM + agregación por proyecto (con mock)
# -------------------------


def test_llm_researchers_and_topics_and_project_aggregation(
    extractor: EntityExtractor, tmp_path: Path, monkeypatch
):
    """
    Testea la funcionalidad actual:
    - Investigador se deduplica POR PROYECTO (distintos proyectos => entidades distintas)
    - Tópicos se deduplican GLOBALMENTE (mismo topic => 1 entidad)
    - _aggregate_topics_for_project crea TIENE_TOPICO con mention_count por proyecto
    """
    # ---- setup dirs ----
    extractor.documents_dir.mkdir(parents=True, exist_ok=True)
    extractor.chunks_dir.mkdir(parents=True, exist_ok=True)
    extractor.table_dir.mkdir(parents=True, exist_ok=True)
    extractor.input_dir.mkdir(parents=True, exist_ok=True)

    # ---- docs + proyectos + relaciones ES_DESCRITO_POR ----
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
    extractor.add_entities([doc1, doc2, doc3])
    extractor._build_doc_indexes()

    proyecto1 = Proyecto(id="gi_2010_152", value="Proyecto 152")
    proyecto2 = Proyecto(id="gi_2010_391", value="Proyecto 391")
    extractor.add_entities([proyecto1, proyecto2])

    extractor.add_relationship(
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

    # ---- chunks + DE_DOCUMENTO ----
    # proyecto1: 2 docs => 2 chunks que mencionan Machine Learning
    write_chunks_file(
        extractor.chunks_dir / "gi_2010_152_informe_chunks.json",
        source="C:/tmp/gi_2010_152_informe.pdf",
        chunks=[
            {
                "chunk_id": "gi_2010_152_informe_chunk0",
                "text": "Juan Pérez investiga machine learning.",
                "metadata": {},
            }
        ],
    )
    write_chunks_file(
        extractor.chunks_dir / "gi_2010_152_propuesta_chunks.json",
        source="C:/tmp/gi_2010_152_propuesta.pdf",
        chunks=[
            {
                "chunk_id": "gi_2010_152_propuesta_chunk0",
                "text": "Juan Pérez investiga machine learning.",
                "metadata": {},
            }
        ],
    )
    # proyecto2: 1 doc => 1 chunk
    write_chunks_file(
        extractor.chunks_dir / "gi_2010_391_informe_chunks.json",
        source="C:/tmp/gi_2010_391_informe.pdf",
        chunks=[
            {
                "chunk_id": "gi_2010_391_informe_chunk0",
                "text": "Juan Pérez investiga machine learning.",
                "metadata": {},
            }
        ],
    )

    # cargar chunks en el grafo (Chunk + DE_DOCUMENTO, etc.)
    extractor._extract_chunks()

    # ---- registry: evitar que marque cache en el path real ----
    # forzamos a que nunca esté cacheado y que mark_success no escriba
    monkeypatch.setattr(extractor, "already_run", lambda *args, **kwargs: False)
    monkeypatch.setattr(extractor, "mark_success", lambda *args, **kwargs: None)

    # ---- mock LLM extractor ----
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

    # ---- mock factories para controlar IDs ----
    # investigador: id distinto por proyecto (porque tu comentario actual dice eso)
    def fake_create_entities_and_relationships_from_llm_extraction(
        llm_result, project_id, existing_ids
    ):
        new_entities = []
        new_relationships = []
        for m in llm_result.researchers:
            name = m.name
            chunk_id = m.chunk_id
            evidence = m.evidence or ""
            inv_id = f"juan_perez__{project_id}"  # 👈 distinto por proyecto

            if inv_id not in existing_ids:
                new_entities.append(Investigador(id=inv_id, value={"name": name, "source": "llm"}))

            new_relationships.append(
                Relationship(
                    type="PARTICIPO_EN", source_id=inv_id, target_id=project_id, properties={}
                )
            )
            new_relationships.append(
                Relationship(
                    type="EVIDENCIA_DE",
                    source_id=chunk_id,
                    target_id=inv_id,
                    properties={"evidence_text": evidence},
                )
            )
        return new_entities, new_relationships

    monkeypatch.setattr(
        ie_mod,
        "create_entities_and_relationships_from_llm_extraction",
        fake_create_entities_and_relationships_from_llm_extraction,
    )

    # tópico: id global (compartido)
    def fake_create_topics_from_llm_extraction(llm_result, existing_topic_ids):
        new_entities = []
        new_relationships = []
        for m in llm_result.topics:
            chunk_id = m.chunk_id
            evidence = m.evidence or ""
            topic_id = "machine_learning"
            if topic_id not in existing_topic_ids:
                new_entities.append(
                    Topico(id=topic_id, value={"value": "Machine Learning", "source": "llm"})
                )
            new_relationships.append(
                Relationship(
                    type="EVIDENCIA_DE",
                    source_id=chunk_id,
                    target_id=topic_id,
                    properties={"evidence_text": evidence},
                )
            )
        return new_entities, new_relationships

    monkeypatch.setattr(
        ie_mod,
        "create_topics_from_llm_extraction",
        fake_create_topics_from_llm_extraction,
    )

    # ---- ejecutar pipeline LLM actual ----
    extractor._extract_with_llm(llm_researchers=True, llm_topics=True)

    # ---- asserts investigadores ----
    investigadores = [e for e in extractor.res.entities if e.label == "Investigador"]
    assert len(investigadores) == 2, "Distintos proyectos => 2 entidades Investigador"
    inv_ids = {inv.id for inv in investigadores}
    assert inv_ids == {"juan_perez__gi_2010_152", "juan_perez__gi_2010_391"}

    participo_rels = [r for r in extractor.res.relationships if r.type == "PARTICIPO_EN"]
    assert len(participo_rels) == 2
    assert {r.target_id for r in participo_rels} == {"gi_2010_152", "gi_2010_391"}

    # ---- asserts tópicos ----
    topicos = [e for e in extractor.res.entities if e.label == "Topico"]
    assert len(topicos) == 1, "Topic global => 1 entidad Topico"

    evidencia_top = [
        r
        for r in extractor.res.relationships
        if r.type == "EVIDENCIA_DE" and r.target_id == "machine_learning"
    ]
    assert len(evidencia_top) == 3, "3 chunks => 3 evidencias del tópico"

    # ---- asserts agregación proyecto->tópico ----
    tiene_topico_rels = [
        r
        for r in extractor.res.relationships
        if r.type == "TIENE_TOPICO" and r.target_id == "machine_learning"
    ]
    assert len(tiene_topico_rels) == 2, "2 proyectos => 2 relaciones TIENE_TOPICO"

    rel_p1 = next(r for r in tiene_topico_rels if r.source_id == "gi_2010_152")
    rel_p2 = next(r for r in tiene_topico_rels if r.source_id == "gi_2010_391")
    assert rel_p1.properties.get("mention_count") == 2, "Proyecto 152 tiene 2 menciones (2 chunks)"
    assert rel_p2.properties.get("mention_count") == 1, "Proyecto 391 tiene 1 mención (1 chunk)"
