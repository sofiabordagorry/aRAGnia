"""
Tests para extraction/ie.py (alineados al ie.py actual)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import institutional_graphrag.extraction.ie as ie_mod
from institutional_graphrag.extraction.bert_extractor import TopicMention
from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult
from institutional_graphrag.graph.schema import (
    Anio,
    Chunk,
    Documento,
    Investigador,
    Proyecto,
    Relationship,
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

    # El type exacto depende del RuleBasedExtractor; chequeamos robusto:
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

    (extractor.documents_dir / "proy_2014_148_informe.pdf").write_text("x", encoding="utf-8")

    extractor._extract_documents()

    docs = [e for e in extractor.res.entities if e.label == "Documento"]
    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, Documento)
    assert d.value["nombre_base"] == "proy_2014_148_informe"
    assert str(d.value["es_grupo"]).lower() == "proy"
    assert d.value["anio_publicacion"] == "2014"
    assert d.value["sub_id"] == "148"
    assert str(d.value["tipo"]).lower() == "informe"


# -------------------------
# _build_doc_indexes
# -------------------------


def test_build_doc_indexes(extractor: EntityExtractor):
    extractor.res.entities.append(
        Documento(
            id="doc1",
            value={
                "nombre_base": "proy_2014_148_informe",
                "es_grupo": "proy",
                "anio_publicacion": "2014",
                "sub_id": "148",
                "tipo": "informe",
            },
        )
    )
    extractor._build_doc_indexes()

    assert "doc1" in extractor.doc_by_id
    assert "proy_2014_148_informe" in extractor.doc_by_basename
    assert "proy_2014_148" in extractor.id_projects
    assert len(extractor.id_projects) == 1


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
                "nombre_base": "proy_2014_148_informe",
                "es_grupo": "proy",
                "anio_publicacion": "2014",
                "sub_id": "148",
                "tipo": "informe",
            },
        )
    )
    extractor._build_doc_indexes()

    chunks = [
        {
            "chunk_id": "proy_2014_148_informe_chunk0",
            "text": "hola",
            "metadata": {"headings": ["Titulo X"]},
        },
        {"chunk_id": "proy_2014_148_informe_chunk1", "text": "mundo", "metadata": {}},
    ]
    write_chunks_file(
        extractor.chunks_dir / "proy_2014_148_informe_chunks.json",
        source="C:/tmp/proy_2014_148_informe.pdf",
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
        extractor.chunks_dir / "proy_2014_148_informe_chunks.json",
        source="C:/tmp/proy_2014_148_informe.pdf",
        chunks=[{"chunk_id": "x", "text": "t", "metadata": {}}],
    )

    extractor._extract_chunks()
    # Esto lo genera RuleBasedExtractor; chequeo por tipo si coincide, sino por mensaje:
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

    extractor.res.entities.append(Documento(id="doc1", value={"nombre_base": "a"}))
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

    (ex.documents_dir / "proy_2014_148_informe.pdf").write_text("x", encoding="utf-8")

    write_chunks_file(
        ex.chunks_dir / "proy_2014_148_informe_chunks.json",
        source="C:/tmp/proy_2014_148_informe.pdf",
        chunks=[
            {
                "chunk_id": "proy_2014_148_informe_chunk0",
                "text": "Titulo: Proyecto X",
                "metadata": {"headings": ["Proyecto X"]},
            },
        ],
    )

    res = ex.run()

    assert any(e.label == "Documento" for e in res.entities)
    assert any(e.label == "Chunk" for e in res.entities)


# -------------------------
# LLM + agregación por proyecto (con mock)
# -------------------------


def test_bert_topics_and_project_aggregation():
    bert_extractor = ie_mod.BertTopicExtractor(
        threshold=0.5,
        confidence_logit_threshold=0.5,
        coverage_logit_threshold=0.5,
    )
    bert_extractor._en_to_es_topic = {"Machine Learning": "machine learning"}

    mentions = [
        TopicMention(
            topic="Machine Learning",
            evidence="count=1 score=0.9500",
            chunk_id="chunk1",
            logit=0.8,
        ),
        TopicMention(
            topic="Machine Learning",
            evidence="count=1 score=0.9000",
            chunk_id="chunk2",
            logit=0.7,
        ),
    ]

    relationships = bert_extractor.aggregate_topics_for_project(
        project_id="proy_2014_148",
        project_bert_results=mentions,
        total_chunks=2,
    )

    tiene_topico_rels = [r for r in relationships if r.type == "TIENE_TOPICO"]
    extraido_de_rels = [r for r in relationships if r.type == "EXTRAIDO_DE"]

    assert len(tiene_topico_rels) == 1
    assert len(extraido_de_rels) == 2

    rel = tiene_topico_rels[0]
    assert rel.source_id == "proy_2014_148"
    assert rel.target_id == "machine_learning"
    assert rel.properties["mention_count"] == 2
    assert rel.properties["coverage_logit"] == 1.0
    assert rel.properties["confidence_logit"] == 0.75
    assert isinstance(rel.properties["coverage_logit"], float)
    assert isinstance(rel.properties["confidence_logit"], float)


def test_aggregate_topics_for_project_below_threshold_returns_no_relationships():
    bert_extractor = ie_mod.BertTopicExtractor(
        threshold=0.5,
        confidence_logit_threshold=0.5,
        coverage_logit_threshold=0.5,
        min_topics=0,
    )
    bert_extractor._en_to_es_topic = {"Machine Learning": "machine learning"}

    mentions = [
        TopicMention(
            topic="Machine Learning",
            evidence="count=1 score=0.3000",
            chunk_id="chunk1",
            logit=0.3,
        )
    ]

    relationships = bert_extractor.aggregate_topics_for_project(
        project_id="proy_2014_148",
        project_bert_results=mentions,
        total_chunks=1,
    )

    assert relationships == []


def test_aggregate_topics_for_project_fallback_by_coverage():
    bert_extractor = ie_mod.BertTopicExtractor(
        threshold=0.5,
        confidence_logit_threshold=0.5,
        coverage_logit_threshold=0.5,
        min_topics=1,
    )
    bert_extractor._en_to_es_topic = {
        "Rare Topic": "rare topic",
        "Frequent Topic": "frequent topic",
    }

    mentions = [
        TopicMention(topic="Rare Topic", evidence="count=1 score=0.3000", chunk_id="chunk1", logit=0.4),
        TopicMention(topic="Frequent Topic", evidence="count=3 score=0.3000", chunk_id="chunk2", logit=0.3),
    ]

    relationships = bert_extractor.aggregate_topics_for_project(
        project_id="proy_2014_148",
        project_bert_results=mentions,
        total_chunks=4,
    )

    tiene_topico_rels = [r for r in relationships if r.type == "TIENE_TOPICO"]
    assert len(tiene_topico_rels) == 1
    rel = tiene_topico_rels[0]
    assert rel.target_id == "frequent_topic"
    assert rel.properties["fallback"] is True


# -------------------------
# Tabular extractor: propiedad calidad en PARTICIPO_EN
# -------------------------


def test_tabular_extractor_calidad_property(tmp_path: Path):
    """La propiedad 'calidad' se guarda correctamente en la relación PARTICIPO_EN."""
    import csv

    from institutional_graphrag.extraction.tabular_extractor import TabularExtractor

    csv_path = tmp_path / "equipos_test.csv"
    rows = [
        {
            "row_id": "0",
            "pais_documento": "UY",
            "tipo_documento": "CI",
            "documento": "11111",
            "nombres": "ANA",
            "apellidos": "GARCIA",
            "sexo": "F",
            "calidad": "Responsable",
            "id_formulario": "1",
            "anio": "2018",
            "programa": "I+D",
            "titulo": "Proyecto A",
            "id_archivo": "proy_2018_1",
        },
        {
            "row_id": "1",
            "pais_documento": "UY",
            "tipo_documento": "CI",
            "documento": "22222",
            "nombres": "LUIS",
            "apellidos": "PEREZ",
            "sexo": "M",
            "calidad": "Integrante",
            "id_formulario": "1",
            "anio": "2018",
            "programa": "I+D",
            "titulo": "Proyecto A",
            "id_archivo": "proy_2018_1",
        },
        {
            "row_id": "2",
            "pais_documento": "UY",
            "tipo_documento": "CI",
            "documento": "33333",
            "nombres": "JOSE",
            "apellidos": "RUIZ",
            "sexo": "M",
            "calidad": "Otros",
            "id_formulario": "1",
            "anio": "2018",
            "programa": "I+D",
            "titulo": "Proyecto A",
            "id_archivo": "proy_2018_1",
        },
    ]
    id_projects: set[str] = set()
    id_projects.add("proy_2018_1")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = TabularExtractor().extract_from_csv(csv_path, id_projects)

    participo_rels = [r for r in result.relationships if r.type == "PARTICIPO_EN"]
    calidades = {r.properties.get("calidad") for r in participo_rels}
    print("calidad", calidades)
    assert calidades == {"responsable", "integrante", "otros"}
    assert all(
        r.properties.get("calidad") for r in participo_rels
    ), "Toda PARTICIPO_EN debe tener calidad"


# -------------------------
# Preservación de Propiedad 'source' (Reglas + LLM)
# -------------------------


def test_ie_add_entities_dedup_by_id(extractor: EntityExtractor):
    """
    Agregar dos investigadores con el mismo ID reemplaza con el último visto
    (sin crear duplicados).
    """
    inv1 = Investigador(id="uy_ci_12345678", value={"nombre": "Juan Perez"})
    extractor.add_entities([inv1])

    inv2 = Investigador(id="uy_ci_12345678", value={"nombre": "Juan Pérez"})
    extractor.add_entities([inv2])

    matches = [e for e in extractor.res.entities if e.id == "uy_ci_12345678"]
    assert len(matches) == 1
    assert matches[0].value["nombre"] == "Juan Pérez"


def test_tabular_extractor_creates_project_and_title_extracted_from_chunk(tmp_path: Path):
    """Crea Proyecto y relación TITULO_EXTRAIDO_DE hacia el chunk de la fila."""
    import csv

    from institutional_graphrag.extraction.tabular_extractor import TabularExtractor

    csv_path = tmp_path / "equipos_test.csv"
    rows = [
        {
            "row_id": "0",
            "pais_documento": "UY",
            "tipo_documento": "CI",
            "documento": "11111",
            "nombres": "ANA",
            "apellidos": "GARCIA",
            "sexo": "F",
            "calidad": "Responsable",
            "id_formulario": "1",
            "anio": "2018",
            "programa": "I+D",
            "titulo": "Proyecto A",
            "palabras_claves2": "salud",
            "descripcion": "AA",
            "id_archivo": "proy_2018_1",
        },
    ]

    id_projects: set[str] = set()
    id_projects.add("proy_2018_1")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = TabularExtractor().extract_from_csv(csv_path, id_projects)

    proyectos = [e for e in result.entities if e.label == "Proyecto"]
    assert len(proyectos) == 1

    proyecto = proyectos[0]
    assert proyecto.id == "proy_2018_1"
    assert isinstance(proyecto.value, dict)
    assert proyecto.value == {
        "titulo": "proyecto a",
        "titulo_de_despliegue": "Proyecto A",
        "palabras_clave": ["salud"],
        "descripcion": "AA",
    }
    titulo_rels = [r for r in result.relationships if r.type == "TITULO_EXTRAIDO_DE"]

    assert len(titulo_rels) == 1

    rel = titulo_rels[0]
    assert rel.source_id == "proy_2018_1"

    # El target debe ser un Chunk creado desde la fila del CSV
    assert rel.target_id == "equipos_test#Chunk0"
