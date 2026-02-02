from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from institutional_graphrag.graph.builder import (
    GraphBuilder,
    Neo4jGraphBuilder,
    build_containment_remap,
    load_graph_json,
)

# ============================================================
# Dummies / fakes para aislar tests (sin Neo4j real)
# ============================================================


@dataclass
class DummyEntity:
    id: str
    value: Any = None
    label: str = "Dummy"


def make_entity_class(lbl: str):
    @dataclass
    class _E:
        id: str
        value: Any = None

        @property
        def label(self) -> str:
            return lbl

    return _E


@dataclass
class DummyRel:
    type: str
    properties: dict[str, Any] | None = None


class FakeSession:
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls

    def run(self, query: str, **params):
        self.calls.append({"query": query, "params": params})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeDriver:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def session(self):
        return FakeSession(self.calls)

    def close(self):
        pass


# ============================================================
# Tests unitarios puros (sin Neo4j)
# ============================================================


def test_build_containment_remap_drops_shorter_contained_ids():
    entities_by_id = {
        # Investigadores
        "ana": DummyEntity(id="ana", label="Investigador"),
        "ana_maria": DummyEntity(id="ana_maria", label="Investigador"),
        "juan": DummyEntity(id="juan", label="Investigador"),
        "x_juan_perez": DummyEntity(id="x_juan_perez", label="Investigador"),
        # otro tipo no debe entrar
        "proy_1": DummyEntity(id="proy_1", label="Proyecto"),
    }

    remap = build_containment_remap(entities_by_id)

    # "ana" está contenido en "ana_maria" -> remap ana -> ana_maria
    assert remap["ana"] == "ana_maria"

    # "juan" está contenido en "x_juan_perez" -> remap juan -> x_juan_perez
    assert remap["juan"] == "x_juan_perez"

    # "ana_maria" no debe remapearse a sí mismo
    assert "ana_maria" not in remap


def test_graphbuilder_requires_credentials():
    with pytest.raises(ValueError, match="Faltan credenciales"):
        GraphBuilder(neo4j_uri=None, neo4j_user="neo4j", neo4j_password="x")

    with pytest.raises(ValueError, match="Faltan credenciales"):
        GraphBuilder(neo4j_uri="bolt://localhost:7687", neo4j_user=None, neo4j_password="x")

    with pytest.raises(ValueError, match="Faltan credenciales"):
        GraphBuilder(neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password=None)


def test_chunks_splits_iterable():
    data = list(range(10))
    chunks = list(Neo4jGraphBuilder._chunks(data, 4))
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


# ============================================================
# Tests de load_graph_json con monkeypatch de GraphSchema
# ============================================================


def test_load_graph_json_remaps_relationships_and_drops_contained_investigators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    # Payload mínimo
    payload = {
        "entities": [
            {"label": "Investigador", "id": "stella_peña", "value": {"name": "Stella Peña"}},
            {"label": "Investigador", "id": "qf_stella_peña", "value": {"name": "Stella Peña"}},
            {"label": "Proyecto", "id": "proy_1", "value": {"title": "P1"}},
        ],
        "relationships": [
            # Esta relación viene apuntando al ID corto -> debe remapearse al largo
            {
                "type": "PARTICIPO_EN",
                "source_id": "stella_peña",
                "target_id": "proy_1",
                "properties": {},
            },
            # Target también remapea (si existiera)
            {
                "type": "EVIDENCIA_DE",
                "source_id": "proy_1",
                "target_id": "stella_peña",
                "properties": {"x": 1},
            },
        ],
    }

    json_path = tmp_path / "Entity_documents.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # --- Monkeypatch GraphSchema y validate_relationship_endpoints dentro del módulo ---
    import institutional_graphrag.graph.builder as graph_mod

    def fake_get_entity_class(label: str):
        return make_entity_class(label)

    def fake_get_rel_factory(rel_type: str) -> Callable[[str, str, dict[str, Any]], DummyRel]:
        def _factory(source_id: str, target_id: str, properties: dict[str, Any]):
            return DummyRel(type=rel_type, properties=properties)

        return _factory

    class FakeGraphSchema:
        ENTITIES = ["Investigador", "Proyecto"]

        @staticmethod
        def get_entity_class(label: str):
            return fake_get_entity_class(label)

        @staticmethod
        def get_relationship_factory(rel_type: str):
            return fake_get_rel_factory(rel_type)

    monkeypatch.setattr(graph_mod, "GraphSchema", FakeGraphSchema)
    monkeypatch.setattr(graph_mod, "validate_relationship_endpoints", lambda rel, src, tgt: True)

    entities, relationships = load_graph_json(json_path)

    # 1) Debe haberse eliminado el Investigador contenido: "stella_peña"
    entity_ids = {e.id for e in entities}
    assert "stella_peña" not in entity_ids
    assert "qf_stella_peña" in entity_ids

    # 2) Relaciones deben estar remapeadas
    # relationships es lista de (rel, src_entity, tgt_entity)
    rel_types_and_endpoints = {(rel.type, src.id, tgt.id) for (rel, src, tgt) in relationships}

    assert ("PARTICIPO_EN", "qf_stella_peña", "proy_1") in rel_types_and_endpoints
    assert ("EVIDENCIA_DE", "proy_1", "qf_stella_peña") in rel_types_and_endpoints

    assert "source_id no existe" not in caplog.text


# ============================================================
# Tests de Neo4jGraphBuilder (mockeando GraphDatabase.driver)
# ============================================================


def test_neo4j_builder_creates_constraints(monkeypatch: pytest.MonkeyPatch):
    import institutional_graphrag.graph.builder as graph_mod

    fake_driver = FakeDriver()

    # Mock del driver neo4j
    monkeypatch.setattr(graph_mod.GraphDatabase, "driver", lambda *args, **kwargs: fake_driver)

    # Mock GraphSchema.ENTITIES
    class FakeGraphSchema:
        ENTITIES = ["Proyecto", "Investigador"]

    monkeypatch.setattr(graph_mod, "GraphSchema", FakeGraphSchema)

    b = graph_mod.Neo4jGraphBuilder("bolt://x:7687", "u", "p", batch_size=2)

    # Se deben haber ejecutado 2 CREATE CONSTRAINT ...
    q = "\n".join(call["query"] for call in fake_driver.calls)
    assert "CREATE CONSTRAINT IF NOT EXISTS" in q
    assert "FOR (n:Proyecto)" in q
    assert "FOR (n:Investigador)" in q

    b.close()


def test_upsert_entities_batches_by_label(monkeypatch: pytest.MonkeyPatch):
    import institutional_graphrag.graph.builder as graph_mod

    fake_driver = FakeDriver()
    monkeypatch.setattr(graph_mod.GraphDatabase, "driver", lambda *args, **kwargs: fake_driver)

    class FakeGraphSchema:
        ENTITIES = ["A", "B"]

    monkeypatch.setattr(graph_mod, "GraphSchema", FakeGraphSchema)

    b = graph_mod.Neo4jGraphBuilder("bolt://x:7687", "u", "p", batch_size=2)

    entities = [
        DummyEntity(id="1", value={"k": 1}, label="A"),
        DummyEntity(id="2", value="v2", label="A"),
        DummyEntity(id="3", value={"k": 3}, label="B"),
    ]

    # limpiar calls de constraints para inspeccionar solo upsert
    fake_driver.calls.clear()

    b.upsert_entities(entities)

    # Debe haber 2 runs (A en batch de 2, B en batch de 1)
    assert len(fake_driver.calls) == 2

    # Revisar que query tenga MERGE con label correcto
    assert "(e:A" in fake_driver.calls[0]["query"]
    assert "(e:B" in fake_driver.calls[1]["query"]

    # Revisar rows
    rows_a = fake_driver.calls[0]["params"]["rows"]
    assert rows_a[0]["id"] == "1" and rows_a[0]["k"] == 1
    assert rows_a[1]["id"] == "2" and rows_a[1]["value"] == "v2"

    b.close()


def test_upsert_relationships_dedupes_and_blocks_self_loops(monkeypatch: pytest.MonkeyPatch):
    import institutional_graphrag.graph.builder as graph_mod

    fake_driver = FakeDriver()
    monkeypatch.setattr(graph_mod.GraphDatabase, "driver", lambda *args, **kwargs: fake_driver)

    class FakeGraphSchema:
        ENTITIES = ["A", "B"]

    monkeypatch.setattr(graph_mod, "GraphSchema", FakeGraphSchema)
    monkeypatch.setattr(graph_mod, "validate_relationship_endpoints", lambda rel, src, tgt: True)

    b = graph_mod.Neo4jGraphBuilder("bolt://x:7687", "u", "p", batch_size=50)

    fake_driver.calls.clear()

    a1 = DummyEntity(id="1", label="A")
    b1 = DummyEntity(id="2", label="B")

    rel = DummyRel(type="REL", properties={"p": 1})

    relationships = [
        (rel, a1, b1),
        (rel, a1, b1),  # duplicada -> se omite
        (rel, a1, a1),  # self-loop -> se omite
    ]

    b.upsert_relationships(relationships)

    # Debe ejecutar 1 query de MERGE de relaciones (solo 1 relación válida)
    assert len(fake_driver.calls) == 1
    call = fake_driver.calls[0]
    assert "MERGE (s)-[r:REL]->(t)" in call["query"]
    assert len(call["params"]["rows"]) == 1
    assert call["params"]["rows"][0]["source_id"] == "1"
    assert call["params"]["rows"][0]["target_id"] == "2"
    assert call["params"]["rows"][0]["properties"]["p"] == 1

    b.close()
