from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

# 🔁 Ajustá este import si tu archivo está en otro módulo
# Ej: from institutional_graphrag.graph.graph import ...
from institutional_graphrag.graph.builder import (
    GraphBuilder,
    Neo4jGraphBuilder,
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

        # usamos un field "computed" fijo para la clase
        @property
        def label(self) -> str:
            return lbl

    return _E


@dataclass
class DummyRel:
    type: str
    properties: dict[str, Any] | None = None


class FakeResult:
    def __init__(self, record=None):
        self._record = record or {"created": 0, "total": 0}

    def single(self):
        return self._record


class FakeSession:
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls

    def run(self, query: str, **params):
        self.calls.append({"query": query, "params": params})

        rows = params.get("rows", [])
        return FakeResult({"created": 0, "total": len(rows)})

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


def _only_upsert_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Las queries de upsert son las que llevan UNWIND $rows
    return [c for c in calls if "UNWIND $rows" in c["query"]]


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

    upsert_calls = _only_upsert_calls(fake_driver.calls)

    # Debe haber 2 upserts (A en batch de 2, B en batch de 1)
    assert len(upsert_calls) == 2

    # Revisar que query tenga MERGE con label correcto
    assert "(e:A" in upsert_calls[0]["query"]
    assert "(e:B" in upsert_calls[1]["query"]

    # Revisar rows
    rows_a = upsert_calls[0]["params"]["rows"]
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

    upsert_calls = _only_upsert_calls(fake_driver.calls)

    # Debe ejecutar 1 upsert de relaciones (solo 1 relación válida)
    assert len(upsert_calls) == 1
    call = upsert_calls[0]
    assert "MERGE (s)-[r:REL]->(t)" in call["query"]
    assert len(call["params"]["rows"]) == 1
    assert call["params"]["rows"][0]["source_id"] == "1"
    assert call["params"]["rows"][0]["target_id"] == "2"
    assert call["params"]["rows"][0]["properties"]["p"] == 1

    b.close()
