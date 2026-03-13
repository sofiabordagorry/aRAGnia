from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# -------------------------
# Neo4j-like summary/counters fakes
# -------------------------
@dataclass
class FakeCounters:
    nodes_created: int = 0
    nodes_deleted: int = 0
    relationships_created: int = 0
    relationships_deleted: int = 0
    properties_set: int = 0
    labels_added: int = 0
    labels_removed: int = 0


@dataclass
class FakeSummary:
    counters: FakeCounters


class FakeRecord(dict):
    """Soporta record['x'] igual que neo4j."""

    pass


class FakeResult:
    def __init__(self, record: dict | None = None, counters: FakeCounters | None = None):
        self._record = FakeRecord(record) if record is not None else None
        self._summary = FakeSummary(counters=counters or FakeCounters())

    def single(self):
        return self._record

    def consume(self):
        return self._summary


class FakeSession:
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls

    def run(self, query: str, **params):
        self.calls.append({"query": query, "params": params})

        q_norm = " ".join(query.split()).upper()
        rows = params.get("rows", [])

        # ping / constraints: no importa el record, pero consume() debe existir
        if "RETURN 1" in q_norm or "CREATE CONSTRAINT" in q_norm:
            return FakeResult(record=None, counters=FakeCounters())

        # upsert entities: tu código espera ids/eids en record.single()
        if "UNWIND $ROWS" in q_norm and "MERGE (E:" in q_norm:
            ids = [r.get("id") for r in rows]
            eids = [f"e{i}" for i in range(len(ids))]
            return FakeResult(
                record={"ids": ids, "eids": eids},
                counters=FakeCounters(nodes_created=len(ids), properties_set=len(ids)),
            )

        # upsert relationships: espera rel_eids/pairs en record.single()
        if "UNWIND $ROWS" in q_norm and "MERGE (S)-[R:" in q_norm:
            rel_eids = [f"r{i}" for i in range(len(rows))]
            pairs = [[r.get("source_id"), r.get("target_id")] for r in rows]
            return FakeResult(
                record={"rel_eids": rel_eids, "pairs": pairs},
                counters=FakeCounters(relationships_created=len(rows), properties_set=len(rows)),
            )

        # default: algo consumible
        return FakeResult(record=None, counters=FakeCounters())

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
