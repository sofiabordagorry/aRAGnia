from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
from neo4j import GraphDatabase
import json
from pathlib import Path

JSON_PATH = Path(__file__).parents[4] / "data" / "entities_relations" / "Entity_documents.json"

from institutional_graphrag.graph.schema import (
    Entity,
    Relationship,
    GraphSchema,
    validate_relationship_endpoints,
)


# ============================================================
# Neo4j backend
# ============================================================

class Neo4jGraphBuilder:
    """
    Builder de grafo usando Neo4j como backend.

    - Usa MERGE para evitar duplicación de nodos y relaciones
    - Valida esquema antes de persistir
    - Implementa batching para nodos y relaciones
    """

    def __init__(self, uri: str, user: str, password: str, batch_size: int = 500):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size
        self._create_constraints()

    def close(self):
        self.driver.close()

    # -------------------------
    # Constraints
    # -------------------------

    def _create_constraints(self):
        """Crear constraints únicos para cada label automáticamente."""
        with self.driver.session() as session:
            for label in GraphSchema.ENTITIES:
                query = f"""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.id IS UNIQUE;
                """
                session.run(query)

    # -------------------------
    # Utilidades de batching
    # -------------------------

    @staticmethod
    def _chunks(iterable: Iterable, size: int):
        """Yield successive chunks of size `size` from iterable."""
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # -------------------------
    # Nodos
    # -------------------------

    def upsert_entities(self, entities: Iterable[Entity]):
        """Upsert entities usando batching correcto, por label."""
        # Agrupar por label
        entities_by_label: dict[str, list[Entity]] = {}
        for e in entities:
            entities_by_label.setdefault(e.label, []).append(e)

        # Ahora sí hacemos batching por label
        for label, entities_label in entities_by_label.items():
            for batch in self._chunks(entities_label, self.batch_size):
                rows = []
                for e in batch:
                    props = {"id": e.id}
                    if isinstance(e.value, dict):
                        props.update(e.value)
                    else:
                        props["value"] = e.value
                    rows.append(props)

                if not rows:
                    continue

                query = f"""
                UNWIND $rows AS row
                MERGE (e:{label} {{id: row.id}})
                SET e += row
                """
                with self.driver.session() as session:
                    session.run(query, rows=rows)

    # -------------------------
    # Relaciones
    # -------------------------

    def upsert_relationships(self,relationships: Iterable[Tuple[Relationship, Entity, Entity]],):
        # juntamos todo a una lista para poder batchear
        rel_list = list(relationships)

        for batch in self._chunks(rel_list, self.batch_size):
            groups: Dict[Tuple[str, str, str], List[Dict]] = {}

            for rel, src, tgt in batch:
                if not validate_relationship_endpoints(rel, src, tgt):
                    raise ValueError(
                        f"Relación inválida: {rel.type} ({src.label} -> {tgt.label})"
                    )

                key = (rel.type, src.label, tgt.label)
                groups.setdefault(key, []).append({
                    "source_id": src.id,
                    "target_id": tgt.id,
                    "properties": rel.properties or {},
                })

            with self.driver.session() as session:
                for (rel_type, src_label, tgt_label), rows in groups.items():
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (s:{src_label} {{id: row.source_id}})
                    MATCH (t:{tgt_label} {{id: row.target_id}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    SET r += row.properties
                    """
                    session.run(query, rows=rows)

    
    def clear_graph(self):
        """
        Borra **todos los nodos y relaciones** del grafo.
        Úsalo con cuidado, es irreversible.
        """
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)
        print("Grafo borrado completamente.")
# ============================================================
# Fachada unificada
# ============================================================

class GraphBuilder:
    def __init__(self, neo4j_uri: Optional[str] = None, neo4j_user: Optional[str] = None, neo4j_password: Optional[str] = None):
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError("Faltan credenciales de Neo4j (uri/user/password)")

        self.backend = Neo4jGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)

    def close(self):
        if hasattr(self.backend, "close"):
            self.backend.close()

    def ingest(
        self,
        entities: Iterable[Entity],
        relationships: Iterable[Tuple[Relationship, Entity, Entity]],
    ):
        """Inserta entidades y relaciones usando batching genérico."""
        self.backend.upsert_entities(entities)
        self.backend.upsert_relationships(relationships)


# ============================================================
# Funciones auxiliares
# ============================================================

def load_graph_json(json_path: str | Path) -> Tuple[list[Entity], list[Tuple[Relationship, Entity, Entity]]]:
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    # 1) Entities: ahora vienen con "label"
    entities_by_id: dict[str, Entity] = {}
    ids_seen: dict[str, list[str]] = {}   # id -> [labels]

    for raw in payload.get("entities", []):
        entity_label = raw["label"]
        entity_id = raw["id"]
        value = raw.get("value")

        ids_seen.setdefault(entity_id, []).append(entity_label)

        entity_cls = GraphSchema.get_entity_class(entity_label)
        entity = entity_cls(id=entity_id, value=value)

        # seguimos guardando UNA entidad por id (como antes)
        if entity_id not in entities_by_id:
            entities_by_id[entity_id] = entity

    duplicates = {eid: labels for eid, labels in ids_seen.items() if len(labels) > 1}
    if duplicates:
        details = ", ".join(
            f"{eid} -> {sorted(set(labels))}"
            for eid, labels in duplicates.items()
        )
        raise ValueError(
            "IDs duplicados en entidades (mismo id con distintos labels): "
            + details
        )
    # 2) Relationships: endpoints se resuelven por id
    relationships: list[Tuple[Relationship, Entity, Entity]] = []
    for raw in payload.get("relationships", []):
        rel_type = raw["type"]
        source_id = raw["source_id"]
        target_id = raw["target_id"]
        properties = raw.get("properties") or {}

        rel_factory = GraphSchema.get_relationship_factory(rel_type)
        rel = rel_factory(source_id, target_id, properties)

        try:
            src = entities_by_id[source_id]
        except KeyError:
            raise ValueError(f"Relación {rel_type}: source_id no existe como entidad: {source_id}")

        try:
            tgt = entities_by_id[target_id]
        except KeyError:
            raise ValueError(f"Relación {rel_type}: target_id no existe como entidad: {target_id}")

        relationships.append((rel, src, tgt))

    return list(entities_by_id.values()), relationships

# ============================================================
# Prueba
# ============================================================

entities, relationships = load_graph_json(JSON_PATH)

grafo = GraphBuilder("bolt://localhost:7687", "neo4j", "matias2001")
grafo.backend.clear_graph()
grafo.ingest(entities=entities, relationships=relationships)


