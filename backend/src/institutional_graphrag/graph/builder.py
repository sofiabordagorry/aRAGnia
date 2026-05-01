from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired

from institutional_graphrag.graph.schema import (
    Entity,
    GraphSchema,
    Relationship,
    validate_relationship_endpoints,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger("graph_ingest")

T = TypeVar("T")
OUTPUT_PATH = Path(__file__).parents[4] / "data" / "entities_relations" / "export_graph.json"


@dataclass
class Neo4jStats:
    nodes_created: int = 0
    nodes_deleted: int = 0
    relationships_created: int = 0
    relationships_deleted: int = 0
    properties_set: int = 0
    labels_added: int = 0
    labels_removed: int = 0

    node_ids: List[str] = field(default_factory=list)
    node_eids: List[str] = field(default_factory=list)
    rel_eids: List[str] = field(default_factory=list)
    rel_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def add_counters(self, counters: Any) -> None:
        self.nodes_created += counters.nodes_created
        self.nodes_deleted += counters.nodes_deleted
        self.relationships_created += counters.relationships_created
        self.relationships_deleted += counters.relationships_deleted
        self.properties_set += counters.properties_set
        self.labels_added += counters.labels_added
        self.labels_removed += counters.labels_removed


class GraphBuilder:
    """
    Builder de grafo usando Neo4j como backend.

    - Usa MERGE para evitar duplicación de nodos y relaciones
    - Valida esquema antes de persistir
    - Implementa batching para nodos y relaciones
    """

    def __init__(self, uri: str, user: str, password: str, batch_size: int = 500):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.batch_size = batch_size
            self._ping(max_attempts=6, sleep_s=2)
            self._create_constraints(max_attempts=5, sleep_s=2)
        except AuthError as exc:
            logger.error("Neo4j: credenciales inválidas")
            raise RuntimeError(
                "No se pudo autenticar con Neo4j. Verificá usuario y contraseña."
            ) from exc
        except ServiceUnavailable as exc:
            logger.error("Neo4j: servidor no disponible en %s", uri)
            raise RuntimeError(
                "No se pudo conectar a Neo4j. ¿Está el contenedor corriendo?"
            ) from exc
        except SessionExpired as exc:
            logger.error("Neo4j: conexión expirada durante el handshake")
            raise RuntimeError("La conexión con Neo4j falló durante la inicialización.") from exc

    def close(self) -> None:
        self.driver.close()

    def _ping(self, max_attempts: int = 2, sleep_s: int = 2) -> None:
        for attempt in range(1, max_attempts + 1):
            try:
                with self.driver.session() as session:
                    session.run("RETURN 1").consume()
                logger.info("Neo4j listo (ping OK)")
                return
            except (ServiceUnavailable, SessionExpired, OSError) as e:
                logger.warning("Neo4j no listo (ping %d/%d): %s", attempt, max_attempts, e)
                time.sleep(sleep_s)
            except AuthError:
                raise
        raise ValueError("No se pudo conectar a Neo4j")

    # -------------------------
    # Constraints
    # -------------------------

    def _create_constraints(self, max_attempts: int = 5, sleep_s: int = 2) -> None:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self.driver.session() as session:
                    for label in GraphSchema.ENTITIES:
                        query = f"""
                        CREATE CONSTRAINT IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.id IS UNIQUE
                        """
                        session.run(query).consume()
                logger.info("Constraints OK")
                return
            except (ServiceUnavailable, SessionExpired, OSError) as e:
                last_err = e
                logger.warning("Fallo creando constraints (%d/%d): %s", attempt, max_attempts, e)
                time.sleep(sleep_s)
            except AuthError:
                raise

        raise RuntimeError(
            "Neo4j no respondió al crear constraints. "
            "Probablemente el contenedor se reinició o todavía no terminó de iniciar."
        ) from last_err

    # -------------------------
    # Utilidades de batching
    # -------------------------

    @staticmethod
    def _chunks(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
        chunk: List[Any] = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    @staticmethod
    def _extend_sample(dst: List[Any], src: List[Any], sample_ids: int) -> None:
        if sample_ids <= 0:
            return
        remaining = sample_ids - len(dst)
        if remaining <= 0:
            return
        dst.extend(src[:remaining])

    @staticmethod
    def _entity_to_row(entity: Entity) -> dict[str, Any]:
        props: dict[str, Any] = {"id": entity.id}
        if isinstance(entity.value, dict):
            props.update(entity.value)
        else:
            props["value"] = entity.value
        return props

    # -------------------------
    # Entidades
    # -------------------------
    def upsert_entities(self, entities: Iterable[Entity], *, sample_ids: int = 50) -> Neo4jStats:
        stats = Neo4jStats()

        entities_by_label: dict[str, list[Entity]] = {}
        for entity in entities:
            entities_by_label.setdefault(entity.label, []).append(entity)

        for label, entities_label in entities_by_label.items():
            for batch in self._chunks(entities_label, self.batch_size):
                rows = [self._entity_to_row(entity) for entity in batch]
                if not rows:
                    continue

                query = f"""
                UNWIND $rows AS row
                MERGE (e:{label} {{id: row.id}})
                ON CREATE SET e.__created__ = true
                SET e += row
                RETURN collect(row.id) AS ids,
                       collect(elementId(e)) AS eids
                """

                with self.driver.session() as session:
                    result = session.run(query, rows=rows)
                    record = result.single()
                    summary = result.consume()

                stats.add_counters(summary.counters)

                if record is not None:
                    self._extend_sample(stats.node_ids, record["ids"] or [], sample_ids)
                    self._extend_sample(stats.node_eids, record["eids"] or [], sample_ids)

        return stats

    # -------------------------
    # Relaciones
    # -------------------------

    def _prepare_relationships(
        self,
        relationships: Iterable[Tuple[Relationship, Entity, Entity]],
    ) -> list[Tuple[Relationship, Entity, Entity]]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[Tuple[Relationship, Entity, Entity]] = []

        for rel, src, tgt in relationships:
            if src.id == tgt.id:
                logger.warning(
                    "Relación inválida (self-loop) se omite: %s %s -> %s",
                    rel.type,
                    src.id,
                    tgt.id,
                )
                continue

            key = (rel.type, src.id, tgt.id)
            if key in seen:
                logger.warning(
                    "Relación repetida (se omite): %s %s -> %s",
                    rel.type,
                    src.id,
                    tgt.id,
                )
                continue

            if not validate_relationship_endpoints(rel, src, tgt):
                logger.error(
                    "Relación inválida por schema (se omite): %s (%s -> %s)",
                    rel.type,
                    src.label,
                    tgt.label,
                )
                continue

            seen.add(key)
            deduped.append((rel, src, tgt))

        return deduped

    @staticmethod
    def _group_relationship_batch(
        batch: list[Tuple[Relationship, Entity, Entity]],
    ) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

        for rel, src, tgt in batch:
            gkey = (rel.type, src.label, tgt.label)
            groups.setdefault(gkey, []).append(
                {
                    "source_id": src.id,
                    "target_id": tgt.id,
                    "properties": rel.properties or {},
                }
            )

        return groups

    def upsert_relationships(
        self,
        relationships: Iterable[Tuple[Relationship, Entity, Entity]],
        *,
        sample_ids: int = 50,
    ) -> Neo4jStats:
        stats = Neo4jStats()
        deduped = self._prepare_relationships(relationships)

        for batch in self._chunks(deduped, self.batch_size):
            groups = self._group_relationship_batch(batch)

            with self.driver.session() as session:
                for (rel_type, src_label, tgt_label), rows in groups.items():
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (s:{src_label} {{id: row.source_id}})
                    MATCH (t:{tgt_label} {{id: row.target_id}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    ON CREATE SET r.__created__ = true
                    SET r += row.properties
                    RETURN collect(elementId(r)) AS rel_eids,
                           collect([row.source_id, row.target_id]) AS pairs
                    """

                    result = session.run(query, rows=rows)
                    record = result.single()
                    summary = result.consume()

                    stats.add_counters(summary.counters)

                    if record is not None:
                        self._extend_sample(stats.rel_eids, record["rel_eids"] or [], sample_ids)

                        pairs_raw = record["pairs"] or []
                        pairs: list[tuple[str, str]] = [
                            (pair[0], pair[1])
                            for pair in pairs_raw
                            if isinstance(pair, list) and len(pair) == 2
                        ]
                        self._extend_sample(stats.rel_pairs, pairs, sample_ids)

        return stats

    def clear_graph(self) -> None:
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)
        print("Grafo borrado completamente.")

    def fetch_investigador_ids(self) -> list[str]:
        query = "MATCH (i:Investigador) RETURN i.id AS id"
        with self.driver.session() as session:
            rows = session.run(query).data()
        return [row["id"] for row in rows if row.get("id")]

    def fetch_researchers(self) -> list[dict]:
        query = "MATCH (i:Investigador) RETURN i"
        with self.driver.session() as session:
            result = session.run(query)

            entities = []
            for record in result:
                node = record["i"]
                props = dict(node)

                entities.append(
                    {
                        "id": props.get("id"),
                        "label": "Investigador",
                        "value": {k: v for k, v in props.items() if k not in {"id", "__created__"}},
                    }
                )

        return entities

    def merge_node_id(
        self,
        *,
        label: str,
        old_id: str,
        new_id: str,
        sample_ids: int = 50,
    ) -> Neo4jStats:
        stats = Neo4jStats()
        if old_id == new_id:
            return stats

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (a:{label} {{id: $old_id}})
                MERGE (b:{label} {{id: $new_id}})
                RETURN elementId(a) AS a_eid, elementId(b) AS b_eid
                """,
                old_id=old_id,
                new_id=new_id,
            )
            record = result.single()
            summary = result.consume()
            stats.add_counters(summary.counters)

            if record is not None:
                self._extend_sample(
                    stats.node_eids,
                    [record["a_eid"], record["b_eid"]],
                    sample_ids,
                )

            result = session.run(
                f"""
                MATCH (a:{label} {{id: $old_id}})
                MATCH (b:{label} {{id: $new_id}})
                SET b += a{{.*, id: b.id, name: coalesce(b.name, a.name)}}
                """,
                old_id=old_id,
                new_id=new_id,
            )
            stats.add_counters(result.consume().counters)

            rel_types = [
                row["type"]
                for row in session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType AS type RETURN type"
                ).data()
            ]

            for rel_type in rel_types:
                result = session.run(
                    f"""
                    MATCH (a:{label} {{id: $old_id}})-[r:{rel_type}]->(x)
                    MATCH (b:{label} {{id: $new_id}})
                    MERGE (b)-[r2:{rel_type}]->(x)
                    SET r2 += properties(r)
                    DELETE r
                    RETURN collect(elementId(r2)) AS moved_rel_eids,
                           collect([b.id, x.id]) AS pairs
                    """,
                    old_id=old_id,
                    new_id=new_id,
                )
                record = result.single()
                summary = result.consume()
                stats.add_counters(summary.counters)

                if record is not None:
                    self._extend_sample(
                        stats.rel_eids,
                        record["moved_rel_eids"] or [],
                        sample_ids,
                    )

                result = session.run(
                    f"""
                    MATCH (y)-[r:{rel_type}]->(a:{label} {{id: $old_id}})
                    MATCH (b:{label} {{id: $new_id}})
                    MERGE (y)-[r2:{rel_type}]->(b)
                    SET r2 += properties(r)
                    DELETE r
                    RETURN collect(elementId(r2)) AS moved_rel_eids
                    """,
                    old_id=old_id,
                    new_id=new_id,
                )
                record = result.single()
                summary = result.consume()
                stats.add_counters(summary.counters)

                if record is not None:
                    self._extend_sample(
                        stats.rel_eids,
                        record["moved_rel_eids"] or [],
                        sample_ids,
                    )

            result = session.run(
                f"""
                MATCH (a:{label} {{id: $old_id}})
                DETACH DELETE a
                """,
                old_id=old_id,
            )
            stats.add_counters(result.consume().counters)

        return stats

    def fetch_entity_by_id(self, entity_id: str) -> list[Entity]:
        query = """
        MATCH (n {id: $id})
        RETURN labels(n) AS labels
        LIMIT 5
        """
        with self.driver.session() as session:
            rows = session.run(query, id=entity_id).data()

        if not rows:
            return []

        labels = rows[0]["labels"] or []
        out: list[Entity] = []

        for label in labels:
            if label in GraphSchema.ENTITIES:
                entity_cls = GraphSchema.get_entity_class(label)
                out.append(entity_cls(id=entity_id, value={}))

        return out

    @staticmethod
    def _pick_primary_label(labels: list[str]) -> str:
        for label in labels:
            if label in GraphSchema.ENTITIES:
                return label
        return labels[0] if labels else "Entidad"

    @staticmethod
    def _display_value(label: str, props: dict[str, Any]) -> str:
        if label == "Investigador":
            return str(props.get("name") or props.get("id") or "Investigador")
        if label == "Proyecto":
            return str(props.get("title") or props.get("id") or "Proyecto")
        if label == "Topico":
            return str(props.get("value") or props.get("id") or "Topico")
        if label == "Anio":
            return str(props.get("year") or props.get("id") or "Anio")
        return str(props.get("id") or props.get("value") or label)

    @staticmethod
    def _normalized_search(value: Optional[str]) -> str:
        return (value or "").strip().lower()

    def fetch_entities_catalog(
        self,
        *,
        search: Optional[str] = None,
        entity_label: Optional[str] = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 100))
        search_text = self._normalized_search(search)

        allowed_labels = {label for label in GraphSchema.ENTITIES if label != "Chunk"}
        selected_label = (entity_label or "").strip()
        if selected_label and selected_label not in allowed_labels:
            raise ValueError(f"Tipo de entidad no soportado: {selected_label}")

        label_filter = ""
        params: dict[str, Any] = {"search": search_text, "limit": safe_limit}
        if selected_label:
            label_filter = "AND $label IN labels(n)"
            params["label"] = selected_label

        query = f"""
        MATCH (n)
        WHERE NOT n:Chunk
          {label_filter}
          AND (
              $search = ''
              OR toLower(coalesce(n.id, '')) CONTAINS $search
              OR toLower(coalesce(n.name, '')) CONTAINS $search
              OR toLower(coalesce(n.value, '')) CONTAINS $search
              OR toLower(coalesce(n.title, '')) CONTAINS $search
              OR toLower(coalesce(toString(n.year), '')) CONTAINS $search
          )
        RETURN n, labels(n) AS labels
        ORDER BY coalesce(n.name, n.value, n.title, n.id, '') ASC
        LIMIT $limit
        """

        with self.driver.session() as session:
            rows = session.run(query, **params).data()

        entities: list[dict[str, Any]] = []
        for row in rows:
            node = row.get("n")
            if node is None:
                continue
            props = dict(node)
            node_id = str(props.get("id") or "").strip()
            if not node_id:
                continue
            labels = [str(label) for label in (row.get("labels") or [])]
            primary_label = self._pick_primary_label(labels)
            if primary_label == "Chunk":
                continue
            entities.append(
                {
                    "id": node_id,
                    "label": primary_label,
                    "display": self._display_value(primary_label, props),
                }
            )

        return {
            "entities": entities,
            "summary": {
                "result_count": len(entities),
            },
        }

    def fetch_graph_neighborhood(
        self,
        *,
        entity_id: str,
        relationship_limit: int = 320,
        alias_only: bool = False,
    ) -> dict[str, Any]:
        safe_relationship_limit = max(1, min(relationship_limit, 1200))
        entity_id = entity_id.strip()
        if not entity_id:
            return {
                "nodes": [],
                "edges": [],
                "summary": {
                    "node_count": 0,
                    "edge_count": 0,
                    "alias_edge_count": 0,
                },
            }

        with self.driver.session() as session:
            source_check = session.run(
                """
                MATCH (n {id: $entity_id})
                WHERE NOT n:Chunk
                RETURN n, labels(n) AS labels
                LIMIT 1
                """,
                entity_id=entity_id,
            ).data()
            if not source_check:
                return {
                    "nodes": [],
                    "edges": [],
                    "summary": {
                        "node_count": 0,
                        "edge_count": 0,
                        "alias_edge_count": 0,
                    },
                }

            edge_rows = session.run(
                """
                MATCH (n {id: $entity_id})-[r]-(m)
                WHERE NOT n:Chunk
                  AND NOT m:Chunk
                  AND ($alias_only = false OR type(r) = 'POSIBLE_ALIAS')
                RETURN startNode(r).id AS source_id,
                       endNode(r).id AS target_id,
                       type(r) AS type,
                       properties(r) AS properties
                LIMIT $relationship_limit
                """,
                entity_id=entity_id,
                alias_only=alias_only,
                relationship_limit=safe_relationship_limit,
            ).data()

            node_rows = session.run(
                """
                MATCH (n {id: $entity_id})
                WHERE NOT n:Chunk
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE NOT m:Chunk
                  AND ($alias_only = false OR type(r) = 'POSIBLE_ALIAS')
                WITH collect(DISTINCT n) + collect(DISTINCT m) AS node_list
                UNWIND node_list AS node
                WITH DISTINCT node
                RETURN node, labels(node) AS labels
                """,
                entity_id=entity_id,
                alias_only=alias_only,
            ).data()

        edges = [
            {
                "source": str(row.get("source_id") or ""),
                "target": str(row.get("target_id") or ""),
                "type": str(row.get("type") or "RELACION"),
                "is_alias": str(row.get("type") or "") == "POSIBLE_ALIAS",
                "properties": row.get("properties") or {},
            }
            for row in edge_rows
            if row.get("source_id") and row.get("target_id")
        ]

        degree_by_id: dict[str, int] = {}
        for edge in edges:
            source_id = str(edge["source"])
            target_id = str(edge["target"])
            degree_by_id[source_id] = degree_by_id.get(source_id, 0) + 1
            degree_by_id[target_id] = degree_by_id.get(target_id, 0) + 1

        nodes: list[dict[str, Any]] = []
        node_ids: list[str] = []
        for row in node_rows:
            node = row.get("node")
            if node is None:
                continue
            props = dict(node)
            node_id = str(props.get("id") or "").strip()
            if not node_id:
                continue
            labels = [str(label) for label in (row.get("labels") or [])]
            primary_label = self._pick_primary_label(labels)
            if primary_label == "Chunk":
                continue
            nodes.append(
                {
                    "id": node_id,
                    "label": primary_label,
                    "display": self._display_value(primary_label, props),
                    "degree": degree_by_id.get(node_id, 0),
                    "is_alias_candidate": False,
                }
            )
            node_ids.append(node_id)

        alias_node_ids: set[str] = set()
        for edge in edges:
            if bool(edge["is_alias"]):
                alias_node_ids.add(str(edge["source"]))
                alias_node_ids.add(str(edge["target"]))

        for node in nodes:
            node["is_alias_candidate"] = node["id"] in alias_node_ids

        alias_edge_count = sum(1 for edge in edges if bool(edge["is_alias"]))

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "alias_edge_count": alias_edge_count,
            },
        }

    def fetch_graph_snapshot(
        self,
        *,
        node_limit: int = 160,
        relationship_limit: int = 320,
        alias_only: bool = False,
    ) -> dict[str, Any]:
        safe_node_limit = max(1, min(node_limit, 500))
        safe_relationship_limit = max(1, min(relationship_limit, 1200))

        node_query = (
            """
            MATCH (n:Investigador)-[:POSIBLE_ALIAS]-()
            OPTIONAL MATCH (n)-[r]-()
            WITH DISTINCT n, count(r) AS degree
            ORDER BY degree DESC, coalesce(n.id, "") ASC
            LIMIT $node_limit
            RETURN n, labels(n) AS labels, degree
            """
            if alias_only
            else """
            MATCH (n)
                        WHERE NOT n:Chunk
            OPTIONAL MATCH (n)-[r]-()
                        WHERE NOT startNode(r):Chunk
                            AND NOT endNode(r):Chunk
            WITH n, count(r) AS degree
            ORDER BY degree DESC, coalesce(n.id, "") ASC
            LIMIT $node_limit
            RETURN n, labels(n) AS labels, degree
            """
        )

        with self.driver.session() as session:
            node_rows = session.run(node_query, node_limit=safe_node_limit).data()

            if not node_rows:
                return {
                    "nodes": [],
                    "edges": [],
                    "summary": {
                        "node_count": 0,
                        "edge_count": 0,
                        "alias_edge_count": 0,
                    },
                }

            node_ids: list[str] = []
            nodes: list[dict[str, Any]] = []
            seen_node_ids: set[str] = set()

            for row in node_rows:
                node = row.get("n")
                if node is None:
                    continue

                props = dict(node)
                node_id = str(props.get("id") or "").strip()
                if not node_id or node_id in seen_node_ids:
                    continue

                labels = [str(label) for label in (row.get("labels") or [])]
                primary_label = self._pick_primary_label(labels)
                degree = int(row.get("degree") or 0)

                nodes.append(
                    {
                        "id": node_id,
                        "label": primary_label,
                        "display": self._display_value(primary_label, props),
                        "degree": degree,
                        "is_alias_candidate": False,
                    }
                )
                node_ids.append(node_id)
                seen_node_ids.add(node_id)

            edge_filter = "AND type(r) = 'POSIBLE_ALIAS'" if alias_only else ""
            edge_query = f"""
            MATCH (source)-[r]->(target)
                        WHERE source.id IN $node_ids AND target.id IN $node_ids
                            AND NOT source:Chunk
                            AND NOT target:Chunk
            {edge_filter}
            RETURN source.id AS source_id,
                   target.id AS target_id,
                   type(r) AS type,
                   properties(r) AS properties
            ORDER BY type, source_id, target_id
            LIMIT $relationship_limit
            """

            edge_rows = session.run(
                edge_query,
                node_ids=node_ids,
                relationship_limit=safe_relationship_limit,
            ).data()

            alias_rows = session.run(
                """
                MATCH (source:Investigador)-[r:POSIBLE_ALIAS]->(target:Investigador)
                WHERE source.id IN $node_ids AND target.id IN $node_ids
                RETURN DISTINCT source.id AS source_id, target.id AS target_id
                """,
                node_ids=node_ids,
            ).data()

        alias_node_ids: set[str] = set()
        for row in alias_rows:
            source_id = str(row.get("source_id") or "")
            target_id = str(row.get("target_id") or "")
            if source_id:
                alias_node_ids.add(source_id)
            if target_id:
                alias_node_ids.add(target_id)

        for node in nodes:
            node["is_alias_candidate"] = node["id"] in alias_node_ids

        edges = [
            {
                "source": str(row.get("source_id") or ""),
                "target": str(row.get("target_id") or ""),
                "type": str(row.get("type") or "RELACION"),
                "is_alias": str(row.get("type") or "") == "POSIBLE_ALIAS",
                "properties": row.get("properties") or {},
            }
            for row in edge_rows
            if row.get("source_id") and row.get("target_id")
        ]

        alias_edge_count = sum(1 for edge in edges if edge["is_alias"])

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "alias_edge_count": alias_edge_count,
            },
        }

    def merge_researchers(self, *, source_id: str, target_id: str) -> Neo4jStats:
        """
        Unifica dos nodos Investigador: transfiere todas las relaciones de source a target
        y elimina el nodo source. Usa Cypher nativo (sin APOC).
        """
        if source_id == target_id:
            raise ValueError("source_id y target_id deben ser diferentes")

        stats = self.merge_node_id(
            label="Investigador",
            old_id=source_id,
            new_id=target_id,
        )

        # Eliminar self-loops que puedan haber quedado en el nodo target
        # (e.g. si source tenía POSIBLE_ALIAS -> target)
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (b:Investigador {id: $target_id})-[r]-(b)
                DELETE r
                """,
                target_id=target_id,
            )
            stats.add_counters(result.consume().counters)

        return stats

    def fetch_alias_candidates(self, *, search: Optional[str] = None) -> dict[str, Any]:
        search_text = self._normalized_search(search)
        query = """
        MATCH (source:Investigador)-[r:POSIBLE_ALIAS]->(target:Investigador)
        WHERE $search = ''
           OR toLower(coalesce(source.id, '')) CONTAINS $search
           OR toLower(coalesce(source.name, '')) CONTAINS $search
           OR toLower(coalesce(target.id, '')) CONTAINS $search
           OR toLower(coalesce(target.name, '')) CONTAINS $search
        RETURN DISTINCT
            source.id AS source_id,
            coalesce(source.name, source.id) AS source_name,
            target.id AS target_id,
            coalesce(target.name, target.id) AS target_name,
            properties(r) AS relationship_properties
        ORDER BY source_name, target_name
        """

        with self.driver.session() as session:
            rows = session.run(query, search=search_text).data()

        alias_entities: dict[str, dict[str, Any]] = {}
        pairs: list[dict[str, Any]] = []

        for row in rows:
            source_id = str(row.get("source_id") or "").strip()
            target_id = str(row.get("target_id") or "").strip()
            source_name = str(row.get("source_name") or source_id)
            target_name = str(row.get("target_name") or target_id)
            if not source_id or not target_id:
                continue

            alias_entities[source_id] = {
                "id": source_id,
                "name": source_name,
                "label": "Investigador",
            }
            alias_entities[target_id] = {
                "id": target_id,
                "name": target_name,
                "label": "Investigador",
            }
            pairs.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "target_id": target_id,
                    "target_name": target_name,
                    "relationship_properties": row.get("relationship_properties") or {},
                }
            )

        return {
            "pairs": pairs,
            "entities": sorted(
                alias_entities.values(), key=lambda item: (item["name"], item["id"])
            ),
            "summary": {
                "pair_count": len(pairs),
                "entity_count": len(alias_entities),
            },
        }

    def ingest(
        self,
        entities: Iterable[Entity],
        relationships: Iterable[Tuple[Relationship, Entity, Entity]],
    ) -> None:
        ent_stats = self.upsert_entities(entities, sample_ids=15)
        rel_stats = self.upsert_relationships(relationships, sample_ids=15)
        self.export_graph(OUTPUT_PATH)
        self._print_stats(ent_stats, rel_stats)

    @staticmethod
    def _print_stats(ent_stats: Neo4jStats, rel_stats: Neo4jStats) -> None:
        print(
            "ENTIDADES:",
            "nodes_created=",
            ent_stats.nodes_created,
            "props_set=",
            ent_stats.properties_set,
            "sample_node_ids=",
            ent_stats.node_ids[:10],
            "sample_node_eids=",
            ent_stats.node_eids[:10],
        )

        print(
            "RELACIONES:",
            "rels_created=",
            rel_stats.relationships_created,
            "props_set=",
            rel_stats.properties_set,
            "sample_rel_eids=",
            rel_stats.rel_eids[:10],
            "sample_pairs=",
            rel_stats.rel_pairs[:10],
        )

    def export_graph(
        self,
        output_json_path: str | Path,
    ) -> Dict[str, Any]:
        output_json_path = Path(output_json_path)

        # -----------------------------
        # 2) Extracción desde Neo4j
        # -----------------------------

        neo4j_entities: List[Dict[str, Any]] = []
        neo4j_relationships: List[Dict[str, Any]] = []
        entity_type_counts: Dict[str, int] = {}
        relationship_type_counts: Dict[str, int] = {}
        with self.driver.session() as session:
            # Entidades
            node_query = """
            MATCH (n)
            RETURN
            coalesce(n.id, elementId(n)) AS id,
            head(labels(n)) AS label,
            properties(n) AS value
            ORDER BY id
            """
            node_result = session.run(node_query)

            for record in node_result:
                label = record["label"] or "SinLabel"
                entity_id = record["id"]

                props = dict(record["value"]) if record["value"] is not None else {}

                props.pop("__created__", None)
                props.pop("id", None)

                if label in {"Proyecto", "Topico", "Dominio"}:
                    value = props.get("value")

                    if isinstance(value, dict) and "value" in value:
                        value = value["value"]

                    if value is None:
                        value = ""

                else:
                    value = props

                neo4j_entities.append(
                    {
                        "id": entity_id,
                        "label": label,
                        "value": value,
                    }
                )

                entity_type_counts[label] = entity_type_counts.get(label, 0) + 1

            # Relaciones
            rel_query = """
            MATCH (a)-[r]->(b)
            RETURN
            type(r) AS type,
            coalesce(a.id, elementId(a)) AS source_id,
            coalesce(b.id, elementId(b)) AS target_id,
            properties(r) AS properties
            ORDER BY type, source_id, target_id
            """
            rel_result = session.run(rel_query)

            for record in rel_result:
                rel_type = record["type"]
                properties = dict(record["properties"]) if record["properties"] is not None else {}

                # eliminar propiedad
                properties.pop("__created__", None)

                neo4j_relationships.append(
                    {
                        "type": rel_type,
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "properties": properties,
                    }
                )

                relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        # -----------------------------
        # 3) Guardado final
        # -----------------------------
        output_data = {
            "entities": neo4j_entities,
            "relationships": neo4j_relationships,
            "source_summary": {
                "neo4j_entities": len(neo4j_entities),
                "neo4j_relationships": len(neo4j_relationships),
                "entity_type_counts": entity_type_counts,
                "relationship_type_counts": relationship_type_counts,
            },
        }
        print("Resumen:")
        print(output_data["source_summary"])
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_data
