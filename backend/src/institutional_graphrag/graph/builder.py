from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from neo4j import GraphDatabase

from institutional_graphrag.graph.schema import (
    Entity,
    GraphSchema,
    Relationship,
    validate_relationship_endpoints,
)

logger = logging.getLogger("graph_ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


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

    def upsert_entities(self, entities: Iterable[Entity]) -> None:
        """Upsert entities usando batching correcto, por label."""
        entities_by_label: dict[str, list[Entity]] = {}
        for e in entities:
            entities_by_label.setdefault(e.label, []).append(e)

        for label, entities_label in entities_by_label.items():
            for batch in self._chunks(entities_label, self.batch_size):
                rows: list[dict] = []
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
                ON CREATE SET e.__created__ = true
                SET e += row
                RETURN
                sum(CASE WHEN e.__created__ = true THEN 1 ELSE 0 END) AS created,
                count(*) AS total
                """

                with self.driver.session() as session:
                    result = session.run(query, rows=rows).single()
                    if result is None:
                        created = 0
                        total = 0
                    else:
                        created = int(result["created"])
                        total = int(result["total"])
                    matched = total - created

                    # limpiar flag (solo para los que se crearon en este batch)
                session.run(
                    f"""
                MATCH (e:{label})
                WHERE e.__created__ = true
                REMOVE e.__created__
                """
                )
                logger.info(
                    "Neo4j ENTIDADES label=%s total=%d creadas=%d ya_existian=%d",
                    label,
                    total,
                    created,
                    matched,
                )

    # -------------------------
    # Relaciones
    # -------------------------
    def upsert_relationships(
        self,
        relationships: Iterable[Tuple[Relationship, Entity, Entity]],
    ) -> None:
        # 1) Filtrado + dedupe global
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

        # 2) batching + agrupación por (rel_type, src_label, tgt_label)
        for batch in self._chunks(deduped, self.batch_size):
            groups: Dict[Tuple[str, str, str], List[Dict]] = {}

            for rel, src, tgt in batch:
                gkey = (rel.type, src.label, tgt.label)
                groups.setdefault(gkey, []).append(
                    {
                        "source_id": src.id,
                        "target_id": tgt.id,
                        "properties": rel.properties or {},
                    }
                )

            with self.driver.session() as session:
                for (rel_type, src_label, tgt_label), rows in groups.items():
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (s:{src_label} {{id: row.source_id}})
                    MATCH (t:{tgt_label} {{id: row.target_id}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    ON CREATE SET r.__created__ = true
                    SET r += row.properties
                    RETURN
                    sum(CASE WHEN r.__created__ = true THEN 1 ELSE 0 END) AS created,
                    count(*) AS total
                    """

                    result = session.run(query, rows=rows).single()
                    if result is None:
                        created = 0
                        total = 0
                    else:
                        created = int(result["created"])
                        total = int(result["total"])
                    matched = total - created

                    # limpiar flag
                    session.run(
                        f"""
                    MATCH ()-[r:{rel_type}]->()
                    WHERE r.__created__ = true
                    REMOVE r.__created__
                    """
                    )

                    logger.info(
                        "Neo4j RELACIONES type=%s (%s->%s) total=%d creadas=%d ya_existian=%d",
                        rel_type,
                        src_label,
                        tgt_label,
                        total,
                        created,
                        matched,
                    )

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
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
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
        entities_list = list(entities)
        relationships_list = list(relationships)

        logger.info(
            "INGEST inicio: entidades=%d relaciones=%d",
            len(entities_list),
            len(relationships_list),
        )
        self.backend.upsert_entities(entities_list)
        self.backend.upsert_relationships(relationships_list)


# ============================================================
# Funciones auxiliares
# ============================================================


def load_graph_json(
    json_path: str | Path,
) -> Tuple[list[Entity], list[Tuple[Relationship, Entity, Entity]], list[dict[str, Any]]]:
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    entities_by_id: dict[str, Entity] = {}
    seen_labels_by_id: dict[str, set[str]] = {}

    # Política simple: "primero gana" (first wins)
    # Si preferís "último gana" (last wins), te dejo más abajo.
    for raw in payload.get("entities", []):
        entity_label = raw["label"]
        entity_id = raw["id"]
        value = raw.get("value")

        seen_labels_by_id.setdefault(entity_id, set()).add(entity_label)

        # Si ya existe, no reemplazamos: nos quedamos con 1 sola entidad
        if entity_id in entities_by_id:
            prev = entities_by_id[entity_id]
            if prev.label != entity_label:
                logger.error(
                    "ID duplicado con distinto label: id=%s (keep=%s, drop=%s)",
                    entity_id,
                    prev.label,
                    entity_label,
                )
            else:
                logger.warning(
                    "Entidad duplicada: id=%s label=%s (se ignora la repetida)",
                    entity_id,
                    entity_label,
                )
            continue

        entity_cls = GraphSchema.get_entity_class(entity_label)
        entities_by_id[entity_id] = entity_cls(id=entity_id, value=value)

    # (Opcional) resumen final de duplicados
    dup_diff = {eid: labels for eid, labels in seen_labels_by_id.items() if len(labels) > 1}
    if dup_diff:
        logger.error(
            "Se detectaron %d IDs con múltiples labels. Se guardó solo 1 entidad por id.",
            len(dup_diff),
        )

    inv_remap = build_containment_remap(entities_by_id)

    if inv_remap:
        # Eliminamos las entidades Investigador "contenidas"
        for drop_id in inv_remap.keys():
            # por seguridad, solo borramos si sigue siendo Investigador
            e = entities_by_id.get(drop_id)
            if e is not None and e.label == "Investigador":
                del entities_by_id[drop_id]

        logger.warning(
            "Investigador containment: eliminados=%d (se redirigen relaciones)", len(inv_remap)
        )

    # Relationships (igual que antes)
    relationships: list[Tuple[Relationship, Entity, Entity]] = []
    for raw in payload.get("relationships", []):
        rel_type = raw["type"]
        source_id = inv_remap.get(raw["source_id"], raw["source_id"])
        target_id = inv_remap.get(raw["target_id"], raw["target_id"])

        properties = raw.get("properties") or {}

        rel_factory = GraphSchema.get_relationship_factory(rel_type)
        rel = rel_factory(source_id, target_id, properties)

        src = entities_by_id.get(source_id)
        if src is None:
            logger.error("Relación %s: source_id no existe: %s (se omite)", rel_type, source_id)
            continue

        tgt = entities_by_id.get(target_id)
        if tgt is None:
            logger.error("Relación %s: target_id no existe: %s (se omite)", rel_type, target_id)
            continue

        relationships.append((rel, src, tgt))

    raw_errors = payload.get("errors", [])
    errors: list[dict[str, Any]] = raw_errors if isinstance(raw_errors, list) else []
    errors = [e for e in errors if isinstance(e, dict)]

    # Logs de conteos
    raw_entities_n = len(payload.get("entities", []))
    raw_rels_n = len(payload.get("relationships", []))

    logger.info("JSON: entidades leídas=%d", raw_entities_n)
    logger.info("JSON: relaciones leídas=%d", raw_rels_n)
    logger.info("JSON: errores leídos=%d", len(errors))

    logger.info("Post-dedupe: entidades finales=%d", len(entities_by_id))
    logger.info("Post-dedupe/remap: relaciones finales=%d", len(relationships))
    return list(entities_by_id.values()), relationships, errors


def build_containment_remap(entities_by_id: dict[str, Entity]) -> dict[str, str]:

    inv_ids = [eid for eid, e in entities_by_id.items() if e.label == "Investigador"]

    # Ordenamos por largo DESC: primero los más largos (candidatos a quedar)
    inv_ids_sorted = sorted(inv_ids, key=len, reverse=True)

    kept: list[str] = []
    remap: dict[str, str] = {}

    for cand in inv_ids_sorted:
        # Si ya está “contenido” en alguno que quedó, lo dropeamos
        container = next((k for k in kept if cand != k and cand in k), None)
        if container:
            remap[cand] = container
            logger.warning(
                "Investigador id contenido (se elimina): drop=%s keep=%s", cand, container
            )
        else:
            kept.append(cand)

    return remap
