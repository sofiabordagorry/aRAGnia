from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
ENABLE_NON_EQUAL_NAME_UNIFICATION = False


# ============================================================
# Neo4j backend
# ============================================================
@dataclass
class Neo4jStats:
    nodes_created: int = 0
    nodes_deleted: int = 0
    relationships_created: int = 0
    relationships_deleted: int = 0
    properties_set: int = 0
    labels_added: int = 0
    labels_removed: int = 0

    # IDs (opcionales / sample)
    node_ids: List[str] = field(default_factory=list)  # tus ids lógicos (row.id)
    node_eids: List[str] = field(default_factory=list)  # elementId(e)
    rel_eids: List[str] = field(default_factory=list)  # elementId(r)
    rel_pairs: List[Tuple[str, str]] = field(default_factory=list)  # (source_id, target_id)

    def add_counters(self, counters) -> None:
        self.nodes_created += counters.nodes_created
        self.nodes_deleted += counters.nodes_deleted
        self.relationships_created += counters.relationships_created
        self.relationships_deleted += counters.relationships_deleted
        self.properties_set += counters.properties_set
        self.labels_added += counters.labels_added
        self.labels_removed += counters.labels_removed


class Neo4jGraphBuilder:
    """
    Builder de grafo usando Neo4j como backend.

    - Usa MERGE para evitar duplicación de nodos y relaciones
    - Valida esquema antes de persistir
    - Implementa batching para nodos y relaciones
    """

    def __init__(self, uri: str, user: str, password: str, batch_size: int = 500):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._ping(max_attempts=6, sleep_s=2)
        except AuthError:
            logger.error("Neo4j: credenciales inválidas")
            raise RuntimeError("No se pudo autenticar con Neo4j. Verificá usuario y contraseña.")

        except ServiceUnavailable:
            logger.error("Neo4j: servidor no disponible en %s", uri)
            raise RuntimeError("No se pudo conectar a Neo4j. ¿Está el contenedor corriendo?")

        except SessionExpired:
            logger.error("Neo4j: conexión expirada durante el handshake")
            raise RuntimeError("La conexión con Neo4j falló durante la inicialización.")
        self.batch_size = batch_size
        self._create_constraints()

    def _ping(self, max_attempts: int = 2, sleep_s: int = 2) -> None:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self.driver.session() as session:
                    session.run("RETURN 1").consume()
                logger.info("Neo4j listo (ping OK)")
                return
            except (ServiceUnavailable, SessionExpired, OSError) as e:
                last_err = e
                logger.warning("Neo4j no listo (ping %d/%d): %s", attempt, max_attempts, e)
                time.sleep(sleep_s)
            except AuthError:
                raise
        raise last_err  # type: ignore[misc]

    def close(self):
        self.driver.close()

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

    def upsert_entities(self, entities: Iterable[Entity], *, sample_ids: int = 50) -> Neo4jStats:
        """Upsert entities usando batching correcto, por label. Devuelve stats + sample de IDs."""
        stats = Neo4jStats()

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

                # Devuelve ids procesados (lógicos + elementId)
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
    def upsert_relationships(
        self,
        relationships: Iterable[Tuple[Relationship, Entity, Entity]],
        *,
        sample_ids: int = 50,
    ) -> Neo4jStats:
        stats = Neo4jStats()

        seen: set[tuple[str, str, str]] = set()
        deduped: list[Tuple[Relationship, Entity, Entity]] = []

        for rel, src, tgt in relationships:
            if src.id == tgt.id:
                logger.warning(
                    "Relación inválida (self-loop) se omite: %s %s -> %s", rel.type, src.id, tgt.id
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
                    RETURN collect(elementId(r)) AS rel_eids,
                        collect([row.source_id, row.target_id]) AS pairs
                    """

                    result = session.run(query, rows=rows)
                    record = result.single()
                    summary = result.consume()

                    stats.add_counters(summary.counters)

                    if record is not None:
                        self._extend_sample(stats.rel_eids, record["rel_eids"] or [], sample_ids)
                        pairs = record["pairs"] or []
                        pairs = [(p[0], p[1]) for p in pairs if isinstance(p, list) and len(p) == 2]
                        self._extend_sample(stats.rel_pairs, pairs, sample_ids)

        return stats

    def clear_graph(self):
        """
        Borra **todos los nodos y relaciones** del grafo.
        Úsalo con cuidado, es irreversible.
        """
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)
        print("Grafo borrado completamente.")

    def fetch_investigador_ids(self) -> list[str]:
        query = "MATCH (i:Investigador) RETURN i.id AS id"
        with self.driver.session() as session:
            rows = session.run(query).data()
        return [r["id"] for r in rows if r.get("id")]

    def merge_node_id(
        self,
        *,
        label: str,
        old_id: str,
        new_id: str,
        sample_ids: int = 50,
    ) -> Neo4jStats:
        if old_id == new_id:
            return Neo4jStats()

        stats = Neo4jStats()

        with self.driver.session() as session:
            # 0) asegurar A y B
            r = session.run(
                f"""
                MATCH (a:{label} {{id: $old_id}})
                MERGE (b:{label} {{id: $new_id}})
                RETURN elementId(a) AS a_eid, elementId(b) AS b_eid
                """,
                old_id=old_id,
                new_id=new_id,
            )
            rec = r.single()
            summ = r.consume()
            stats.add_counters(summ.counters)
            if rec:
                self._extend_sample(stats.node_eids, [rec["a_eid"], rec["b_eid"]], sample_ids)

            # merge props
            r = session.run(
                f"""
                MATCH (a:{label} {{id: $old_id}})
                MATCH (b:{label} {{id: $new_id}})
                SET b += a
                """,
                old_id=old_id,
                new_id=new_id,
            )
            stats.add_counters(r.consume().counters)

            rel_types = [
                row["type"]
                for row in session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType AS type RETURN type"
                ).data()
            ]

            for t in rel_types:
                # salientes
                r = session.run(
                    f"""
                    MATCH (a:{label} {{id: $old_id}})-[r:{t}]->(x)
                    MATCH (b:{label} {{id: $new_id}})
                    MERGE (b)-[r2:{t}]->(x)
                    SET r2 += properties(r)
                    DELETE r
                    RETURN collect(elementId(r2)) AS moved_rel_eids,
                        collect([b.id, x.id]) AS pairs
                    """,
                    old_id=old_id,
                    new_id=new_id,
                )
                rec = r.single()
                summ = r.consume()
                stats.add_counters(summ.counters)
                if rec:
                    self._extend_sample(stats.rel_eids, rec["moved_rel_eids"] or [], sample_ids)

                # entrantes
                r = session.run(
                    f"""
                    MATCH (y)-[r:{t}]->(a:{label} {{id: $old_id}})
                    MATCH (b:{label} {{id: $new_id}})
                    MERGE (y)-[r2:{t}]->(b)
                    SET r2 += properties(r)
                    DELETE r
                    RETURN collect(elementId(r2)) AS moved_rel_eids
                    """,
                    old_id=old_id,
                    new_id=new_id,
                )
                rec = r.single()
                summ = r.consume()
                stats.add_counters(summ.counters)
                if rec:
                    self._extend_sample(stats.rel_eids, rec["moved_rel_eids"] or [], sample_ids)

            # borrar A
            r = session.run(
                f"""
                MATCH (a:{label} {{id: $old_id}})
                DETACH DELETE a
                """,
                old_id=old_id,
            )
            stats.add_counters(r.consume().counters)

        return stats

    def _extend_sample(self, dst: List[Any], src: List[Any], sample_ids: int) -> None:
        if sample_ids <= 0:
            return
        remaining = sample_ids - len(dst)
        if remaining <= 0:
            return
        dst.extend(src[:remaining])

    def fetch_entity_by_id(self, entity_id: str) -> list[Entity]:
        """
        Busca un nodo por id en todos los labels del schema y devuelve entidades mínimas.
        (Si hay más de un label con el mismo id, devuelve varias.)
        """
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
        for lab in labels:
            if lab in GraphSchema.ENTITIES:
                cls = GraphSchema.get_entity_class(lab)
                out.append(cls(id=entity_id, value={}))  # value vacío (o {"source":"db"} si querés)
        return out


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
        """Inserta entidades y relaciones usando batching genérico."""
        ent_stats = self.backend.upsert_entities(entities, sample_ids=15)
        rel_stats = self.backend.upsert_relationships(relationships, sample_ids=15)

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
        self.backend.close()


# ============================================================
# Funciones auxiliares
# ============================================================


def load_graph_json(
    json_path: str | Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    enable_non_equal_name_unification: bool = ENABLE_NON_EQUAL_NAME_UNIFICATION,
) -> Tuple[list[Entity], list[Tuple[Relationship, Entity, Entity]], List[Tuple[str, str]]]:
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    entities_by_id: dict[str, Entity] = {}
    seen_labels_by_id: dict[str, set[str]] = {}

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

    database = Neo4jGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    common_remap: Dict[str, str] = {}

    if enable_non_equal_name_unification:
        inv_ids_db = database.fetch_investigador_ids()
        inv_ids = [eid for eid, e in entities_by_id.items() if e.label == "Investigador"]
        inv_remap, db_merge, _ = build_containment_plan(
            inv_ids_batch=inv_ids,
            inv_ids_db=inv_ids_db,
        )
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
        common_remap = {**inv_remap, **dict(db_merge)}
    else:
        inv_remap, db_merge = {}, []
    # Relationships
    relationships: list[Tuple[Relationship, Entity, Entity]] = []
    for raw in payload.get("relationships", []):
        rel_type = raw["type"]
        source_id = common_remap.get(raw["source_id"], raw["source_id"])
        target_id = common_remap.get(raw["target_id"], raw["target_id"])

        properties = raw.get("properties") or {}

        rel_factory = GraphSchema.get_relationship_factory(rel_type)
        rel = rel_factory(source_id, target_id, properties)

        src = entities_by_id.get(source_id)
        if src is None:
            db_candidates = database.fetch_entity_by_id(source_id)
            if len(db_candidates) == 1:
                src = db_candidates[0]
            else:
                logger.error(
                    "Relación %s: source_id no existe en batch y en DB es %s: %s (se omite)",
                    rel_type,
                    "inexistente" if not db_candidates else "ambiguo",
                    source_id,
                )
                continue

        tgt = entities_by_id.get(target_id)
        if tgt is None:
            db_candidates = database.fetch_entity_by_id(target_id)
            if len(db_candidates) == 1:
                tgt = db_candidates[0]
            else:
                logger.error(
                    "Relación %s: source_id no existe en batch y en DB es %s: %s (se omite)",
                    rel_type,
                    "inexistente" if not db_candidates else "ambiguo",
                    source_id,
                )
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
    return list(entities_by_id.values()), relationships, db_merge


def build_containment_plan(
    *,
    inv_ids_batch: List[str],
    inv_ids_db: List[str],
    enable_non_equal_name_unification: bool = ENABLE_NON_EQUAL_NAME_UNIFICATION,
) -> Tuple[Dict[str, str], List[Tuple[str, str]], List[str]]:
    """
    Devuelve:
      1) batch_remap: drop->keep donde drop NO está en la DB (se arregla en el batch)
      2) db_merge: [(old_id, new_id), ...] donde old_id SÍ está en la DB y debe migrar a new_id
      3) kept: lista de ids canónicos (los que "quedan")
    """
    if not enable_non_equal_name_unification:
        # Sin remapeos, sin merges, "kept" = ids únicos
        kept = sorted(set(inv_ids_batch) | set(inv_ids_db))
        return {}, [], kept

    db_set = set(inv_ids_db)
    batch_set = set(inv_ids_batch)

    # Ordenamos por largo DESC y sin repetidos
    all_ids_sorted = sorted(set(inv_ids_batch) | set(inv_ids_db), key=len, reverse=True)

    kept: List[str] = []
    remap_all: Dict[str, str] = {}

    # 1) Remap global (misma lógica que tenías)
    for cand in all_ids_sorted:
        container = next((k for k in kept if cand != k and cand in k), None)
        if container:
            remap_all[cand] = container
        else:
            kept.append(cand)

    # 2) Clasificación en dos “salidas”
    batch_remap: Dict[str, str] = {}
    db_merge: List[Tuple[str, str]] = []

    for drop, keep in remap_all.items():
        drop_in_db = drop in db_set
        keep_in_db = keep in db_set
        drop_in_batch = drop in batch_set
        keep_in_batch = keep in batch_set

        if drop_in_db:
            # Caso DB: el ID viejo existe en la DB.
            # Querés que sus relaciones pasen al "nuevo".
            if keep_in_db or keep_in_batch:
                db_merge.append((drop, keep))
                logger.warning("DB merge plan: old=%s -> new=%s", drop, keep)
            else:
                # keep no existe ni en DB ni en batch (raro)
                logger.warning("DB merge skip (keep inexistente): old=%s keep=%s", drop, keep)

        else:
            # Caso batch: drop no está en DB
            # Solo tiene sentido si drop viene en batch (si no, ni aparece)
            if drop_in_batch:
                batch_remap[drop] = keep
                logger.warning("Batch remap: drop=%s -> keep=%s", drop, keep)

    return batch_remap, db_merge, kept
