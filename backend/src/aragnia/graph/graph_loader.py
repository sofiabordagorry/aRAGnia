from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Tuple

from aragnia.graph.builder import GraphBuilder
from aragnia.graph.schema import Entity, GraphSchema, Relationship

logger = logging.getLogger("graph_ingest")


def _read_json(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("El JSON raíz debe ser un objeto.")

    return payload


def _parse_entities(payload: dict[str, Any]) -> dict[str, Entity]:
    entities_by_id: dict[str, Entity] = {}
    seen_labels_by_id: dict[str, set[str]] = {}

    for raw in payload.get("entities", []):
        entity_label = raw["label"]
        entity_id = raw["id"]
        value = raw.get("value")

        seen_labels_by_id.setdefault(entity_id, set()).add(entity_label)

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

    dup_diff = {eid: labels for eid, labels in seen_labels_by_id.items() if len(labels) > 1}
    if dup_diff:
        logger.error(
            "Se detectaron %d IDs con múltiples labels. Se guardó solo 1 entidad por id.",
            len(dup_diff),
        )

    return entities_by_id


def _resolve_entity(
    *,
    entity_id: str,
    entities_by_id: dict[str, Entity],
    database: GraphBuilder,
) -> Entity | None:
    entity = entities_by_id.get(entity_id)
    if entity is not None:
        return entity

    db_candidates = database.fetch_entity_by_id(entity_id)
    if len(db_candidates) == 1:
        return db_candidates[0]

    return None


def _parse_relationships(
    *,
    payload: dict[str, Any],
    entities_by_id: dict[str, Entity],
    database: GraphBuilder,
) -> list[tuple[Relationship, Entity, Entity]]:
    relationships: list[tuple[Relationship, Entity, Entity]] = []

    for raw in payload.get("relationships", []):
        rel_type = raw["type"]
        source_id = raw["source_id"]
        target_id = raw["target_id"]
        properties = raw.get("properties") or {}

        rel_factory = GraphSchema.get_relationship_factory(rel_type)
        rel = rel_factory(source_id, target_id, properties)

        src = _resolve_entity(
            entity_id=source_id,
            entities_by_id=entities_by_id,
            database=database,
        )
        if src is None:
            db_candidates = database.fetch_entity_by_id(source_id)
            logger.error(
                "Relación %s: source_id no existe en batch y en DB es %s: %s (se omite)",
                rel_type,
                "inexistente" if not db_candidates else "ambiguo",
                source_id,
            )
            continue

        tgt = _resolve_entity(
            entity_id=target_id,
            entities_by_id=entities_by_id,
            database=database,
        )
        if tgt is None:
            db_candidates = database.fetch_entity_by_id(target_id)
            logger.error(
                "Relación %s: target_id no existe en batch y en DB es %s: %s (se omite)",
                rel_type,
                "inexistente" if not db_candidates else "ambiguo",
                target_id,
            )
            continue

        relationships.append((rel, src, tgt))

    return relationships


def _extract_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list):
        return []
    return [err for err in raw_errors if isinstance(err, dict)]


def _log_counts(
    *,
    payload: dict[str, Any],
    entities_by_id: dict[str, Entity],
    relationships: list[tuple[Relationship, Entity, Entity]],
    errors: list[dict[str, Any]],
) -> None:
    raw_entities_n = len(payload.get("entities", []))
    raw_rels_n = len(payload.get("relationships", []))

    logger.info("JSON: entidades leídas=%d", raw_entities_n)
    logger.info("JSON: relaciones leídas=%d", raw_rels_n)
    logger.info("JSON: errores leídos=%d", len(errors))

    logger.info("Post-dedupe: entidades finales=%d", len(entities_by_id))
    logger.info("Post-dedupe/remap: relaciones finales=%d", len(relationships))


def load_graph_json(
    json_path: str | Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> Tuple[list[Entity], list[tuple[Relationship, Entity, Entity]]]:
    payload = _read_json(json_path)
    entities_by_id = _parse_entities(payload)

    database = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    try:
        relationships = _parse_relationships(
            payload=payload,
            entities_by_id=entities_by_id,
            database=database,
        )

        errors = _extract_errors(payload)
        _log_counts(
            payload=payload,
            entities_by_id=entities_by_id,
            relationships=relationships,
            errors=errors,
        )

        return list(entities_by_id.values()), relationships
    finally:
        database.close()
