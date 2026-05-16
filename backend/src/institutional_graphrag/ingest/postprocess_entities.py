"""Post-procesar entidades y relaciones para mejorar calidad."""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("graph_ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class Postprocessor:
    # -------------------------
    # Helpers de nombres
    # -------------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalizar nombre a minúsculas, eliminar tildes y limpiar espacios."""
        normalized = name.strip().lower()

        normalized = "".join(
            c
            for c in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(c) != "Mn" or c == "̃"
        )
        normalized = unicodedata.normalize("NFC", normalized)

        normalized = re.sub(r"\s+", " ", normalized)

        titles = [
            r"\bDR\.\s*",
            r"\bDRA\.\s*",
            r"\bDOCTOR\s+",
            r"\bDOCTORA\s+",
            r"\bPROF\.\s*",
            r"\bPROFESORA?\s+",
            r"\bLIC\.\s*",
            r"\bLICENCIADO/A\s+",
            r"\bING\.\s*",
            r"\bINGENIERO/A\s+",
            r"\bMSC\.\s*",
            r"\bPHD\.\s*",
        ]
        for title_pattern in titles:
            normalized = re.sub(title_pattern, "", normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def is_garbage_researcher(name: str) -> bool:
        """Detectar nombres de investigadores que parecen basura/errores OCR."""
        garbage_patterns = [
            r"^[A-Z]+\s*=\s*[A-Z]+",
            r"^TGR\.",
            r"^[A-Z]+\s*[=.].*[A-Z]$",
            r"^[A-Z]\s*[A-Z]\s*[A-Z]$",
        ]
        return any(re.search(p, name) for p in garbage_patterns)

    # -------------------------
    # Evidencias genéricas
    # -------------------------
    @staticmethod
    def is_generic_evidence(evidence_text: str) -> bool:
        if not evidence_text:
            return True
        evidence_lower = evidence_text.lower().strip()

        generic_patterns = [
            r"^mencionado en ",
            r"^investigador\s*$",
            r"^proyecto\s*$",
            r"^tópico\s*$",
            r"^colaborador\s*$",
        ]
        if any(re.search(p, evidence_lower) for p in generic_patterns):
            return True
        return len(evidence_lower) < 5

    # -------------------------
    # Paso grande: consolidación
    # -------------------------
    def consolidate_researchers(
        self, entities: List[dict], relationships: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Limpiar investigadores: quitar basura, normalizar nombres, deduplicar por ID
        exacto, y eliminar los que no tienen evidencia válida.
        """
        researchers = [e for e in entities if e.get("label") == "Investigador"]
        other_entities = [e for e in entities if e.get("label") != "Investigador"]

        print(f"\n[Consolidación] Procesando {len(researchers)} investigadores...")
        transformation_log: List[dict] = []

        # PASO 1: basura
        garbage_ids: Set[str] = set()
        garbage_changes = []
        for r in researchers:
            name = r["value"].get("name", "")
            if self.is_garbage_researcher(name):
                garbage_ids.add(r["id"])
                garbage_changes.append({"type": "garbage_removal", "id": r["id"], "name": name})
        researchers = [r for r in researchers if r["id"] not in garbage_ids]

        if garbage_changes:
            print(f"[Consolidación] Removidos {len(garbage_changes)} investigadores basura")
            transformation_log.append(
                {
                    "step": "Eliminación de investigadores basura",
                    "count": len(garbage_changes),
                    "changes": garbage_changes[:20],
                }
            )

        # PASO 2: normalizar nombres (in-place)
        normalization_changes = []
        for r in researchers:
            original = r["value"].get("name", "")

            if "display_name" not in r["value"]:
                r["value"]["display_name"] = original.strip().title()

            normalized = self.normalize_name(original)
            if original != normalized:
                normalization_changes.append(
                    {
                        "type": "normalization",
                        "id": r["id"],
                        "original": original,
                        "normalized": normalized,
                    }
                )
            r["value"]["name"] = normalized

        if normalization_changes:
            transformation_log.append(
                {
                    "step": "Normalización a minúsculas",
                    "count": len(normalization_changes),
                    "changes": normalization_changes[:20],
                }
            )

        # PASO 3: deduplicar por ID exacto (mantener el primero visto)
        seen_ids: Set[str] = set()
        dedup_changes = []
        deduped: List[dict] = []
        for r in researchers:
            if r["id"] in seen_ids:
                dedup_changes.append({"type": "dedup", "id": r["id"]})
            else:
                seen_ids.add(r["id"])
                deduped.append(r)
        researchers = deduped

        if dedup_changes:
            print(f"[Consolidación] {len(dedup_changes)} investigadores duplicados por ID eliminados")
            transformation_log.append(
                {
                    "step": "Deduplicación por ID",
                    "count": len(dedup_changes),
                    "changes": dedup_changes[:20],
                }
            )

        # PASO 4: filtrar relaciones duplicadas y genéricas; recolectar IDs con evidencia válida
        seen_relationships: Set[tuple] = set()
        updated_relationships: List[dict] = []
        researchers_with_valid_rels: Set[str] = set()
        duplicate_rels_removed = 0
        generic_rels_removed = 0

        for rel in relationships:
            if rel.get("type") == "EXTRAIDO_DE":
                evidence = rel.get("properties", {}).get("evidence_text", "")
                if self.is_generic_evidence(evidence):
                    generic_rels_removed += 1
                    continue
                target_id = rel.get("target_id")
                if isinstance(target_id, str):
                    researchers_with_valid_rels.add(target_id)

            props = rel.get("properties") or {}
            rel_key = (rel.get("source_id"), rel.get("target_id"), rel.get("type"), props.get("calidad"))
            if rel_key in seen_relationships:
                duplicate_rels_removed += 1
                continue
            seen_relationships.add(rel_key)
            updated_relationships.append(rel)

        print(f"[Consolidación] {duplicate_rels_removed} relaciones duplicadas eliminadas")
        print(f"[Consolidación] {generic_rels_removed} relaciones EXTRAIDO_DE genéricas eliminadas")

        if duplicate_rels_removed + generic_rels_removed > 0:
            transformation_log.append(
                {
                    "step": "Actualización y filtrado de relaciones",
                    "count": duplicate_rels_removed + generic_rels_removed,
                    "duplicate_relationships_removed": duplicate_rels_removed,
                    "generic_relationships_removed": generic_rels_removed,
                }
            )

        # PASO 5: eliminar investigadores sin relaciones válidas
        to_remove = [r["id"] for r in researchers if r["id"] not in researchers_with_valid_rels]
        researchers = [r for r in researchers if r["id"] in researchers_with_valid_rels]

        if to_remove:
            print(
                f"[Consolidación] {len(to_remove)} investigadores sin relaciones válidas eliminados"
            )
            transformation_log.append(
                {
                    "step": "Eliminación de investigadores sin relaciones válidas",
                    "count": len(to_remove),
                    "researcher_ids": to_remove[:20],
                }
            )

        final_entities = other_entities + researchers
        return final_entities, updated_relationships, transformation_log

    def add_missing_evidence_text(self, relationships: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Descarta EXTRAIDO_DE sin evidence_text real."""
        updated: List[dict] = []
        filtered_rels: List[dict] = []
        filtered_count = 0

        for rel in relationships:
            if rel.get("type") == "EXTRAIDO_DE":
                evidence = rel.get("properties", {}).get("evidence_text", "")
                if not evidence or self.is_generic_evidence(evidence):
                    filtered_count += 1
                    filtered_rels.append(
                        {
                            "source_id": rel.get("source_id"),
                            "target_id": rel.get("target_id"),
                            "evidence": evidence,
                        }
                    )
                    continue
            updated.append(rel)

        print(
            f"[Evidence Filter] {filtered_count} relaciones EXTRAIDO_DE sin evidence válido descartadas"
        )

        transformation_log: List[dict] = []
        if filtered_count > 0:
            transformation_log.append(
                {
                    "step": "Filtrado de EXTRAIDO_DE genéricas",
                    "count": filtered_count,
                    "sample_filtered": filtered_rels[:10],
                }
            )
        return updated, transformation_log

    # -------------------------
    # API principal
    # -------------------------
    def postprocess_payload(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Recibe dict con {entities: [...], relationships: [...]}
        Devuelve (data_actualizado, logs)
        """
        entities: list[dict[str, Any]] = data.get("entities", [])
        relationships: list[dict[str, Any]] = data.get("relationships", [])

        full_log: dict[str, Any] = {
            "timestamp": logging.Formatter().formatTime(
                logging.LogRecord("", 0, "", 0, "", (), None)
            ),
            "original_counts": {"entities": len(entities), "relationships": len(relationships)},
            "transformations": [],
        }

        entities, relationships, consolidation_log = self.consolidate_researchers(
            entities, relationships
        )
        full_log["transformations"].extend(consolidation_log)

        relationships, evidence_log = self.add_missing_evidence_text(relationships)
        full_log["transformations"].extend(evidence_log)

        data["entities"] = entities
        data["relationships"] = relationships
        full_log["final_counts"] = {"entities": len(entities), "relationships": len(relationships)}
        full_log["summary"] = {
            "entities_removed": len(data.get("entities", [])) - len(entities),
            "entities_kept": len(entities),
        }
        return data, full_log

    def postprocess_file(self, json_path: Optional[Path] = None) -> None:
        """Carga JSON, aplica postprocess y guarda."""
        if json_path is None:
            json_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / "entities_relations"
                / "entity_documents.json"
            )

        print(f"\n[Postprocess] Cargando {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_entities = len(data.get("entities", []))
        original_relationships = len(data.get("relationships", []))
        print(
            f"[Postprocess] Original: {original_entities} entidades, {original_relationships} relaciones"
        )

        data, full_log = self.postprocess_payload(data)

        print(
            f"\n[Postprocess] Final: {len(data['entities'])} entidades, {len(data['relationships'])} relaciones"
        )
        print(
            f"[Postprocess] Reducción: {original_entities - len(data['entities'])} entidades, "
            f"{original_relationships - len(data['relationships'])} relaciones"
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Guardado en: {json_path}")

        summary_path = json_path.parent / f"{json_path.stem}_postprocess_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMEN DE POST-PROCESAMIENTO\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Archivo procesado: {json_path.name}\n")
            f.write(f"Fecha: {full_log['timestamp']}\n\n")
            f.write("CONTEOS:\n")
            f.write(f"  Entidades originales: {original_entities}\n")
            f.write(f"  Entidades finales: {len(data['entities'])}\n")
            f.write(f"  Relaciones: {len(data['relationships'])}\n\n")

            for transformation in full_log["transformations"]:
                f.write("-" * 80 + "\n")
                f.write(f"PASO: {transformation['step']}\n")
                f.write(f"Cambios: {transformation['count']}\n\n")

        print(f"[OK] Resumen legible guardado en: {summary_path}")
