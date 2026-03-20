"""Post-procesar entidades y relaciones para mejorar calidad."""

import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("graph_ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class Postprocessor:
    """
    Postprocesador con módulo apagable.
    Si self.enable_researcher_consolidation == False, NO se hace la unificacion de Investigadores
    """

    def __init__(
        self,
        enable_researcher_consolidation: bool = False,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.enable_researcher_consolidation = enable_researcher_consolidation
        self.similarity_threshold = similarity_threshold

    # -------------------------
    # Helpers de nombres
    # -------------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalizar nombre a mayúsculas, eliminar tildes y limpiar espacios."""
        normalized = name.strip().upper()

        normalized = "".join(
            c for c in unicodedata.normalize("NFD", normalized) if unicodedata.category(c) != "Mn"
        )

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

    @staticmethod
    def name_similarity(name1: str, name2: str) -> float:
        return SequenceMatcher(None, name1, name2).ratio()

    def is_inverted_name(self, name1: str, name2: str) -> bool:
        n1 = self.normalize_name(name1).replace(",", " ")
        n2 = self.normalize_name(name2).replace(",", " ")

        tokens1 = n1.split()
        tokens2 = n2.split()

        if set(tokens1) != set(tokens2):
            return False
        return len(tokens1) == len(tokens2) and tokens1 != tokens2

    def is_partial_name(self, name1: str, name2: str) -> bool:
        """
        Verificar si name1 es una versión parcial de name2 con restricciones
        para evitar falsos positivos.
        """
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)

        n1_parts = n1.split()
        n2_parts = n2.split()

        # Caso 1: una palabra
        if len(n1_parts) == 1:
            return n1 in n2_parts

        # Caso 2: múltiples palabras, debe coincidir primer token
        if n1_parts[0] == n2_parts[0]:
            if set(n1_parts).issubset(set(n2_parts)) and len(n1_parts) < len(n2_parts):
                return True

        # Caso 3: iniciales
        if "." in n1 and len(n1_parts) >= 2:
            initial = n1_parts[0].replace(".", "")
            if len(initial) == 1 and len(n2_parts) >= len(n1_parts):
                if n2_parts[0].startswith(initial):
                    if set(n1_parts[1:]).issubset(set(n2_parts[1:])):
                        return True

        return False

    # -------------------------
    # Duplicados investigadores
    # -------------------------
    def find_researcher_duplicates(self, researchers: List[dict]) -> Tuple[
        Dict[str, Tuple[str, List[str]]],
        Dict[str, Tuple[str, List[str]]],
    ]:
        """
        Returns:
            {canonical_id: (canonical_name, [duplicate_ids])}
        """
        processed = set()
        duplicate_groups: Dict[str, Tuple[str, List[str]]] = {}
        duplicate_groups_relation: Dict[str, Tuple[str, List[str]]] = {}

        sorted_researchers = sorted(
            researchers, key=lambda x: len(x["value"].get("name", "")), reverse=True
        )

        for researcher in sorted_researchers:
            researcher_id = researcher["id"]
            researcher_name = researcher["value"].get("name", "")

            duplicates: List[str] = []
            duplicates_relation: List[str] = []
            processed.add(researcher_id)

            for other in sorted_researchers:
                other_id = other["id"]
                other_name = other["value"].get("name", "")

                norm1 = self.normalize_name(researcher_name)
                norm2 = self.normalize_name(other_name)

                if norm1 == norm2:
                    if other_id not in processed:
                        duplicates.append(other_id)
                        processed.add(other_id)
                elif self.is_inverted_name(researcher_name, other_name):
                    if self.enable_researcher_consolidation:
                        if other_id not in processed:
                            duplicates.append(other_id)
                            processed.add(other_id)
                    else:
                        duplicates_relation.append(other_id)

                elif self.name_similarity(norm1, norm2) >= self.similarity_threshold:
                    tokens1 = norm1.split()
                    tokens2 = norm2.split()
                    common = set(tokens1) & set(tokens2)

                    # restricciones contra falsos positivos
                    if len(common) >= 2 and tokens1 and tokens2 and tokens1[0] == tokens2[0]:
                        if self.enable_researcher_consolidation:
                            if other_id not in processed:
                                duplicates.append(other_id)
                                processed.add(other_id)
                        else:
                            duplicates_relation.append(other_id)

                elif self.is_partial_name(other_name, researcher_name) or self.is_partial_name(
                    researcher_name, other_name
                ):
                    if self.enable_researcher_consolidation:
                        if other_id not in processed:
                            duplicates.append(other_id)
                            processed.add(other_id)
                    else:
                        duplicates_relation.append(other_id)

            if duplicates:
                duplicate_groups[researcher_id] = (researcher_name, duplicates)

            if duplicates_relation:
                duplicate_groups_relation[researcher_id] = (researcher_name, duplicates_relation)
        return duplicate_groups, duplicate_groups_relation

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
        Consolidar investigadores: eliminar duplicados, variantes y basura.
        También elimina investigadores sin relaciones EXTRAIDO_DE válidas.
        """
        researchers = [e for e in entities if e.get("label") == "Investigador"]
        other_entities = [e for e in entities if e.get("label") != "Investigador"]

        print(f"\n[Consolidación] Procesando {len(researchers)} investigadores...")
        transformation_log: List[dict] = []

        # PASO 1: basura
        garbage_ids = set()
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
                    "step": "Normalización a MAYÚSCULAS",
                    "count": len(normalization_changes),
                    "changes": normalization_changes[:20],
                }
            )

        # PASO 3: duplicados
        duplicate_groups, duplicate_groups_relation = self.find_researcher_duplicates(researchers)

        # PASO 4: mapear IDs
        updated_relationships: List[dict] = []
        id_mapping: Dict[str, str] = {}
        consolidated: List[dict] = []
        merge_changes: List[dict] = []
        processed_ids: Set[str] = set()

        if not self.enable_researcher_consolidation:
            for canonical_id, (canonical_name, duplicate_ids) in duplicate_groups_relation.items():
                for dup_id in duplicate_ids:
                    dup_r = next(r for r in researchers if r["id"] == dup_id)
                    updated_relationships.append(
                        {
                            "source_id": canonical_id,
                            "target_id": dup_id,
                            "type": "POSIBLE_ALIAS",
                        }
                    )
                    print(
                        f" Creada Relacion Potencial Igualdad entre '{dup_r['value'].get('name')}' y '{canonical_name}'"
                    )

        for canonical_id, (canonical_name, duplicate_ids) in duplicate_groups.items():
            canonical_r = next(r for r in researchers if r["id"] == canonical_id)
            consolidated.append(next(r for r in researchers if r["id"] == canonical_id))
            processed_ids.add(canonical_id)

            for dup_id in duplicate_ids:
                dup_r = next(r for r in researchers if r["id"] == dup_id)
                id_mapping[dup_id] = canonical_id
                processed_ids.add(dup_id)
                # Si el investigador canonico y el duplicado tienen diferentes fuentes, se mantienen ambas
                old_source = dup_r["value"].get("source")
                new_source = canonical_r["value"].get("source")

                if (
                    isinstance(old_source, list)
                    or isinstance(new_source, list)
                    or (old_source and new_source and old_source != new_source)
                ):
                    canonical_r["value"]["source"] = ["rule_based", "llm"]

                merge_changes.append(
                    {
                        "type": "merge",
                        "duplicate_id": dup_id,
                        "duplicate_name": dup_r["value"].get("name"),
                        "canonical_id": canonical_id,
                        "canonical_name": canonical_name,
                    }
                )
                print(f"  Consolidando '{dup_r['value'].get('name')}' -> '{canonical_name}'")

        for r in researchers:
            if r["id"] not in processed_ids:
                consolidated.append(r)

        print(f"[Consolidación] {len(merge_changes)} investigadores consolidados")
        if merge_changes:
            transformation_log.append(
                {
                    "step": "Consolidación de duplicados",
                    "count": len(merge_changes),
                    "changes": merge_changes[:30],
                }
            )

        # PASO 5: actualizar relaciones y filtrar duplicadas / genéricas
        seen_relationships = set()
        relationship_updates: List[dict] = []
        duplicate_rels_removed = 0
        generic_rels_removed = 0

        researchers_with_valid_rels: Set[str] = set()

        for rel in relationships:
            rel_copy = rel.copy()
            changed = False

            old_source = rel_copy.get("source_id")
            old_target = rel_copy.get("target_id")

            if old_source in id_mapping:
                rel_copy["source_id"] = id_mapping[old_source]
                changed = True

            if old_target in id_mapping:
                rel_copy["target_id"] = id_mapping[old_target]
                changed = True

            if rel_copy.get("type") == "EXTRAIDO_DE":
                evidence = rel_copy.get("properties", {}).get("evidence_text", "")
                if self.is_generic_evidence(evidence):
                    generic_rels_removed += 1
                    continue

                target_id = rel_copy.get("target_id")
                if isinstance(target_id, str):
                    researchers_with_valid_rels.add(target_id)

            rel_key = (rel_copy.get("source_id"), rel_copy.get("target_id"), rel_copy.get("type"))
            if rel_key in seen_relationships:
                duplicate_rels_removed += 1
                continue
            seen_relationships.add(rel_key)

            if changed:
                relationship_updates.append(
                    {
                        "type": "relationship_update",
                        "relationship_type": rel_copy.get("type"),
                        "old_source": old_source,
                        "new_source": rel_copy.get("source_id"),
                        "old_target": old_target,
                        "new_target": rel_copy.get("target_id"),
                    }
                )

            updated_relationships.append(rel_copy)

        # PASO 6: eliminar investigadores sin relaciones válidas
        to_remove = [r["id"] for r in consolidated if r["id"] not in researchers_with_valid_rels]
        consolidated = [r for r in consolidated if r["id"] in researchers_with_valid_rels]

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

        print(f"[Consolidación] {duplicate_rels_removed} relaciones duplicadas eliminadas")
        print(f"[Consolidación] {generic_rels_removed} relaciones EXTRAIDO_DE genéricas eliminadas")

        if relationship_updates or (duplicate_rels_removed + generic_rels_removed) > 0:
            transformation_log.append(
                {
                    "step": "Actualización y filtrado de relaciones",
                    "count": len(relationship_updates)
                    + duplicate_rels_removed
                    + generic_rels_removed,
                    "relationship_updates": len(relationship_updates),
                    "duplicate_relationships_removed": duplicate_rels_removed,
                    "generic_relationships_removed": generic_rels_removed,
                    "sample_updates": relationship_updates[:20],
                }
            )

        final_entities = other_entities + consolidated
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
        """Carga JSON, aplica postprocess según flags, y guarda."""
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
            f"[Postprocess] Reducción: {original_entities - len(data['entities'])} entidades, {original_relationships - len(data['relationships'])} relaciones"
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Guardado en: {json_path}")

        # Generar resumen legible en texto
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
            f.write(
                f"  Entidades eliminadas (duplicados): {original_entities - len(data['entities'])}\n"
            )
            f.write(f"  Relaciones: {len(data['relationships'])}\n\n")

            for transformation in full_log["transformations"]:
                f.write("-" * 80 + "\n")
                f.write(f"PASO: {transformation['step']}\n")
                f.write(f"Cambios: {transformation['count']}\n\n")

                if transformation["step"] == "Normalización a MAYÚSCULAS":
                    f.write("Ejemplos de normalizaciones:\n")
                    for change in transformation["changes"][:10]:
                        f.write(f"  '{change['original']}' → '{change['normalized']}'\n")

                elif transformation["step"] == "Consolidación de nombres parciales":
                    f.write("Merges realizados:\n")
                    for change in transformation["changes"]:
                        f.write(f"  '{change['partial_name']}' → '{change['canonical_name']}'\n")
                        f.write(f"    (ID: {change['partial_id']} → {change['canonical_id']})\n")

                elif transformation["step"] == "Actualización de relaciones":
                    f.write(f"Relaciones actualizadas: {transformation['count']}\n")
                    f.write("(Ver log JSON completo para detalles)\n")

                elif transformation["step"] == "Agregar evidence_text faltante":
                    f.write(f"Relaciones corregidas: {transformation['count']}\n")
                    f.write("Ejemplos:\n")
                    for fix in transformation.get("sample", [])[:10]:
                        f.write(f"  {fix['relation']}\n")

                f.write("\n")

        print(f"[OK] Resumen legible guardado en: {summary_path}")
