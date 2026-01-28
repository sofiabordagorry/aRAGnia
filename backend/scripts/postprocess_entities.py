"""Post-procesar entidades y relaciones para mejorar calidad."""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import re
from difflib import SequenceMatcher


def normalize_name(name: str) -> str:
    """Normalizar nombre a mayúsculas y limpiar espacios."""
    normalized = re.sub(r'\s+', ' ', name.strip().upper())
    
    # Eliminar títulos académicos/profesionales
    titles = [
        r'\bDR\.\s*', r'\bDRA\.\s*', r'\bDOCTOR\s+', r'\bDOCTORA\s+',
        r'\bPROF\.\s*', r'\bPROFESORA?\s+',
        r'\bLIC\.\s*', r'\bLICENCIADO/A\s+',
        r'\bING\.\s*', r'\bINGENIERO/A\s+',
        r'\bMSC\.\s*', r'\bPHD\.\s*',
    ]
    
    for title_pattern in titles:
        normalized = re.sub(title_pattern, '', normalized, flags=re.IGNORECASE)
    
    # Limpiar espacios múltiples resultantes
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def is_garbage_researcher(name: str) -> bool:
    """Detectar nombres de investigadores que parecen basura/errores OCR.
    
    Ejemplos:
    - "SI C = NFX"
    - "TGR., = X. IV)"
    - Nombres muy cortos con caracteres especiales
    - Solo acrónimos
    """
    garbage_patterns = [
        r'^[A-Z]+\s*=\s*[A-Z]+',  # "SI C = NFX"
        r'^TGR\.',  # "TGR., = ..."
        r'^[A-Z]+\s*[=.].*[A-Z]$',  # Formato extraño con =
        r'^[A-Z]\s*[A-Z]\s*[A-Z]$',  # Solo 3 iniciales: "N. B. C"
    ]
    
    for pattern in garbage_patterns:
        if re.search(pattern, name):
            return True
    
    return False


def name_similarity(name1: str, name2: str) -> float:
    """Calcular similitud entre dos nombres (0-1)."""
    return SequenceMatcher(None, name1, name2).ratio()


def is_partial_name(name1: str, name2: str) -> bool:
    """Verificar si name1 es una versión parcial de name2.
    
    Ejemplos:
    - "SERRA" es parcial de "GLORIA SERRA" 
    - "MAHLER" es parcial de "GRACIELA MAHLER"
    """
    # Normalizar ambos (eliminar títulos)
    n1_normalized = normalize_name(name1)
    n2_normalized = normalize_name(name2)
    
    n1_parts = set(n1_normalized.split())
    n2_parts = set(n2_normalized.split())
    
    # Si name1 es subconjunto de name2, es parcial
    if n1_parts.issubset(n2_parts) and len(n1_parts) < len(n2_parts):
        return True
    
    # También verificar si comparten apellido (última palabra)
    if n1_normalized.split() and n2_normalized.split():
        last1 = n1_normalized.split()[-1]
        last2 = n2_normalized.split()[-1]
        
        # Si el apellido es igual y uno es más corto, es parcial
        if last1 == last2 and len(n1_parts) < len(n2_parts):
            return True
    
    return False


def find_researcher_duplicates(researchers: List[dict], similarity_threshold: float = 0.85) -> dict:
    """Detectar duplicados exactos y variantes de investigadores.
    
    Returns:
        dict: {canonical_id: (canonical_name, [list_of_duplicate_ids])}
    """
    processed = set()
    duplicate_groups = {}
    
    # Ordenar por longitud (más largo primero)
    sorted_researchers = sorted(researchers, key=lambda x: len(x["value"]), reverse=True)
    
    for i, researcher in enumerate(sorted_researchers):
        researcher_id = researcher["id"]
        researcher_name = researcher["value"]
        
        if researcher_id in processed:
            continue
        
        duplicates = []
        processed.add(researcher_id)
        
        # Buscar duplicados/variantes en los restantes
        for other in sorted_researchers[i+1:]:
            other_id = other["id"]
            other_name = other["value"]
            
            if other_id in processed:
                continue
            
            # Verificar exacto o muy similar
            norm1 = normalize_name(researcher_name)
            norm2 = normalize_name(other_name)
            
            if norm1 == norm2:
                # Duplicado exacto
                duplicates.append(other_id)
                processed.add(other_id)
            elif name_similarity(norm1, norm2) >= similarity_threshold:
                # Variante fuzzy (similar)
                tokens1 = set(norm1.split())
                tokens2 = set(norm2.split())
                common = tokens1 & tokens2
                
                # Debe compartir al menos 2 tokens
                if len(common) >= 2:
                    duplicates.append(other_id)
                    processed.add(other_id)
            elif is_partial_name(other_name, researcher_name):
                # El otro es una versión parcial (usar nombres ORIGINALES)
                duplicates.append(other_id)
                processed.add(other_id)
        
        if duplicates:
            duplicate_groups[researcher_id] = (researcher_name, duplicates)
    
    return duplicate_groups


def is_generic_evidence(evidence_text: str) -> bool:
    """Detectar si el evidence_text es genérico/vacío.
    
    Ejemplos de genéricos:
    - "Mencionado en..."
    - "investigador"
    - "proyecto"
    - Muy corto (< 5 caracteres)
    """
    if not evidence_text:
        return True
    
    evidence_lower = evidence_text.lower().strip()
    
    generic_patterns = [
        r'^mencionado en ',
        r'^investigador\s*$',
        r'^proyecto\s*$',
        r'^tópico\s*$',
        r'^colaborador\s*$',
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, evidence_lower):
            return True
    
    # Si es muy corto, probablemente es genérico
    if len(evidence_lower) < 5:
        return True
    
    return False


def consolidate_researchers(entities: List[dict], relationships: List[dict]) -> tuple[List[dict], List[dict], List[dict]]:
    """Consolidar investigadores: eliminar duplicados, variantes y basura.
    
    También elimina investigadores sin relaciones válidas.
    
    Returns:
        tuple: (entities, relationships, transformation_log)
    """
    
    # Separar investigadores de otras entidades
    researchers = [e for e in entities if e.get("label") == "Investigador"]
    other_entities = [e for e in entities if e.get("label") != "Investigador"]
    
    print(f"\n[Consolidación] Procesando {len(researchers)} investigadores...")
    
    transformation_log = []
    
    # --- PASO 1: Eliminar basura ---
    garbage_ids = set()
    garbage_changes = []
    
    for researcher in researchers:
        name = researcher["value"]
        if is_garbage_researcher(name):
            garbage_ids.add(researcher["id"])
            garbage_changes.append({
                "type": "garbage_removal",
                "id": researcher["id"],
                "name": name
            })
    
    researchers = [r for r in researchers if r["id"] not in garbage_ids]
    
    if garbage_changes:
        print(f"[Consolidación] Removidos {len(garbage_changes)} investigadores basura")
        transformation_log.append({
            "step": "Eliminación de investigadores basura",
            "count": len(garbage_changes),
            "changes": garbage_changes[:20]
        })
    
    # --- PASO 2: Normalizar nombres a mayúsculas ---
    normalization_changes = []
    
    for researcher in researchers:
        original_name = researcher["value"]
        normalized_name = normalize_name(researcher["value"])
        if original_name != normalized_name:
            normalization_changes.append({
                "type": "normalization",
                "id": researcher["id"],
                "original": original_name,
                "normalized": normalized_name
            })
        researcher["value"] = normalized_name
    
    if normalization_changes:
        transformation_log.append({
            "step": "Normalización a MAYÚSCULAS",
            "count": len(normalization_changes),
            "changes": normalization_changes[:20]
        })
    
    # --- PASO 3: Detectar duplicados ---
    duplicate_groups = find_researcher_duplicates(researchers, similarity_threshold=0.85)
    
    # --- PASO 4: Mapear IDs ---
    id_mapping = {}  # old_id -> canonical_id
    consolidated = []
    merge_changes = []
    
    processed_ids = set()
    
    for canonical_id, (canonical_name, duplicate_ids) in duplicate_groups.items():
        consolidated.append(next(r for r in researchers if r["id"] == canonical_id))
        processed_ids.add(canonical_id)
        
        for dup_id in duplicate_ids:
            dup_researcher = next(r for r in researchers if r["id"] == dup_id)
            id_mapping[dup_id] = canonical_id
            processed_ids.add(dup_id)
            
            merge_changes.append({
                "type": "merge",
                "duplicate_id": dup_id,
                "duplicate_name": dup_researcher["value"],
                "canonical_id": canonical_id,
                "canonical_name": canonical_name
            })
            print(f"  Consolidando '{dup_researcher['value']}' -> '{canonical_name}'")
    
    # Agregar investigadores sin duplicados
    for researcher in researchers:
        if researcher["id"] not in processed_ids:
            consolidated.append(researcher)
    
    total_merged = len(merge_changes)
    print(f"[Consolidación] {total_merged} investigadores consolidados")
    
    if merge_changes:
        transformation_log.append({
            "step": "Consolidación de duplicados",
            "count": total_merged,
            "changes": merge_changes[:30]
        })
    
    # --- PASO 5: Actualizar relaciones, eliminar duplicadas y genéricas ---
    updated_relationships = []
    seen_relationships = {}  # (source_id, target_id, type) -> True
    relationship_updates = []
    duplicate_rels_removed = 0
    generic_rels_removed = 0
    
    # También rastrear qué investigadores tienen relaciones VÁLIDAS
    researchers_with_valid_rels = set()
    
    for rel in relationships:
        rel_copy = rel.copy()
        changed = False
        
        # Actualizar IDs si están en el mapeo
        old_source = rel_copy.get("source_id")
        old_target = rel_copy.get("target_id")
        
        if old_source in id_mapping:
            rel_copy["source_id"] = id_mapping[old_source]
            changed = True
        
        if old_target in id_mapping:
            rel_copy["target_id"] = id_mapping[old_target]
            changed = True
        
        # FILTRO: Eliminar relaciones EVIDENCIA_DE con evidence_text genérico/vacío
        if rel_copy.get("type") == "EVIDENCIA_DE":
            evidence = rel_copy.get("properties", {}).get("evidence_text", "")
            if is_generic_evidence(evidence):
                # No agregar esta relación
                generic_rels_removed += 1
                continue
            else:
                # Registrar que este investigador tiene evidencia válida
                researchers_with_valid_rels.add(rel_copy.get("target_id"))
        
        # Detectar relaciones duplicadas
        rel_key = (
            rel_copy.get("source_id"),
            rel_copy.get("target_id"),
            rel_copy.get("type")
        )
        
        if rel_key in seen_relationships:
            # Relación duplicada, no agregarla
            duplicate_rels_removed += 1
            continue
        
        seen_relationships[rel_key] = True
        
        if changed:
            relationship_updates.append({
                "type": "relationship_update",
                "relationship_type": rel_copy.get("type"),
                "old_source": old_source,
                "new_source": rel_copy.get("source_id"),
                "old_target": old_target,
                "new_target": rel_copy.get("target_id")
            })
        
        updated_relationships.append(rel_copy)
    
    # --- PASO 6: Eliminar investigadores sin relaciones válidas ---
    researchers_without_valid_rels = []
    for researcher in consolidated:
        if researcher["id"] not in researchers_with_valid_rels:
            researchers_without_valid_rels.append(researcher["id"])
    
    consolidated = [r for r in consolidated if r["id"] in researchers_with_valid_rels]
    
    if researchers_without_valid_rels:
        print(f"[Consolidación] {len(researchers_without_valid_rels)} investigadores sin relaciones válidas eliminados")
        transformation_log.append({
            "step": "Eliminación de investigadores sin relaciones válidas",
            "count": len(researchers_without_valid_rels),
            "researcher_ids": researchers_without_valid_rels[:20]
        })
    
    total_rels_removed = duplicate_rels_removed + generic_rels_removed
    print(f"[Consolidación] {duplicate_rels_removed} relaciones duplicadas eliminadas")
    print(f"[Consolidación] {generic_rels_removed} relaciones EVIDENCIA_DE genéricas eliminadas")
    
    if relationship_updates or total_rels_removed > 0:
        transformation_log.append({
            "step": "Actualización y filtrado de relaciones",
            "count": len(relationship_updates) + total_rels_removed,
            "relationship_updates": len(relationship_updates),
            "duplicate_relationships_removed": duplicate_rels_removed,
            "generic_relationships_removed": generic_rels_removed,
            "sample_updates": relationship_updates[:20]
        })
    
    # Combinar con otras entidades
    final_entities = other_entities + consolidated
    
    return final_entities, updated_relationships, transformation_log


def add_missing_evidence_text(relationships: List[dict]) -> tuple[List[dict], List[dict]]:
    """Filtrar relaciones EVIDENCIA_DE sin evidence_text real.
    
    NO agrega relaciones genéricas. Si no tiene evidence_text válido, se ignora.
    
    Returns:
        tuple: (relationships, transformation_log)
    """
    
    updated = []
    filtered_count = 0
    transformation_log = []
    filtered_rels = []
    
    for rel in relationships:
        # Si es EVIDENCIA_DE sin evidence_text válido, NO agregarla
        if rel.get("type") == "EVIDENCIA_DE":
            evidence = rel.get("properties", {}).get("evidence_text", "")
            
            if not evidence or is_generic_evidence(evidence):
                # Filtrar esta relación
                filtered_count += 1
                filtered_rels.append({
                    "source_id": rel.get("source_id"),
                    "target_id": rel.get("target_id"),
                    "evidence": evidence
                })
                continue
        
        # Mantener todas las demás relaciones
        updated.append(rel)
    
    print(f"[Evidence Filter] {filtered_count} relaciones EVIDENCIA_DE sin evidence válido descartadas")
    
    if filtered_count > 0:
        transformation_log.append({
            "step": "Filtrado de EVIDENCIA_DE genéricas",
            "count": filtered_count,
            "sample_filtered": filtered_rels[:10]
        })
    
    return updated, transformation_log



def postprocess_entities(json_path: Path = None) -> None:
    """Post-procesar entidades y relaciones para mejorar calidad."""
    
    if json_path is None:
        json_path = Path(__file__).parent.parent.parent / "data" / "entities_relations" / "entity_documents.json"
    
    print(f"\n[Postprocess] Cargando {json_path}...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_entities = len(data["entities"])
    original_relationships = len(data["relationships"])
    
    print(f"[Postprocess] Original: {original_entities} entidades, {original_relationships} relaciones")
    
    # 1. Consolidar investigadores (limpieza completa)
    entities, relationships, consolidation_log = consolidate_researchers(
        data["entities"],
        data["relationships"]
    )
    
    # 2. Filtrar relaciones EVIDENCIA_DE genéricas (no agregar, descartar)
    relationships, filter_log = add_missing_evidence_text(relationships)
    
    # Actualizar data
    data["entities"] = entities
    data["relationships"] = relationships
    
    print(f"\n[Postprocess] Final: {len(entities)} entidades, {len(relationships)} relaciones")
    print(f"[Postprocess] Reducción: {original_entities - len(entities)} entidades, {original_relationships - len(relationships)} relaciones")
    
    # Guardar en el mismo archivo
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Guardado en: {json_path}")


if __name__ == "__main__":
    postprocess_entities()
