"""Post-procesar entidades y relaciones para mejorar calidad."""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import re


def normalize_name(name: str) -> str:
    """Normalizar nombre a mayúsculas y limpiar."""
    return name.strip().upper()


def is_partial_name(name1: str, name2: str) -> bool:
    """Verificar si name1 es una versión parcial (solo apellido) de name2."""
    n1_parts = set(name1.split())
    n2_parts = set(name2.split())
    
    # Si name1 es subconjunto de name2, es parcial
    if n1_parts.issubset(n2_parts) and len(n1_parts) < len(n2_parts):
        return True
    
    return False


def consolidate_researchers(entities: List[dict], relationships: List[dict]) -> tuple[List[dict], List[dict], List[dict]]:
    """Consolidar investigadores con nombres parciales/duplicados.
    
    Returns:
        tuple: (entities, relationships, transformation_log)
    """
    
    # Separar investigadores de otras entidades
    researchers = [e for e in entities if e.get("label") == "Investigador"]
    other_entities = [e for e in entities if e.get("label") != "Investigador"]
    
    print(f"\n[Consolidación] Procesando {len(researchers)} investigadores...")
    
    transformation_log = []
    normalization_changes = []
    
    # Normalizar nombres a mayúsculas
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
            "changes": normalization_changes
        })
    
    # Agrupar por palabras clave (apellidos comunes)
    groups = defaultdict(list)
    for researcher in researchers:
        words = researcher["value"].split()
        # Usar último apellido como clave
        key = words[-1] if words else researcher["value"]
        groups[key].append(researcher)
    
    # Encontrar duplicados/parciales y crear mapeo
    id_mapping = {}  # old_id -> new_id
    consolidated = []
    merged_count = 0
    merge_changes = []
    
    for key, group in groups.items():
        if len(group) == 1:
            consolidated.append(group[0])
            continue
        
        # Ordenar por longitud del nombre (más completo primero)
        group.sort(key=lambda x: len(x["value"]), reverse=True)
        
        # El más completo es el canónico
        canonical = group[0]
        consolidated.append(canonical)
        
        # Mapear los demás al canónico
        for researcher in group[1:]:
            if is_partial_name(researcher["value"], canonical["value"]):
                id_mapping[researcher["id"]] = canonical["id"]
                merged_count += 1
                merge_changes.append({
                    "type": "merge",
                    "partial_id": researcher["id"],
                    "partial_name": researcher["value"],
                    "canonical_id": canonical["id"],
                    "canonical_name": canonical["value"]
                })
                print(f"  Merging '{researcher['value']}' -> '{canonical['value']}'")
            else:
                # No es partial, mantener ambos
                consolidated.append(researcher)
    
    print(f"[Consolidación] {merged_count} investigadores consolidados")
    
    if merge_changes:
        transformation_log.append({
            "step": "Consolidación de nombres parciales",
            "count": merged_count,
            "changes": merge_changes
        })
    
    # Actualizar relaciones con el mapeo
    updated_relationships = []
    relationship_updates = []
    for rel in relationships:
        rel_copy = rel.copy()
        changed = False
        
        # Actualizar source_id si está en el mapeo
        if rel_copy.get("source_id") in id_mapping:
            old_source = rel_copy["source_id"]
            rel_copy["source_id"] = id_mapping[rel_copy["source_id"]]
            changed = True
        
        # Actualizar target_id si está en el mapeo
        if rel_copy.get("target_id") in id_mapping:
            old_target = rel_copy.get("target_id")
            rel_copy["target_id"] = id_mapping[rel_copy["target_id"]]
            changed = True
        
        if changed:
            relationship_updates.append({
                "type": "relationship_update",
                "relationship_type": rel_copy.get("type"),
                "old_source": rel.get("source_id"),
                "new_source": rel_copy.get("source_id"),
                "old_target": rel.get("target_id"),
                "new_target": rel_copy.get("target_id")
            })
        
        updated_relationships.append(rel_copy)
    
    if relationship_updates:
        transformation_log.append({
            "step": "Actualización de relaciones",
            "count": len(relationship_updates),
            "sample_changes": relationship_updates[:10]  # Solo primeros 10 como muestra
        })
    
    # Combinar con otras entidades
    final_entities = other_entities + consolidated
    
    return final_entities, updated_relationships, transformation_log


def add_missing_evidence_text(relationships: List[dict]) -> tuple[List[dict], List[dict]]:
    """Agregar evidence_text faltante en relaciones EVIDENCIA_DE.
    
    Returns:
        tuple: (relationships, transformation_log)
    """
    
    updated = []
    fixed_count = 0
    transformation_log = []
    evidence_fixes = []
    
    for rel in relationships:
        rel_copy = rel.copy()
        
        if rel_copy.get("type") == "EVIDENCIA_DE":
            props = rel_copy.get("properties", {})
            
            # Si no tiene evidence_text, agregarlo
            if not props.get("evidence_text"):
                source_id = rel_copy.get("source_id", "unknown")
                target_id = rel_copy.get("target_id", "unknown")
                
                # Generar evidence_text genérico
                props["evidence_text"] = f"Mencionado en {source_id}"
                rel_copy["properties"] = props
                fixed_count += 1
                evidence_fixes.append({
                    "type": "evidence_fix",
                    "source_id": source_id,
                    "target_id": target_id,
                    "added_text": props["evidence_text"]
                })
        
        updated.append(rel_copy)
    
    print(f"[Evidence Fix] {fixed_count} relaciones sin evidence_text corregidas")
    
    if evidence_fixes:
        transformation_log.append({
            "step": "Agregar evidence_text faltante",
            "count": fixed_count,
            "sample_changes": evidence_fixes[:10]  # Solo primeros 10 como muestra
        })
    
    return updated, transformation_log



def postprocess_entities():
    """Ejecutar post-procesamiento completo."""
    
    json_path = Path(__file__).parent.parent.parent / "data" / "entities_relations" / "entity_documents.json"
    
    print(f"[Postprocess] Cargando {json_path}...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_entities = len(data["entities"])
    original_relationships = len(data["relationships"])
    
    print(f"[Postprocess] Original: {original_entities} entidades, {original_relationships} relaciones")
    
    # 1. Consolidar investigadores
    entities, relationships, consolidation_log = consolidate_researchers(
        data["entities"],
        data["relationships"]
    )
    
    # 2. Agregar evidence_text faltante
    relationships, evidence_log = add_missing_evidence_text(relationships)
    
    # Combinar logs
    all_transformations = consolidation_log + evidence_log

    
    # Actualizar data
    data["entities"] = entities
    data["relationships"] = relationships
    
    print(f"\n[Postprocess] Final: {len(entities)} entidades, {len(relationships)} relaciones")
    print(f"[Postprocess] Reducción: {original_entities - len(entities)} entidades eliminadas")
    
    # Guardar
    output_path = json_path.parent / "entity_documents_postprocessed.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Guardado en: {output_path}")


if __name__ == "__main__":
    postprocess_entities()
