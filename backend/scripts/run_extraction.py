# backend/scripts/run_extraction.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult

# Importar funciones de postprocesamiento
import sys
sys.path.insert(0, str(Path(__file__).parent))
from postprocess_entities import (
    consolidate_researchers,
    add_missing_evidence_text
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

LLM_PROVIDER = "ollama"  # "ollama", "groq", o "local"
LLM_MODEL = None  # None usa el default del provider

# Los tópicos disponibles están definidos en llm_extractor.py (DEFAULT_TOPICS)
# Para modificarlos, editá: backend/src/institutional_graphrag/extraction/llm_extractor.py

def normalize_result(res: ExtractionResult) -> dict:
    return {
        "entities": sorted(
            [e.to_dict() for e in res.entities],
            key=lambda x: (x["label"], x["id"]),
        ),
        "relationships": sorted(
            [r.to_dict() for r in res.relationships],
            key=lambda x: (x["type"], x["source_id"], x["target_id"]),
        ),
        "errors": sorted(
            res.errors,
            key=lambda x: (x.get("type", ""), x.get("message", "")),
        ),
    }


def postprocess_extraction(json_path: Path) -> None:
    """Post-procesar entidades y relaciones para mejorar calidad."""
    
    print(f"\n[Postprocess] Cargando {json_path}...")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_entities = len(data["entities"])
    original_relationships = len(data["relationships"])
    
    print(f"[Postprocess] Original: {original_entities} entidades, {original_relationships} relaciones")
    
    # Registro completo de transformaciones
    full_log = {
        "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None)),
        "original_counts": {
            "entities": original_entities,
            "relationships": original_relationships
        },
        "transformations": []
    }
    
    # 1. Consolidar investigadores
    entities, relationships, consolidation_log = consolidate_researchers(
        data["entities"],
        data["relationships"]
    )
    full_log["transformations"].extend(consolidation_log)
    
    # 2. Agregar evidence_text faltante
    relationships, evidence_log = add_missing_evidence_text(relationships)
    full_log["transformations"].extend(evidence_log)
    
    # Actualizar data
    data["entities"] = entities
    data["relationships"] = relationships
    
    full_log["final_counts"] = {
        "entities": len(entities),
        "relationships": len(relationships)
    }
    full_log["summary"] = {
        "entities_removed": original_entities - len(entities),
        "entities_kept": len(entities)
    }
    
    print(f"[Postprocess] Final: {len(entities)} entidades, {len(relationships)} relaciones")
    print(f"[Postprocess] Reducción: {original_entities - len(entities)} entidades eliminadas")
    
    # Sobrescribir archivo original
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Post-procesamiento completado en: {json_path}")
    
    # Guardar log de transformaciones
    log_path = json_path.parent / f"{json_path.stem}_postprocess_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_log, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Log de transformaciones guardado en: {log_path}")
    
    # Generar resumen legible en texto
    summary_path = json_path.parent / f"{json_path.stem}_postprocess_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DE POST-PROCESAMIENTO\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Archivo procesado: {json_path.name}\n")
        f.write(f"Fecha: {full_log['timestamp']}\n\n")
        
        f.write("CONTEOS:\n")
        f.write(f"  Entidades originales: {original_entities}\n")
        f.write(f"  Entidades finales: {len(entities)}\n")
        f.write(f"  Entidades eliminadas (duplicados): {original_entities - len(entities)}\n")
        f.write(f"  Relaciones: {len(relationships)}\n\n")
        
        for transformation in full_log["transformations"]:
            f.write("-"*80 + "\n")
            f.write(f"PASO: {transformation['step']}\n")
            f.write(f"Cambios: {transformation['count']}\n\n")
            
            if transformation['step'] == "Normalización a MAYÚSCULAS":
                f.write("Ejemplos de normalizaciones:\n")
                for change in transformation['changes'][:10]:
                    f.write(f"  '{change['original']}' → '{change['normalized']}'\n")
            
            elif transformation['step'] == "Consolidación de nombres parciales":
                f.write("Merges realizados:\n")
                for change in transformation['changes']:
                    f.write(f"  '{change['partial_name']}' → '{change['canonical_name']}'\n")
                    f.write(f"    (ID: {change['partial_id']} → {change['canonical_id']})\n")
            
            elif transformation['step'] == "Actualización de relaciones":
                f.write(f"Relaciones actualizadas: {transformation['count']}\n")
                f.write("(Ver log JSON completo para detalles)\n")
            
            elif transformation['step'] == "Agregar evidence_text faltante":
                f.write(f"Relaciones corregidas: {transformation['count']}\n")
                f.write("Ejemplos:\n")
                for fix in transformation.get('sample', [])[:10]:
                    f.write(f"  {fix['relation']}\n")
            
            f.write("\n")
    
    print(f"[OK] Resumen legible guardado en: {summary_path}")


def main(max_docs: int | None = None) -> int:
    extractor = EntityExtractor(
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
    )

    # 1) Ejecutar extracción
    res = extractor.run(max_docs=max_docs)

    # 2) Guardar resultado
    filename = "entity_documents.json"
    extractor.save_in_file(filename)

    out_path = extractor.input_dir / filename
    print(f"[OK] Guardado en: {out_path.as_posix()}")
    print(f"[RUN] entidades={len(res.entities)} relaciones={len(res.relationships)} errores={len(res.errors)}")

    # 3) Leer el mismo archivo y reconstruir res
    loaded = extractor.load_from_json(filename)
    if loaded is None:
        print("[ERROR] No se pudo cargar el JSON generado.")
        return 1

    print(f"[LOAD] entidades={len(loaded.entities)} relaciones={len(loaded.relationships)} errores={len(loaded.errors)}")

    # 4) Mini chequeo de consistencia (opcional)
    if len(loaded.entities) != len(res.entities) or len(loaded.relationships) != len(res.relationships):
        print("[WARN] Los conteos RUN vs LOAD no coinciden (revisar serialización/deserialización).")
    else:
        print("[OK] RUN y LOAD coinciden en conteos.")
        equal = normalize_result(res) == normalize_result(loaded)
        print("RES == LOADED ?", equal)
    
    # 5) Post-procesamiento automático
    print("\n" + "="*60)
    print("INICIANDO POST-PROCESAMIENTO")
    print("="*60)
    postprocess_extraction(out_path)
        
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecutar extracción de entidades y relaciones")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Límite de documentos a procesar (None = todos)"
    )
    
    args = parser.parse_args()
    
    if args.max_docs is not None:
        print(f"[CONFIG] Límite de documentos: {args.max_docs}")
    else:
        print("[CONFIG] Procesando todos los documentos")
    
    raise SystemExit(main(max_docs=args.max_docs))
