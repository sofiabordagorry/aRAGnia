# backend/scripts/run_extraction.py
from __future__ import annotations

from pathlib import Path

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult

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

def main() -> int:
    extractor = EntityExtractor()

    # 1) Ejecutar extracción
    res = extractor.run()

    # 2) Guardar resultado
    filename = "Entity_documents.json"
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
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
