# backend/scripts/run_extraction.py
from __future__ import annotations

import logging
import sys
from pathlib import Path

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


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


def main(
    max_projects: int | None = None,
) -> int:
    extractor = EntityExtractor()

    res = extractor.run(max_projects=max_projects)

    filename = "entity_documents.json"
    extractor.save_in_file(filename)

    out_path = extractor.input_dir / filename
    print(f"[OK] Guardado en: {out_path.as_posix()}")
    print(
        f"[RUN] entidades={len(res.entities)} relaciones={len(res.relationships)} errores={len(res.errors)}"
    )

    loaded = extractor.load_from_json(filename)
    if loaded is None:
        print("[ERROR] No se pudo cargar el JSON generado.")
        return 1

    print(
        f"[LOAD] entidades={len(loaded.entities)} relaciones={len(loaded.relationships)} errores={len(loaded.errors)}"
    )

    if len(loaded.entities) != len(res.entities) or len(loaded.relationships) != len(
        res.relationships
    ):
        print("[WARN] Los conteos RUN vs LOAD no coinciden.")
    else:
        print("[OK] RUN y LOAD coinciden en conteos.")
        equal = normalize_result(res) == normalize_result(loaded)
        print("RES == LOADED ?", equal)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar extracción de entidades y relaciones")
    parser.add_argument(
        "--max-projects", type=int, default=None, help="Límite de proyectos a procesar (None = todos)"
    )

    args = parser.parse_args()

    if args.max_projects is not None:
        print(f"[CONFIG] Límite de proyectos: {args.max_projects}")
    else:
        print("[CONFIG] Procesando todos los proyectos")

    raise SystemExit(
        main(
            max_projects=args.max_projects,
        )
    )
