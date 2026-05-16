# backend/scripts/run_extraction.py
from __future__ import annotations

import logging
import sys
from pathlib import Path

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult
from institutional_graphrag.ingest.postprocess_entities import Postprocessor

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
    max_docs: int | None = None,
    llm_topics: bool = True,
    include_headings: bool = True,
) -> int:
    extractor = EntityExtractor()

    # 1) Ejecutar extracción
    res = extractor.run(max_docs=max_docs, llm_topics=llm_topics, include_headings=include_headings)

    # 2) Guardar resultado
    filename = "entity_documents.json"
    extractor.save_in_file(filename)

    out_path = extractor.input_dir / filename
    print(f"[OK] Guardado en: {out_path.as_posix()}")
    print(
        f"[RUN] entidades={len(res.entities)} relaciones={len(res.relationships)} errores={len(res.errors)}"
    )

    # 3) Leer el mismo archivo y reconstruir res
    loaded = extractor.load_from_json(filename)
    if loaded is None:
        print("[ERROR] No se pudo cargar el JSON generado.")
        return 1

    print(
        f"[LOAD] entidades={len(loaded.entities)} relaciones={len(loaded.relationships)} errores={len(loaded.errors)}"
    )

    # 4) Mini chequeo de consistencia
    if len(loaded.entities) != len(res.entities) or len(loaded.relationships) != len(
        res.relationships
    ):
        print(
            "[WARN] Los conteos RUN vs LOAD no coinciden (revisar serialización/deserialización)."
        )
    else:
        print("[OK] RUN y LOAD coinciden en conteos.")
        equal = normalize_result(res) == normalize_result(loaded)
        print("RES == LOADED ?", equal)

    # 5) Post-procesamiento automático
    print("\n" + "=" * 60)
    print("INICIANDO POST-PROCESAMIENTO")
    print("=" * 60)
    Postprocessor().postprocess_file(out_path)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar extracción de entidades y relaciones")
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Límite de documentos a procesar (None = todos)"
    )
    parser.add_argument(
        "--no-llm-topics",
        action="store_true",
        help="Desactiva la búsqueda de tópicos por LLM",
    )
    parser.add_argument(
        "--no-headings",
        action="store_true",
        help="Desactiva la inclusión de encabezados en el texto del LLM",
    )

    args = parser.parse_args()

    if args.max_docs is not None:
        print(f"[CONFIG] Límite de documentos: {args.max_docs}")
    else:
        print("[CONFIG] Procesando todos los documentos")

    print(f"[CONFIG] LLM tópicos: {'ACTIVO' if not args.no_llm_topics else 'DESACTIVADO'}")
    print(f"[CONFIG] Incluir encabezados: {'ACTIVO' if not args.no_headings else 'DESACTIVADO'}")

    raise SystemExit(
        main(
            max_docs=args.max_docs,
            llm_topics=not args.no_llm_topics,
            include_headings=not args.no_headings,
        )
    )
