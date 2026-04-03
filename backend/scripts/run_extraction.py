# backend/scripts/run_extraction.py
from __future__ import annotations

import json
import logging

# Importar funciones de postprocesamiento
import sys
from pathlib import Path

from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult

sys.path.insert(0, str(Path(__file__).parent))
from institutional_graphrag.ingest.postprocess_entities import Postprocessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


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


def main(
    max_docs: int | None = None,
    llm_researchers: bool = True,
    llm_topics: bool = True,
    enable_researcher_consolidation: bool = False,
    include_headings: bool = True,
) -> int:
    extractor = EntityExtractor()

    # 1) Ejecutar extracción
    res = extractor.run(max_docs=max_docs, llm_researchers=llm_researchers, llm_topics=llm_topics, include_headings=include_headings)

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

    # 4) Mini chequeo de consistencia (opcional)
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
    post_processor = Postprocessor(
        enable_researcher_consolidation=enable_researcher_consolidation,
        similarity_threshold=0.85,
    )
    post_processor.postprocess_file(out_path)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar extracción de entidades y relaciones")
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Límite de documentos a procesar (None = todos)"
    )
    parser.add_argument(
        "--no-llm-researchers",
        action="store_true",
        help="Desactiva la búsqueda de investigadores por LLM",
    )

    parser.add_argument(
        "--no-llm-topics",
        action="store_true",
        help="Desactiva la búsqueda de tópicos por LLM",
    )

    parser.add_argument(
        "--enable_researcher_consolidation",
        action="store_true",
        help="Activa la unificacion de Investigadores",
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

    llm_researchers = not args.no_llm_researchers
    llm_topics = not args.no_llm_topics
    enable_researcher_consolidation = args.enable_researcher_consolidation
    include_headings = not args.no_headings
    print(f"[CONFIG] LLM investigadores: {'ACTIVO' if llm_researchers else 'DESACTIVADO'}")
    print(f"[CONFIG] LLM tópicos: {'ACTIVO' if llm_topics else 'DESACTIVADO'}")
    print(f"[CONFIG] Unificacion de Investigadores: {'ACTIVO' if enable_researcher_consolidation else 'DESACTIVADO'}")
    print(f"[CONFIG] Incluir encabezados: {'ACTIVO' if include_headings else 'DESACTIVADO'}")

    # -------------------------------------------------

    raise SystemExit(
        main(
            max_docs=args.max_docs,
            llm_researchers=llm_researchers,
            llm_topics=llm_topics,
            enable_researcher_consolidation=enable_researcher_consolidation,
            include_headings=include_headings,
        )
    )
