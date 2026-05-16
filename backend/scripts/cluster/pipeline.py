"""
pipeline.py
===========
Pipeline completo de procesamiento: Docling → Chunks → Extracción.

Ejecuta cada etapa en secuencia y puede retomar desde donde quedó
(cada etapa chequea si el output ya existe antes de reprocessar).

Uso:
    python cluster/pipeline.py [opciones]

Opciones:
    --data-dir PATH         Directorio de datos (default: ./data)
    --skip-docling          Saltear parseo Docling (si ya está hecho)
    --skip-chunks           Saltear generación de chunks
    --skip-extraction       Saltear extracción de entidades/relaciones
    --no-llm-topics         Desactivar extracción LLM de tópicos
    --max-docs N            Límite de documentos para extracción (debug)
    --env-file PATH         Archivo .env con variables de entorno (default: .env)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env(env_file: Path) -> None:
    """Carga variables de entorno desde un archivo .env si existe."""
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)
        log.info("Variables de entorno cargadas desde: %s", env_file)
    else:
        log.warning(
            "Archivo .env no encontrado en %s — se asume que las variables ya están en el entorno.",
            env_file,
        )


def _require_dir(path: Path, name: str) -> None:
    if not path.exists():
        log.error("No se encontró el directorio '%s': %s", name, path)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Etapas del pipeline
# ---------------------------------------------------------------------------


def step_docling(corpus_dir: Path, docling_dir: Path, batch_size: int = 10) -> None:
    """Parsea los PDFs del corpus con Docling y guarda JSONs estructurados."""
    log.info("Etapa 1: Docling")

    _require_dir(corpus_dir, "corpus")
    docling_dir.mkdir(parents=True, exist_ok=True)

    from institutional_graphrag.ingest.docling_parser import (
        parse_corpus,
    )

    saved = 0

    for doc_dict in parse_corpus(corpus_dir, batch_size=batch_size):
        filename = doc_dict.get("name", "sin_nombre")
        output_path = docling_dir / f"{filename}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        saved += 1

    log.info("Etapa 1 completada — %d documentos guardados", saved)


def step_chunks(docling_dir: Path, chunks_dir: Path) -> None:
    """Genera chunks a partir de los JSONs de Docling."""
    log.info("Etapa 2: Chunking")

    _require_dir(docling_dir, "docling")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    from docling_core.types.doc import DoclingDocument
    from institutional_graphrag.config import EMBED_MODEL_ID
    from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
    from institutional_graphrag.ingest.table_extractors import convert_table_to_chunks

    json_files = sorted(docling_dir.glob("*.json"))
    if not json_files:
        log.error("No hay JSONs en %s — ejecutar primero la etapa Docling.", docling_dir)
        sys.exit(1)

    chunker = get_native_chunker(tokenizer=EMBED_MODEL_ID)
    processed = skipped = errors = 0

    for json_file in json_files:
        output_file = chunks_dir / f"{json_file.stem}_chunks.json"
        if output_file.exists():
            skipped += 1
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                doc_dict = json.load(f)
            doc = DoclingDocument.model_validate(doc_dict)
            chunks = chunk_document(doc=doc, chunker=chunker)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": doc.name,
                        "total_chunks": len(chunks),
                        "tokenizer": EMBED_MODEL_ID,
                        "chunks": chunks,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            processed += 1
        except Exception as exc:
            log.error("Error en %s: %s", json_file.name, exc)
            errors += 1

    try:
        convert_table_to_chunks()
    except FileNotFoundError as exc:
        log.warning("No se procesaron tablas: %s", exc)
    log.info(
        "Etapa 2 completada — total=%d procesados=%d salteados=%d errores=%d",
        len(json_files),
        processed,
        skipped,
        errors,
    )


def step_extraction(
    data_dir: Path,
    max_docs: int | None,
    llm_topics: bool,
    checkpoint_every: int = 5,
    llm_model: str | None = None,
) -> None:
    """Extrae entidades y relaciones (tabular + LLM tópicos) y las postprocesa."""
    log.info("Etapa 3: Extracción")

    from institutional_graphrag.extraction.ie import EntityExtractor
    from institutional_graphrag.ingest.postprocess_entities import Postprocessor

    extractor = EntityExtractor(llm_model=llm_model, data_dir=data_dir)
    res = extractor.run(
        max_docs=max_docs,
        llm_topics=llm_topics,
        checkpoint_every=checkpoint_every,
    )

    filename = "entity_documents.json"
    extractor.save_in_file(filename)
    out_path = extractor.input_dir / filename

    log.info(
        "Extracción completada — entidades=%d relaciones=%d errores=%d",
        len(res.entities),
        len(res.relationships),
        len(res.errors),
    )

    log.info("Iniciando post-procesamiento...")
    Postprocessor().postprocess_file(out_path)

    log.info("Etapa 3 completada — guardado en: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline: Docling → Chunks → Extracción")
    repo_root = Path(__file__).resolve().parents[3]
    default_data = repo_root / "data"

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="Directorio raíz de datos (default: ./data)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=repo_root / "backend" / ".env",
        help="Archivo .env con variables de entorno",
    )

    # Etapas opcionales
    parser.add_argument("--skip-docling", action="store_true", help="Saltear etapa Docling")
    parser.add_argument("--skip-chunks", action="store_true", help="Saltear etapa de chunking")
    parser.add_argument(
        "--skip-extraction", action="store_true", help="Saltear etapa de extracción"
    )
    parser.add_argument(
        "--docling-batch-size",
        type=int,
        default=10,
        help="Cantidad de documentos por lote en Docling (default: 10)",
    )

    # Opciones de extracción
    parser.add_argument("--no-llm-topics", action="store_true", help="Desactivar LLM para tópicos")
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Límite de documentos (None = todos)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Guardar checkpoint cada N documentos en extracción LLM (default: 5)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    corpus_dir = data_dir / "corpus"
    docling_dir = data_dir / "docling"
    chunks_dir = data_dir / "chunks"

    log.info("Pipeline iniciado — data dir: %s", data_dir)

    _load_env(args.env_file)

    if not args.skip_docling:
        step_docling(corpus_dir, docling_dir, batch_size=args.docling_batch_size)
    else:
        log.info("Etapa 1 (Docling) salteada.")

    if not args.skip_chunks:
        step_chunks(docling_dir, chunks_dir)
    else:
        log.info("Etapa 2 (Chunks) salteada.")

    if not args.skip_extraction:
        step_extraction(
            data_dir=data_dir,
            max_docs=args.max_docs,
            llm_topics=not args.no_llm_topics,
            checkpoint_every=args.checkpoint_every,
        )
    else:
        log.info("Etapa 3 (Extracción) salteada.")

    log.info("Pipeline completo")


if __name__ == "__main__":
    main()
