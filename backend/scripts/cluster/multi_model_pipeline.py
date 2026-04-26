"""
multi_model_pipeline.py
=======================
Ejecuta el pipeline de extracción con múltiples modelos LLM y guarda resultados
en carpetas separadas para cada modelo.

Uso:
    python cluster/multi_model_pipeline.py [opciones]

Opciones:
    --data-dir PATH         Directorio raíz de datos (default: ./data)
    --skip-docling          Saltear Docling (si ya está procesado)
    --skip-chunks           Saltear chunking
    --no-llm-researchers    Desactivar extracción LLM de investigadores
    --no-llm-topics         Desactivar extracción LLM de tópicos
    --researcher-consolidation  Activar consolidación de investigadores
    --max-docs N            Límite de documentos por modelo (None = todos)
    --env-file PATH         Archivo .env con variables de entorno
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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
log = logging.getLogger("multi_model_pipeline")

# Modelos LLM a utilizar
LLM_MODELS = [
    #"Qwen/Qwen2.5-7B-Instruct",
    #"meta-llama/Llama-3.1-8B-Instruct",
   # "Qwen/Qwen3.5-9B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it" 
]


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


def run_pipeline_for_model(
    data_dir: Path,
    model_name: str,
    skip_docling: bool = False,
    skip_chunks: bool = False,
    no_llm_researchers: bool = False,
    no_llm_topics: bool = False,
    researcher_consolidation: bool = False,
    max_docs: int | None = None,
    env_file: Path | None = None,
) -> None:
    """Ejecuta el pipeline completo para un modelo LLM específico."""
    import importlib.util
    
    # Cargar dinámicamente pipeline.py
    pipeline_path = Path(__file__).parent / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)

    # Componentes del pipeline
    step_docling = pipeline_module.step_docling
    step_chunks = pipeline_module.step_chunks
    step_extraction = pipeline_module.step_extraction

    # Directorios
    corpus_dir = data_dir / "corpus"
    docling_dir = data_dir / "docling"
    chunks_dir = data_dir / "chunks"

    # Directorio de salida específico para este modelo
    model_output_base = data_dir / "results" / model_name.replace("/", "_")
    model_output_base.mkdir(parents=True, exist_ok=True)

    # Crear data_dir temporal para este modelo con symlinks a corpus/chunks
    # pero con entities_relations separado
    entities_relations_dir = model_output_base / "entities_relations"
    entities_relations_dir.mkdir(parents=True, exist_ok=True)

    # Crear copia de data_dir pero apuntando a la carpeta de entities_relations del modelo
    model_data_dir = model_output_base / "data_links"
    model_data_dir.mkdir(exist_ok=True)

    # Copiar estructura necesaria preservando symlinks.
    # Nota: Path.exists() sigue symlinks, así que devuelve False para symlinks rotos.
    # Usamos is_symlink() + exists() para detectar ambos casos y limpiar symlinks rotos
    # que pueden quedar de corridas previas (el cleanup trap en submit.sh copia todo de
    # vuelta al home, incluyendo estos symlinks que apuntan a /scratch y quedan rotos).
    def _ensure_symlink(link: Path, target: Path) -> None:
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target)

    _ensure_symlink(model_data_dir / "corpus", corpus_dir)
    _ensure_symlink(model_data_dir / "chunks", chunks_dir)
    _ensure_symlink(model_data_dir / "docling", docling_dir)
    tables_src = data_dir / "tables"
    if tables_src.exists():
        _ensure_symlink(model_data_dir / "tables", tables_src)

    # Crear carpeta entities_relations para este modelo
    (model_data_dir / "entities_relations").mkdir(exist_ok=True)

    log.info(f"\n{'='*70}")
    log.info(f"Iniciando pipeline con modelo: {model_name}")
    log.info(f"Directorio de salida: {model_output_base}")
    log.info(f"{'='*70}\n")

    if env_file:
        _load_env(env_file)

    # Ejecutar etapas
    if not skip_docling:
        step_docling(corpus_dir, docling_dir)
    else:
        log.info("Etapa 1 (Docling) salteada.")

    if not skip_chunks:
        step_chunks(docling_dir, chunks_dir)
    else:
        log.info("Etapa 2 (Chunks) salteada.")

    # Extracción con el modelo específico
    step_extraction(
        data_dir=model_data_dir,
        max_docs=max_docs,
        llm_researchers=not no_llm_researchers,
        llm_topics=not no_llm_topics,
        researcher_consolidation=researcher_consolidation,
        llm_model=model_name,
    )

    # Copiar entity_documents.json a la carpeta de salida del modelo
    entity_file = model_data_dir / "entities_relations" / "entity_documents.json"
    if entity_file.exists():
        output_file = model_output_base / "entity_documents.json"
        shutil.copy2(entity_file, output_file)
        log.info(f"✓ Entity documents guardado en: {output_file}")

    # Copiar registry también
    registry_file = model_data_dir / "entities_relations" / "llm_registry.json"
    if registry_file.exists():
        output_registry = model_output_base / "llm_registry.json"
        shutil.copy2(registry_file, output_registry)
        log.info(f"✓ Registry guardado en: {output_registry}")

    log.info(f"\n{'='*70}")
    log.info(f"Pipeline completado para modelo: {model_name}")
    log.info(f"{'='*70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline multi-modelo: Docling → Chunks → Extracción (x3 LLMs)"
    )
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

    # Opciones de etapas
    parser.add_argument(
        "--skip-docling",
        action="store_true",
        help="Saltear etapa Docling",
    )
    parser.add_argument(
        "--skip-chunks",
        action="store_true",
        help="Saltear etapa de chunking",
    )

    # Opciones de extracción
    parser.add_argument(
        "--no-llm-researchers",
        action="store_true",
        help="Desactivar LLM para investigadores",
    )
    parser.add_argument(
        "--no-llm-topics",
        action="store_true",
        help="Desactivar LLM para tópicos",
    )
    parser.add_argument(
        "--researcher-consolidation",
        action="store_true",
        help="Activar consolidación de investigadores",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Límite de documentos (None = todos)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    corpus_dir = data_dir / "corpus"

    log.info("\n" + "="*70)
    log.info("PIPELINE MULTI-MODELO")
    log.info("="*70)
    log.info("Modelos a procesar:")
    for i, model in enumerate(LLM_MODELS, 1):
        log.info(f"  {i}. {model}")
    log.info("="*70 + "\n")

    _require_dir(corpus_dir, "corpus")

    # Ejecutar pipeline para cada modelo.
    # Si un modelo falla (ej. gated sin token, OOM, etc.) seguimos con los demás
    # para no perder los resultados parciales ya generados.
    failed_models: list[tuple[str, str]] = []
    for model in LLM_MODELS:
        try:
            run_pipeline_for_model(
                data_dir=data_dir,
                model_name=model,
                skip_docling=args.skip_docling,
                skip_chunks=args.skip_chunks,
                no_llm_researchers=args.no_llm_researchers,
                no_llm_topics=args.no_llm_topics,
                researcher_consolidation=args.researcher_consolidation,
                max_docs=args.max_docs,
                env_file=args.env_file,
            )
        except Exception as e:
            log.error(
                "Modelo %s falló: %s: %s",
                model,
                type(e).__name__,
                e,
                exc_info=True,
            )
            failed_models.append((model, f"{type(e).__name__}: {e}"))
            continue

    log.info("\n" + "="*70)
    if failed_models:
        log.warning(
            "MODELOS CON ERROR (%d/%d):",
            len(failed_models),
            len(LLM_MODELS),
        )
        for name, err in failed_models:
            log.warning("  - %s: %s", name, err)
        log.info("-"*70)
    log.info("TODOS LOS MODELOS PROCESADOS")
    log.info("="*70)
    log.info(f"Resultados guardados en: {data_dir / 'results'}")
    log.info("\nEstructura de salida:")
    results_dir = data_dir / "results"
    if results_dir.exists():
        for model_dir in sorted(results_dir.iterdir()):
            if model_dir.is_dir():
                entity_file = model_dir / "entity_documents.json"
                if entity_file.exists():
                    size = entity_file.stat().st_size / 1024
                    log.info(f"  - {model_dir.name}/entity_documents.json ({size:.1f} KB)")
    log.info("="*70 + "\n")


if __name__ == "__main__":
    main()
