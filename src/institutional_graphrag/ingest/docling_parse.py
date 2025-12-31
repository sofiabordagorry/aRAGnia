"""
Módulo para procesar PDFs con Docling y persistir el output estructurado.
"""

import json
from pathlib import Path
from typing import Any, Optional

from .load_pdf import DocumentLoadError

DEFAULT_CORPUS_DIR = Path("data/corpus")
DEFAULT_OUTPUT_DIR = Path("data/docling")


def parse_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    recursive: bool = False,
    skip_errors: bool = True,
) -> list[Path]:
    """
    Procesa todos los documentos de un directorio con Docling.
    """
    from .load_pdf import load_corpus

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []

    # Usar load_corpus que ya maneja la carga de documentos
    for loaded_doc in load_corpus(corpus_dir, recursive=recursive, skip_errors=skip_errors):
        try:
            output_data = {
                "source": str(loaded_doc.path),
                "num_documents": len(loaded_doc.documents),
                "documents": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in loaded_doc.documents
                ],
            }

            output_filename = f"{loaded_doc.path.stem}.json"
            output_path = output_dir / output_filename

            # Persistir a disco
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            output_files.append(output_path)
            print(f"✓ Procesado: {loaded_doc.path.name} -> {output_path.name}")
        except Exception as e:
            print(f"✗ Error al guardar {loaded_doc.path.name}: {e}")
            if not skip_errors:
                raise

    return output_files


def load_parsed_document(json_path: Path) -> Optional[dict[Any, Any]]:
    """
    Carga un documento estructurado previamente procesado.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Archivo JSON no encontrado: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            result: dict[Any, Any] = json.load(f)
            return result
    except Exception as e:
        raise DocumentLoadError(f"Error al cargar documento estructurado: {json_path}") from e
