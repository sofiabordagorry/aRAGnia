"""Script para testear manualmente el procesamiento con Docling."""

import sys
import json
from pathlib import Path

from institutional_graphrag.ingest.docling_parser import (
    parse_corpus, 
    parse_single_document,
    DEFAULT_DOCLING_DIR,
    DocumentAlreadyProcessed
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def save_docling_dict(doc_dict: dict, output_dir: Path) -> Path:
    """
    Guarda un diccionario docling como JSON.
    """
    filename = doc_dict.get("name","sin_nombre")
    output_path = output_dir / f"{filename}.json"

    json_output = json.dumps(doc_dict, indent=2, ensure_ascii=False)

    with open (output_path, "w", encoding="utf-8") as f:
        f.write(json_output)

    return output_path

def main():
    """Procesa documentos con Docling y persiste el output estructurado."""
    if len(sys.argv) < 2:
        print("Modo de uso: python docling_manual.py <path>")
        return 1

    path = DATA_DIR / Path(sys.argv[1])
    output_dir = DEFAULT_DOCLING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"Error: No existe {path}")
        return 1

    print(f"Input: {path}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    saved_files = []

    # Si es un archivo, crear un directorio temporal
    if path.is_file():
        # Para un archivo individual. 
        # Usar para testeo ya que no tiene restricciones sobre formato (solo las propias de Docling)
        try:
            doc_dict = parse_single_document(path)
            saved_path = save_docling_dict(doc_dict=doc_dict,output_dir=output_dir)
            print(f"✓ Guardado: {saved_path.name}")
            saved_files.append(saved_path)
        except DocumentAlreadyProcessed as e:
            print(e)
    else:
        # Directorio
        # Para todos los archivos soportados de un directorio
        results = parse_corpus(path)
        for doc_dict in results:
            saved_path = save_docling_dict(doc_dict=doc_dict, output_dir=output_dir)
            print(f"✓ Guardado: {saved_path.name}")
            saved_files.append(saved_path)

    # Resumen
    print("-" * 60)
    print(f"Procesados exitosamente: {len(saved_files)} archivos\n")


if __name__ == "__main__":
    main()
