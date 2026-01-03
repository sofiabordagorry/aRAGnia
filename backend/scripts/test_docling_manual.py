"""Script para testear manualmente el procesamiento con Docling."""

import shutil
import sys
import tempfile
from pathlib import Path

from institutional_graphrag.ingest.docling_parse import parse_corpus, load_parsed_document


def main():
    """Procesa documentos con Docling."""
    if len(sys.argv) < 2:
        print("Uso: python test_docling_manual.py <path>")
        return 1

    path = Path(sys.argv[1])
    output_dir = Path("data/docling")

    if not path.exists():
        print(f"Error: No existe {path}")
        return 1

    print(f"Input: {path}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    # Si es un archivo, crear un directorio temporal
    if path.is_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_corpus = Path(tmpdir) / "corpus"
            tmp_corpus.mkdir()
            shutil.copy(path, tmp_corpus / path.name)
            output_files = parse_corpus(tmp_corpus, output_dir)
    else:
        output_files = parse_corpus(path, output_dir)

    # Resumen
    print("-" * 60)
    print(f"Procesados: {len(output_files)} archivos\n")

    # Mostrar detalles
    for output_path in output_files:
        data = load_parsed_document(output_path)
        print(f"{output_path.name}")
        print(f"  Source: {Path(data['source']).name}")
        print(f"  Chunks: {data['num_documents']}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        print()


if __name__ == "__main__":
    main()
