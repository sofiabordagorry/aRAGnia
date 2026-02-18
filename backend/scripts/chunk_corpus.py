"""Script para procesar y chunkear todos los documentos del corpus.

Usa los JSONs ya procesados por Docling (en data/docling/) para generar los chunks. Guarda los resultados en data/chunks/ (esto no se va a guardar en el pipeline final, solo es parte del desarrollo)
"""

import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from institutional_graphrag.config import EMBED_MODEL_ID
from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
from institutional_graphrag.ingest.table_extractors import convert_table_to_chunks

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main():
    docling_dir = DATA_DIR / Path("docling")
    output_dir = DATA_DIR / Path("chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not docling_dir.exists():
        print(f"No existe {docling_dir}")
        print("   Primero ejecutar: python scripts/docling_manual.py data/corpus")
        return

    json_files = list(docling_dir.glob("*.json"))
    if not json_files:
        print(f"No hay archivos JSON en {docling_dir}")
        print("   Primero ejecutar: python scripts/docling_manual.py data/corpus")
        return

    print(f"Leyendo documentos procesados: {docling_dir}")
    print(f"Output: {output_dir}")
    print(f"Archivos encontrados: {len(json_files)}")
    print("=" * 60)

    total_chunks = 0
    processed = 0
    errors = 0

    # Definir tokenizer a utilizar
    tokenizer = EMBED_MODEL_ID

    # Instanciar el chunker una sola vez
    shared_chunker = get_native_chunker(tokenizer=tokenizer)

    for json_file in sorted(json_files):
        try:
            file_path = Path(output_dir) / Path(json_file).name
            if file_path.exists():
                print("El archivo a particionar en chunks ya existe :", json_file, "en la carpeta ", output_dir)
                print("=" * 60)
                continue
            # Cargar JSON procesado por Docling
            with open(json_file, "r", encoding="utf-8") as f:
                doc_dict = json.load(f)

            # Reconstruir DoclingDocument
            doc = DoclingDocument.model_validate(doc_dict)

            # Aplicar chunking
            chunks = chunk_document(doc=doc, chunker=shared_chunker)

            # Guardar chunks en JSON
            output_file = output_dir / f"{json_file.stem}_chunks.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": doc.name,
                        "total_chunks": len(chunks),
                        "tokenizer": tokenizer,
                        "chunks": chunks,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            total_chunks += len(chunks)
            processed += 1

        except Exception as e:
            errors += 1
            print(f"✗ {json_file.name}: {e}")

    convert_table_to_chunks()
    print("=" * 60)
    print(f"Resumen:")
    print(f"   Archivos procesados: {processed}")
    print(f"   Errores: {errors}")
    print(f"   Chunks generados: {total_chunks}")


if __name__ == "__main__":
    main()
