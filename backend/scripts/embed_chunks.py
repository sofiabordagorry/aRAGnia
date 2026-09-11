import json
from pathlib import Path

import numpy as np

from aragnia.ingest.embedder import E5Embedder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main():
    chunks_dir = DATA_DIR / Path("chunks")
    output_dir = DATA_DIR / Path("embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_dir.exists():
        print(f"No existe {chunks_dir}")
        print("   Primero ejecutar: python scripts/chunk_corpus.py")
        return

    json_files = list(chunks_dir.glob("*.json"))
    if not json_files:
        print(f"No hay archivos JSON en {chunks_dir}")
        print("   Primero ejecutar: python scripts/chunk_corpus.py")
        return

    print(f"Leyendo chunks procesados: {chunks_dir}")
    print(f"Output: {output_dir}")
    print(f"Archivos encontrados: {len(json_files)}")
    print("=" * 60)

    processed = 0
    errors = 0

    embedder = E5Embedder()

    for json_file in sorted(json_files):
        file_id = json_file.stem.removesuffix("_chunks")
        embedding_path = output_dir / f"{file_id}.npy"

        # Fijarse que no hayan sido creados embeddings para ese archivo aún
        if embedding_path.exists():
            print(f"{json_file.name} ya había sido procesado.")
            continue

        print(f"Procesando {json_file.name}...")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = data.get("chunks", [])
            if not chunks:
                print(f"✗ {json_file.name}: no tiene chunks")
                continue

            chunk_content = [c["text"] for c in chunks]

            embeddings = embedder.embed_passages(chunk_content)

            # Guardar embeddings
            np.save(output_dir / f"{file_id}.npy", embeddings)

            # Se guardan tambien los chunks como metadata de los embeddings
            # Esto evita referencias erroneas y hace que el retrieval requiera menos parseo
            with open(output_dir / f"{file_id}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"✗ {json_file.name}: {e}")

    print("=" * 60)
    print("Resumen:")
    print(f"   Archivos procesados: {processed}")
    print(f"   Errores: {errors}")


if __name__ == "__main__":
    main()
