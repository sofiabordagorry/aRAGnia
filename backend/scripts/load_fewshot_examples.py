#!/usr/bin/env python3
"""
Carga los ejemplos few-shot iniciales de Text2Cypher en Qdrant.

Uso:
    python scripts/load_fewshot_examples.py

Para agregar nuevos ejemplos editá DEFAULT_EXAMPLES en fewshot_store.py.
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from institutional_graphrag.retrieval.fewshot_store import DEFAULT_EXAMPLES, FewShotStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carga ejemplos few-shot en Qdrant."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Borra todos los ejemplos existentes antes de cargar los nuevos.",
    )
    args = parser.parse_args()

    print("Conectando a Qdrant...")
    store = FewShotStore()

    # Si se pasa el flag --clear, vaciamos la colección
    if args.clear:
        print("Limpiando ejemplos existentes (flag --clear detectado)...")
        store.clear()

    before = store.count()
    print(f"Ejemplos existentes: {before}")

    store.add_examples(DEFAULT_EXAMPLES)

    after = store.count()
    print(f"Ejemplos cargados: {after - before} nuevos (total: {after})")
    store.close()


if __name__ == "__main__":
    main()
