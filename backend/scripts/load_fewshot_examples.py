#!/usr/bin/env python3
"""
Carga los ejemplos few-shot iniciales de Text2Cypher en Qdrant.

Uso:
    python scripts/load_fewshot_examples.py

Para agregar nuevos ejemplos editá DEFAULT_EXAMPLES en fewshot_store.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from institutional_graphrag.retrieval.fewshot_store import DEFAULT_EXAMPLES, FewShotStore


def main() -> None:
    print("Conectando a Qdrant...")
    store = FewShotStore()

    before = store.count()
    print(f"Ejemplos existentes: {before}")

    store.add_examples(DEFAULT_EXAMPLES)

    after = store.count()
    print(f"Ejemplos cargados: {after - before} nuevos (total: {after})")
    store.close()


if __name__ == "__main__":
    main()
