from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from institutional_graphrag.retrieval.vector_store import VectorStore

EMBEDDINGS_DIR = Path(__file__).resolve().parents[4] / "data" / "embeddings"


@dataclass
class IngestStats:
    files_total: int = 0
    rows_total: int = 0
    skipped_total: int = 0
    inserted_total: int = 0

    def add(self, rows: int, skipped: int, inserted: int) -> None:
        self.files_total += 1
        self.rows_total += rows
        self.skipped_total += skipped
        self.inserted_total += inserted


def persist_all_embeddings_and_metadata(database: VectorStore) -> None:
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(f"No existe: {EMBEDDINGS_DIR}")

    npy_files = sorted(EMBEDDINGS_DIR.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No hay .npy en {EMBEDDINGS_DIR}")

    expected_dim: Optional[int] = None
    total = IngestStats()

    for npy_path in npy_files:
        expected_dim, stats = persist_embedding_and_metadata(
            npy_path=npy_path,
            database=database,
            expected_dim=expected_dim,
            verbose=True,
        )
        total.add(stats["rows"], stats["skipped"], stats["inserted"])

    print("\n" + "=" * 100)
    print("✅ INGEST FINALIZADO")
    print(f"Archivos procesados: {total.files_total}")
    print(f"Total filas (embeddings) vistas: {total.rows_total}")
    print(f"Saltadas por duplicado:         {total.skipped_total}")
    print(f"Insertadas nuevas:              {total.inserted_total}")
    print("=" * 100 + "\n")


def persist_embedding_and_metadata(
    npy_path: Path,
    database: VectorStore,
    expected_dim: Optional[int],
    verbose: bool = True,
) -> tuple[int, dict[str, int]]:
    """
    Devuelve:
      - expected_dim actualizado
      - stats dict: {"rows": N_total, "skipped": N_skip, "inserted": N_insert}
    """
    if verbose:
        print("\n" + "=" * 100)
        print(f"Procesando archivo: {npy_path.name}")

    stem = npy_path.stem
    meta_path = EMBEDDINGS_DIR / f"{stem}_metadata.json"
    if not meta_path.exists():
        raise ValueError(f"Falta metadata para {npy_path.name}: {meta_path.name}")

    embeddings = np.load(npy_path)
    if embeddings.ndim != 2:
        raise ValueError(f"{npy_path.name} no es 2D (N,D). Shape={embeddings.shape}")

    dim = int(embeddings.shape[1])
    if expected_dim is None:
        expected_dim = dim
    elif dim != expected_dim:
        raise ValueError(
            f"Dimensión inconsistente: {npy_path.name} tiene Dim={dim}, pero se esperaba Dim={expected_dim}"
        )

    metadata_list = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_list, list):
        raise ValueError(f"{meta_path.name} debería ser una lista de dicts (uno por embedding)")

    if len(metadata_list) != embeddings.shape[0]:
        raise ValueError(
            f"Cantidad inconsistente en {stem}: embeddings N={embeddings.shape[0]} vs metadata len={len(metadata_list)}"
        )

    # Preparar semantic_ids + ids (UUID) alineados por índice
    semantic_ids: list[str] = []
    ids_all: list[str] = []
    for i, m in enumerate(metadata_list):
        if not isinstance(m, dict):
            raise ValueError(f"{meta_path.name} contiene un item que no es dict")

        m["__npy__"] = npy_path.name
        m["__meta__"] = meta_path.name
        sid = f"{stem}_chunk_{i}"
        m["semantic_id"] = sid
        semantic_ids.append(sid)

        ids_all.append(str(uuid.uuid4()))

    existing = database.existing_payload_values("semantic_id", semantic_ids)

    keep_idxs = [i for i, sid in enumerate(semantic_ids) if sid not in existing]
    skipped = len(semantic_ids) - len(keep_idxs)

    if not keep_idxs:
        if verbose:
            print("✅ Todo ya estaba cargado. Nada para insertar.")
            print("=" * 100)
        return expected_dim, {"rows": len(semantic_ids), "skipped": skipped, "inserted": 0}

    # Filtrar embeddings/metadata/ids en el mismo orden
    embeddings = embeddings.astype(np.float32, copy=False)
    embeddings_to_add = embeddings[keep_idxs]
    metadata_to_add = [metadata_list[i] for i in keep_idxs]
    ids_to_add = [ids_all[i] for i in keep_idxs]

    stored_ids = database.add_documents(
        embeddings_to_add.tolist(),
        metadata_to_add,
        ids_to_add,
    )

    inserted = len(stored_ids)

    if verbose:
        print(f"Saltados por duplicados: {skipped}")
        print(f"Insertados nuevos:       {inserted}")
        print("=" * 100)

    return expected_dim, {"rows": len(semantic_ids), "skipped": skipped, "inserted": inserted}
