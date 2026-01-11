from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import uuid
from institutional_graphrag.retrieval.vector_store import VectorStore

import numpy as np

EMBEDDINGS_DIR = Path(__file__).resolve().parents[4] / "data" / "embeddings"


def persist_all_embeddings_and_metadata(database: VectorStore):
    # Guarda en la base de datos todos los embeddings junto con su metadata
    
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(f"No existe: {EMBEDDINGS_DIR}")

    npy_files = sorted(EMBEDDINGS_DIR.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No hay .npy en {EMBEDDINGS_DIR}")


    expected_dim: Optional[int] = None
    for npy_path in npy_files:
        expected_dim = persist_embedding_and_metadata(npy_path=npy_path, database=database, expected_dim=expected_dim,)
    pass


def persist_embedding_and_metadata(npy_path: Path, database: VectorStore, expected_dim: Optional[int]) -> int:
    # Guarda en la base de datos el par (embeddings, metadata) dado el Path del archivo .npy
    print("\n" + "=" * 100)
    print(f" Procesando archivo: {npy_path.name}")
    stem = npy_path.stem
    meta_path = EMBEDDINGS_DIR / f"{stem}_metadata.json"
    if not meta_path.exists():
        raise ValueError(f"Falta metadata para {npy_path.name}: {meta_path.name}")

    embeddings = np.load(npy_path)  
    if embeddings.ndim != 2:
        raise ValueError(f"{npy_path.name} no es 2D (N,D). Shape={embeddings.shape}")

    Dim = int(embeddings.shape[1])

    if expected_dim is None:
        expected_dim = Dim
    elif Dim != expected_dim:
        raise ValueError(
            f"Dimensión inconsistente: {npy_path.name} tiene Dimension={Dim}, pero se esperaba Dimension={expected_dim}"
        )
    
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    if not isinstance(metadata_list, list):
        raise ValueError(f"{meta_path.name} debería ser una lista de dicts (uno por embedding)")

    if len(metadata_list) != embeddings.shape[0]:
        raise ValueError(f"Cantidad inconsistente en {stem}: embeddings N={embeddings.shape[0]} vs metadata len={len(metadata_list)}")

    # Preparar IDs y metadata
    # Crear mapeo: UUID -> índice para recuperar chunks después
    ids = []
    for i, m in enumerate(metadata_list):
        if not isinstance(m, dict):
            raise ValueError(f"{meta_path.name} contiene un item que no es dict")
        
        # Usar UUID válido para Qdrant
        uuid_id = str(uuid.uuid4())
        ids.append(uuid_id)
        m["__npy__"] = npy_path.name
        m["__meta__"] = meta_path.name
        m["semantic_id"] = f"{npy_path.stem}_chunk_{i}"

    embeddings = embeddings.astype(np.float32, copy=False)

    if len(metadata_list) != embeddings.shape[0]:
        raise ValueError("Invariante roto: metadata y embeddings no están alineados")

    print(f"\n Total embeddings: {embeddings.shape[0]} | Dim: {embeddings.shape[1]}")
    print(f" Total metadata:   {len(metadata_list)}")
    # Agregar documentos
    stored_ids = database.add_documents(embeddings, metadata_list, ids)
    print(f"Almacenados {len(stored_ids)} documentos")
    print("\n" + "=" * 100)

    return expected_dim



