"""
Script de demostración del flujo completo:
1. Cargar documento parseado con Docling
2. Generar chunks
3. Generar embeddings
4. Almacenar en vector store
5. Buscar chunks similares
"""

import json
from pathlib import Path
from docling_core.types.doc import DoclingDocument
import uuid

from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
from institutional_graphrag.ingest.embedder import E5Embedder
from institutional_graphrag.retrieval.vector_store import VectorStore
from institutional_graphrag.config import EMBED_MODEL_ID


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    docling_dir = project_root / "data" / "docling"

    # 1. Seleccionar un JSON parseado de ejemplo
    json_files = list(docling_dir.glob("*.json"))
    if not json_files:
        print(f"No hay archivos JSON en {docling_dir}")
        return

    json_path = json_files[1]
    print(f"Usando JSON: {json_path.name}")

    # 2. Generar chunks
    print("\nGenerando chunks...")

    # Definir tokenizer a utilizar
    tokenizer = EMBED_MODEL_ID

    # Instanciar el chunker una sola vez
    shared_chunker = get_native_chunker(tokenizer=tokenizer)

    with open(json_path, "r", encoding="utf-8") as f:
        doc_dict = json.load(f)

    # Reconstruir DoclingDocument
    doc = DoclingDocument.model_validate(doc_dict)

    # Aplicar chunking
    chunks = chunk_document(doc=doc, chunker=shared_chunker)

    print(f"Generados {len(chunks)} chunks")

    # Mostrar ejemplo
    if chunks:
        print("\nEjemplo de chunk:")
        print(f"  ID: {chunks[0]['chunk_id']}")
        print(f"  Texto (primeros 100 chars): {chunks[0]['text'][:100]}...")
        print(f"  Metadata: {chunks[0]['metadata']}")

    # 3. Generar embeddings
    print("\nGenerando embeddings con E5-large-v2...")
    embedder = E5Embedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings_array = embedder.embed_passages(texts, batch_size=8)
    embeddings = embeddings_array.tolist()
    print(f"Generados {len(embeddings)} embeddings de dimensión {len(embeddings[0])}")

    # 4. Preparar IDs y metadata
    # Crear mapeo: UUID -> índice para recuperar chunks después
    id_to_chunk_index = {}
    ids = []
    metadata_list = []

    for i, chunk in enumerate(chunks):
        # Usar UUID válido para Qdrant
        uuid_id = str(uuid.uuid4())
        ids.append(uuid_id)

        # Mapear UUID -> índice del chunk
        id_to_chunk_index[uuid_id] = i

        # ID semántico basado en el nombre del archivo + índice
        semantic_id = f"{json_path.stem}_chunk_{i}"

        # Metadata (incluir el ID semántico)
        meta = {k: v for k, v in chunk["metadata"].items() if k != "text"}
        meta["semantic_id"] = semantic_id
        metadata_list.append(meta)

    # 5. Crear vector store y agregar chunks
    print("\nAlmacenando en Qdrant...")
    store = VectorStore(collection_name="demo_collection", embedding_dim=1024)  # E5-large-v2

    # Limpiar colección si existe
    store.clear_collection()

    # Agregar chunks
    stored_ids = store.add_documents(embeddings, metadata_list, ids)
    print(f"Almacenados {len(stored_ids)} chunks")
    print(f"  Primeros IDs: {stored_ids[:3]}")

    # Verificar conteo
    count = store.count_documents()
    print(f"Chunks en colección: {count}")

    # 6. Buscar chunks similares
    print("\nProbando búsqueda...")
    query = "¿Cuáles son los objetivos del proyecto?"
    print(f"  Query: '{query}'")

    # Generar embedding de la query
    query_embedding_array = embedder.embed_query(query)
    query_embedding = query_embedding_array[0].tolist()

    # Buscar
    results = store.search(query_embedding, top_k=3)

    print("\nTop 3 resultados:")
    for i, (doc_id, score, meta) in enumerate(results, 1):
        print(f"\n  {i}. Score: {score:.4f}")
        print(f"     UUID: {doc_id}")
        print(f"     Semantic ID: {meta.get('semantic_id', 'N/A')}")
        print(f"     Metadata: {meta}")

        # Recuperar texto del chunk usando el índice
        chunk_index = id_to_chunk_index.get(doc_id, -1)
        if chunk_index >= 0 and chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            print(f"     Texto (primeros 150 chars): {chunk['text'][:150]}...")
        else:
            print("     Texto: (no encontrado)")

    # 7. Buscar con filtros
    if metadata_list and metadata_list[0]:
        print("\n Probando búsqueda con filtros...")
        # Usar el primer valor de metadata como filtro
        filter_key = list(metadata_list[0].keys())[0]
        filter_value = metadata_list[0][filter_key]

        results_filtered = store.search(
            query_embedding, top_k=2, filter_dict={filter_key: filter_value}
        )
        print(f"  Filtro: {filter_key} = {filter_value}")
        print(f"  Resultados: {len(results_filtered)}")

    # 8. Cerrar conexión
    store.close()
    print("\n Demo completada!")


if __name__ == "__main__":
    main()
