from aragnia.ingest.persist_embeddings import persist_all_embeddings_and_metadata
from aragnia.retrieval.vector_store import VectorStore

# Script para probar el archivo persist_embeddings.py el cual carga en la base de datos los embeddings ubicados en data/embeddings


def main():
    store = VectorStore(collection_name="demo_collection", embedding_dim=1024)  # E5-large-v2
    persist_all_embeddings_and_metadata(store)


if __name__ == "__main__":
    main()
