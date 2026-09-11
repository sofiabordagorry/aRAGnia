import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from aragnia.ingest.embedder import E5Embedder

# Paths
BASE_DIR = Path(__file__).resolve().parents[2] / "data"
embeddings_dir = BASE_DIR / Path("embeddings")


def get_top_k_knn(query_text, k=3):

    # Cargar modelo embeddings
    embedder = E5Embedder()

    all_embeddings = []
    all_metadata = []

    # Cargar en las listas los embeddings y metadata existentes
    for npy_file in embeddings_dir.glob("*.npy"):
        embeddings = np.load(npy_file)
        metadata_path = npy_file.with_name(f"{npy_file.stem}_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata)

    # Transformar a matriz para tener una "bolsa de embeddings"
    search_matrix = np.vstack(all_embeddings)

    # Busqueda K-nn con similitud coseno
    # 'brute' para conjuntos pequeños; 'ball_tree' si vemos que demora mucho
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(search_matrix)

    # Generar embedding de la query
    query_embedding = embedder.embed_query(query_text)

    # distances = 1 - similitud coseno
    distances, indices = knn.kneighbors(query_embedding)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({"score": 1 - dist, "content": all_metadata[idx]["text"]})

    return results


if __name__ == "__main__":
    query = input("Ingresar query: ")
    top_results = get_top_k_knn(query)

    for i, res in enumerate(top_results, 1):
        print(f"\n{i}. [Similitud: {res['score']:.4f}] {res['content'][:150]}...")
