import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Paths
embeddings_dir = Path("data/embeddings")

# Cargar modelo embeddings
print("Cargando modelo de embeddings E5-large-v2...")
model = SentenceTransformer("intfloat/e5-large-v2")


def get_top_k_knn(query_text, k=3):
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
    query_embedding = model.encode([f"query: {query_text}"], normalize_embeddings=True)

    # distances = 1 - similitud coseno
    distances, indices = knn.kneighbors(query_embedding)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({"score": 1 - dist, "content": all_metadata[idx]["page_content"]})

    return results


if __name__ == "__main__":
    query = input("Ingresar query: ")
    top_results = get_top_k_knn(query)

    for i, res in enumerate(top_results, 1):
        print(f"\n{i}. [Similitud: {res['score']:.4f}] {res['content'][:150]}...")
