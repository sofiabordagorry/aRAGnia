import torch
from sentence_transformers import SentenceTransformer


class E5Embedder:
    """Realiza embeddings usando el sentence transformer E5 Large v2."""

    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Cargando modelo de embeddings E5-large-v2 usando:  {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_passages(self, texts: list[str], batch_size: int = 16):
        """Dado un chunk hace un embedding con su contenido."""
        # Para usar E5 los chunks tienen que empezar con "passage: "
        formatted_texts = [f"passage: {t}" for t in texts]
        return self.model.encode(
            formatted_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )

    def embed_query(self, query: str):
        """Hace embedding de la query para poder hacer retrieval por similitud."""
        # La query se pasa como lista para que el formato de salida sea el adecuado
        return self.model.encode([f"query: {query}"], normalize_embeddings=True)
