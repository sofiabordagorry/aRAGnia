import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)


class VectorStore:
    """
    Vector store para almacenar y recuperar embeddings usando Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "institutional_docs",
        embedding_dim: int = 384,
    ):
        """
        Inicializa el vector store.
            collection_name: Nombre de la colección de Qdrant
            embedding_dim: Dimensión de los embeddings (default: 384 para sentence-transformers)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self.client = QdrantClient(host=host, port=port)

        # Crear colección si no existe
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Crea la colección si no existe."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Agrega embeddings al vector store.
        """
        # Validar dimensiones de embeddings
        if any(len(e) != self.embedding_dim for e in embeddings):
            raise ValueError(
                f"Todos los embeddings deben tener dimensión {self.embedding_dim}. "
                f"Se encontraron embeddings con dimensiones: {set(len(e) for e in embeddings)}"
            )

        # Validar longitud de metadata si se provee
        if metadata is not None and len(metadata) != len(embeddings):
            raise ValueError(
                f"La longitud de metadata ({len(metadata)}) debe coincidir con la longitud de embeddings ({len(embeddings)})"
            )

        # Validar longitud de ids si se provee
        if ids is not None and len(ids) != len(embeddings):
            raise ValueError(
                f"La longitud de ids ({len(ids)}) debe coincidir con la longitud de embeddings ({len(embeddings)})"
            )

        # Generar IDs si no se proveen
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(embeddings))]

        # Preparar metadata
        if metadata is None:
            metadata = [{} for _ in range(len(embeddings))]

        # Crear points para Qdrant
        points = []
        for embedding, meta, doc_id in zip(embeddings, metadata, ids):
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=meta,
            )
            points.append(point)

        # Subir points a Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Busca documentos similares en el vector store.
        """
        # Validar dimensión del embedding de consulta
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"La dimensión del embedding de consulta ({len(query_embedding)}) debe coincidir con "
                f"la dimensión del vector store ({self.embedding_dim})"
            )

        # Construir filtro si se provee
        qfilter = None
        if filter_dict:
            must = []
            for k, v in filter_dict.items():
                if isinstance(v, (list, tuple, set)):
                    # Usar MatchAny para consultas IN (ej: category in ["research", "education"])
                    must.append(FieldCondition(key=k, match=MatchAny(any=list(v))))
                else:
                    # Usar MatchValue para igualdad exacta
                    must.append(FieldCondition(key=k, match=MatchValue(value=v)))
            qfilter = Filter(must=must)

        # Ejecutar búsqueda
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qfilter,
        ).points

        # Formatear resultados
        results = []
        for scored_point in search_result:
            doc_id = scored_point.id
            score = scored_point.score
            metadata = dict(scored_point.payload or {})

            results.append((doc_id, score, metadata))

        return results

    def count_documents(self) -> int:
        """
        Obtiene el número total de documentos en la colección.
        """
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return int(result.count)

    def clear_collection(self) -> None:
        """Elimina todos los documentos de la colección."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()

    def close(self) -> None:
        """Cierra la conexión con Qdrant."""
        self.client.close()
