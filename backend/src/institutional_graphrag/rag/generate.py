from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from institutional_graphrag.ingest.embedder import E5Embedder
from institutional_graphrag.llm.llm_provider import get_llm_client
from institutional_graphrag.retrieval.vector_store import VectorStore

BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_EMBEDDINGS_DIR = BASE_DIR / Path("data/embeddings")


@dataclass
class RAGChunk:
    id: str
    semantic_id: str
    text: str
    score: float
    source: str


@dataclass
class RAGResult:
    answer: str
    contexts: List[RAGChunk]


class RAG:
    def __init__(
        self,
        *,
        top_k: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 512,
        llm_model: Optional[str] = None,
    ):

        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_model = llm_model
        pass

    def build_context(self, chunks: List[RAGChunk], max_chars: int = 12000) -> str:
        parts: List[str] = []
        total = 0

        for ch in chunks:
            header = f"[[{ch.source or 'unknown'}|{ch.semantic_id}]] "

            block = (
                f"Inicio del chunk"
                f"Extracto de Documento de investigacion con nombre de documento: {header}\n"
                f"Texto a utilizar: {ch.text.strip()}\n"
                f"Fin del chunk"
            )

            if total + len(block) > max_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n...\n\n".join(parts).strip()

    def build_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        system = (
            "Mantener una conversación sobre extractos de documentos relacionados con temas académicos.\n"
            "→ Dar respuestas cortas, concretas y precisas.\n"
            "→ Utilizar únicamente la información de los siguientes extractos para construir la respuesta.\n"
            "===\n"
            "Extracto de Documento de investigación con nombre de documento: {Nombre del documento del primer chunk}\n"
            "→ Chunk: {ID del primer chunk}\n"
            "→ {Texto del primer chunk}\n"
            "...\n"
            "Extracto de Documento de investigación con nombre de documento: {Nombre del documento del último chunk}\n"
            "→ Chunk: {ID del último chunk}\n"
            "→ {Texto del último chunk}\n"
            "===\n"
            "→ Si los extractos no tienen una relación evidente con la pregunta, ignorarlos y responder que no se cuenta con información para responder la pregunta.\n"
            "→ Evitar las suposiciones.\n"
            "→ La conversación debe ser en español. Recordá ser concreto.\n"
            "→ Si la respuesta se construye utilizando información de más de un chunk, indicá claramente qué información proviene de cada chunk.\n"
            "→ Para cada afirmación relevante, mencioná el chunk correspondiente.\n"
            "→ No mezclar información de distintos chunks sin aclarar su origen.\n"
            "Siempre agregar al final de la respuesta los documentos y chunks utilizados bajo el título:\n"
            "**Referencias:**\n"
        )

        user = f"Pregunta: {question}\n\n" f"CONTEXTO:\n{context}\n\n" "===\n" "Respuesta:"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, query: str) -> RAGResult:

        if not query or not query.strip():
            raise ValueError("question is empty")
        # Generar embedding de la query
        embedder = get_embedder()
        query_embedding_array = embedder.embed_query(query)
        query_embedding = query_embedding_array[0].tolist()
        # 1) Retrieve
        store = VectorStore(collection_name="demo_collection", embedding_dim=1024)  # E5-large-v2
        raw = store.search(query_embedding, top_k=self.top_k)
        chunks = []
        for doc_id, score, meta in raw:
            rag_chunk = RAGChunk(
                id=doc_id,
                semantic_id=meta.get("semantic_id", "N/A"),
                text=meta.get("page_content", "N/A"),
                score=float(score),
                source=meta.get("__npy__", "N/A"),
            )
            chunks.append(rag_chunk)

        # 2) Build context
        context = self.build_context(chunks)

        # 3) Generate
        messages = self.build_messages(query, context)

        llm = get_llm_client(model=self.llm_model)

        answer = llm.generate(
            messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        # 4) Return
        return RAGResult(answer=answer, contexts=chunks)


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = E5Embedder()
    return _embedder
