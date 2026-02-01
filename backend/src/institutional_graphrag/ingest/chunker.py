"""
Módulo para generar chunks a partir de la estructura del documento.

Implementa chunking por sección cuando es posible, con fallback por tamaño
para prevenir chunks demasiado grandes que excedan límites de tokens.
"""

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

# Se usa el mismo tokenizer que para realizar los embeddings
from ..config import EMBED_MODEL_ID


def get_native_chunker(
    tokenizer: str = EMBED_MODEL_ID, max_tokens: int = 512, merge_peers: bool = True
) -> HybridChunker:
    # Inicializar el chunker con el mismo tokenizer que se use en los embeddings
    return HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=merge_peers)


def chunk_document(
    doc: DoclingDocument,
    chunker: HybridChunker,
) -> list[dict]:
    """
    Parte en chunks un DoclingDocument usando su HybridChunker nativo.
    """

    chunk_iter = chunker.chunk(dl_doc=doc)
    processed_chunks = []

    for i, chunk in enumerate(chunk_iter):
        # contextualize() añade los encabezados padres al texto del chunk automáticamente
        # Útil para que el embedding capture el contexto de la sección
        text_with_context = chunker.contextualize(chunk)

        page_numbers = sorted(
            set(
                prov.page_no  # obtiene el número de pagina resultante de los siguientes pasos:
                for item in chunk.meta.doc_items  # itera por cada elemento del chunk
                for prov in item.prov  # itera por la source data de cada elemento
                if hasattr(prov, "page_no")  # se queda solo con aquellos que tengan page number
            )
        )

        element_type = chunk.meta.doc_items[0].label if chunk.meta.doc_items else "text"

        doc_name = doc.name

        token_count = chunker._count_chunk_tokens(doc_chunk=chunk)

        processed_chunks.append(
            {
                "chunk_id": f"{doc_name}#chunk{i}",  # ID Único para el Nodo en el Grafo
                "text": text_with_context,
                "metadata": {
                    "headings": chunk.meta.headings,
                    "page_numbers": page_numbers,
                    "element_type": element_type,
                    "parent_doc": doc_name,
                    "token_count": token_count,
                },
            }
        )

    return processed_chunks
