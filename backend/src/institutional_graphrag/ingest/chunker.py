from __future__ import annotations

from typing import Any, cast

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

from ..config import EMBED_MODEL_ID


def get_native_chunker(tokenizer: Any = EMBED_MODEL_ID, merge_peers: bool = True) -> HybridChunker:
    return HybridChunker(tokenizer=tokenizer, merge_peers=merge_peers)


def chunk_document(doc: DoclingDocument, chunker: HybridChunker) -> list[dict]:
    chunk_iter = chunker.chunk(dl_doc=doc)
    processed_chunks: list[dict] = []

    for i, chunk in enumerate(chunk_iter):
        chunk_any = cast(Any, chunk)  # BaseChunk -> Any (por typing de docling)
        meta = cast(Any, chunk_any.meta)  # BaseMeta  -> Any

        text = chunk_any.text

        doc_items = list(getattr(meta, "doc_items", []) or [])
        headings = list(getattr(meta, "headings", []) or [])

        page_numbers = sorted(
            {
                prov.page_no
                for item in doc_items
                for prov in getattr(item, "prov", []) or []
                if hasattr(prov, "page_no")
            }
        )

        element_type = getattr(doc_items[0], "label", "text") if doc_items else "text"
        doc_name = doc.name

        # _count_chunk_tokens en tu versión está tipado para DocChunk,
        # así que evitamos el error de tipos con cast(Any, ...)
        token_count = chunker._count_chunk_tokens(doc_chunk=cast(Any, chunk_any))

        processed_chunks.append(
            {
                "chunk_id": f"{doc_name}#chunk{i}",
                "text": text,
                "metadata": {
                    "headings": headings,
                    "page_numbers": page_numbers,
                    "element_type": element_type,
                    "parent_doc": doc_name,
                    "token_count": token_count,
                },
            }
        )

    return processed_chunks
