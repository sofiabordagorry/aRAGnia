"""Script para procesar y chunkear todos los documentos del corpus.

Usa los JSONs ya procesados por Docling (en data/docling/) para generar los chunks. Guarda los resultados en data/chunks/ (esto no se va a guardar en el pipeline final, solo es parte del desarrollo)
"""

import json
from pathlib import Path

from langchain_core.documents import Document

from institutional_graphrag.ingest.docling_parse import load_parsed_document
from institutional_graphrag.ingest.chunker import chunk_documents


def main():
    docling_dir = Path("data/docling")
    output_dir = Path("data/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not docling_dir.exists():
        print(f"No existe {docling_dir}")
        print("   Primero ejecutar: python scripts/test_docling_manual.py data/corpus")
        return
    
    json_files = list(docling_dir.glob("*.json"))
    if not json_files:
        print(f"No hay archivos JSON en {docling_dir}")
        print("   Primero ejecutar: python scripts/test_docling_manual.py data/corpus")
        return
    
    print(f"Leyendo documentos procesados: {docling_dir}")
    print(f"Output: {output_dir}")
    print(f"Archivos encontrados: {len(json_files)}")
    print("=" * 60)
    
    total_docs = 0
    total_chunks = 0
    processed = 0
    errors = 0
    
    for json_file in sorted(json_files):
        try:
            # Cargar JSON procesado por Docling
            data = load_parsed_document(json_file)
            
            # Reconstruir Documents de LangChain
            documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in data["documents"]
            ]
            
            # Aplicar chunking
            chunks = chunk_documents(documents, max_chunk_size=2000)
            
            # Guardar chunks en JSON
            output_file = output_dir / f"{json_file.stem}_chunks.json"
            
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "index": i,
                    "section_heading": chunk.metadata.get("section_heading"),
                    "chunk_method": chunk.metadata.get("chunk_method"),
                    "section_doc_count": chunk.metadata.get("section_doc_count", 1),
                    "page_content": chunk.page_content,
                    "tamaño_chars": len(chunk.page_content),
                })
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "source": data["source"],
                    "original_docs": len(documents),
                    "total_chunks": len(chunks),
                    "chunks": chunks_data
                }, f, ensure_ascii=False, indent=2)
            
            total_docs += len(documents)
            total_chunks += len(chunks)
            processed += 1
            
            print(f"{json_file.name}: {len(documents)} docs → {len(chunks)} chunks")
            
        except Exception as e:
            errors += 1
            print(f"✗ {json_file.name}: {e}")
    
    print("=" * 60)
    print(f"Resumen:")
    print(f"   Archivos procesados: {processed}")
    print(f"   Errores: {errors}")
    print(f"   Documentos originales: {total_docs}")
    print(f"   Chunks generados: {total_chunks}")
    if total_docs > 0:
        print(f"   Reducción: {total_docs} → {total_chunks} ({total_chunks/total_docs*100:.1f}%)")


if __name__ == "__main__":
    main()
