# src/extraction/ie.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from institutional_graphrag.graph.schema import Entity, Relationship, Documento, Chunk, PRIMER_CHUNK, SIGUIENTE_CHUNK, DE_DOCUMENTO

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
DOCUMENTS_DIR = DATA_DIR / "corpus"
CHUNKS_DIR = DATA_DIR / "chunks"
INPUT_DIR = DATA_DIR / "entities_relations"


@dataclass
class ExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


def extract_documents(res: ExtractionResult) -> Dict[str, Any]:
    if not DOCUMENTS_DIR.exists():
        res.errors.append(
            {
                "type": "MissingFolder",
                "message": f"No existe la carpeta: {DOCUMENTS_DIR.as_posix()}",
            }
        )
        return res

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file():
            continue

        # value sin extensión (stem). La extensión la guardamos si querés en value como dict.
        base_name = path.stem

        # id único (como pediste) + prefijo de tipo para el grafo
        document_id = f"{base_name}_{uuid.uuid4()}"

        # Si querés guardar más info, podés pasar un dict como value
        # value = {"name": base_name, "filename": path.name, "ext": path.suffix.lower(), "path": path.as_posix()}
        value = base_name

        res.entities.append(
            Documento(
                id=document_id,
                value=value,
            )
        )

    return res

def extract_chunks(res:ExtractionResult) -> Dict[str, Any]:
    if not CHUNKS_DIR.exists():
        res.errors.append(
            {
                "type": "MissingFolder",
                "message": f"No existe la carpeta: {CHUNKS_DIR.as_posix()}",
            }
        )
        return res

    for path in sorted(CHUNKS_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() != ".json":
            continue

        # leer el json
        payload = json.loads(path.read_text(encoding="utf-8"))
        
        # base_name: gi_2010_152_informe (sin _chunks)
        source = payload.get("source", "")
        if not isinstance(source, str) or source == "":
            res.errors.append({
                "type": "InvalidChunksFile",
                "message": f"'chunks' no es lista en {path.name}",
            })
            continue
        source_stem = Path(source).stem
    
        chunks_list = payload.get("chunks", [])

        if not isinstance(chunks_list, list):
            res.errors.append({
                "type": "InvalidChunksFile",
                "message": f"'chunks' no es lista en {path.name}",
            })
            continue
        
        # por cada chunk dentro del archivo
        chunk_id = 0
        for c in chunks_list:
            if not isinstance(c, dict):
                res.errors.append({
                    "type": "InvalidChunk",
                    "message": f"Chunk no es dict en {path.name}: {repr(c)[:200]}",
                })
                continue

            idx = c.get("index")
            if not isinstance(idx, int):
                res.errors.append({
                    "type": "InvalidChunkIndex",
                    "message": f"Chunk sin index int en {path.name}: {repr(c)[:200]}",
                })
                continue
            
            chunk_anterior = chunk_id
            chunk_id = f"{source_stem}_{idx}_{uuid.uuid4()}"
            
            if idx == 0: 
                document = [e for e in res.entities if (e.label == "Documento" and source_stem == e.value)]
                if document == [] :
                    res.errors.append({
                        "type": "InvalidChunkFile",
                        "message": f"Chunk sin documento asociado {path.name}",
                    })
                    break
                relation = PRIMER_CHUNK(document[0].id, chunk_id)
            else:
                relation = SIGUIENTE_CHUNK(chunk_anterior, chunk_id)
            
            relation_document = DE_DOCUMENTO(chunk_id, document[0].id)

            res.relationships.append(relation)    
            res.relationships.append(relation_document)

            # value con tus campos reales
            value = {
                "section_heading": c.get("section_heading"),
                "chunk_method": c.get("chunk_method"),
                "section_doc_count": c.get("section_doc_count"),
                "tamaño_chars": c.get("tamaño_chars"),
                "page_content": c.get("page_content"),
            }

            res.entities.append(
                Chunk(
                    id=chunk_id,
                    value=value,
                )
            )

    return res


def _result_to_json(res: ExtractionResult) -> Dict[str, Any]:
    return {
        "entities": [e.to_dict() for e in res.entities],
        "relationships": [r.to_dict() for r in res.relationships],
        "errors": res.errors,
    }


def _write_output(res: ExtractionResult, filename: str) -> Dict[str, Any]:

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    out = _result_to_json(res)

    out_path = INPUT_DIR / filename
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out


if __name__ == "__main__":
    res = ExtractionResult(entities=[], relationships=[], errors=[])
    res = extract_documents(res)
    res = extract_chunks(res)

    _write_output(res, "Entity_documents.json")

    print("Entidades:", len(res.entities))
    print("Relaciones:", len(res.relationships))