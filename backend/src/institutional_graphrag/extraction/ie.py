# src/extraction/ie.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import re
from institutional_graphrag.graph.schema import Entity, Relationship, Documento, Chunk, PRIMER_CHUNK, SIGUIENTE_CHUNK, DE_DOCUMENTO

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
DOCUMENTS_DIR = DATA_DIR / "corpus"
CHUNKS_DIR = DATA_DIR / "chunks"
TABLE_DIR = DATA_DIR / "tables"
INPUT_DIR = DATA_DIR / "entities_relations"


PATTERN_DOCUMENT= re.compile(
    r"^(?P<group>[A-Za-z]+)_(?P<year>\d{4})_(?P<doc_id>\d+)_(?P<kind>informe|propuesta|resumen)$",
    re.IGNORECASE,
)

PATTERN_TABLE = re.compile(
    r"^(?P<group>[^_]+)_(?P<year>\d{4})_.*$",
    re.IGNORECASE
)

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
        m = PATTERN_DOCUMENT.match(base_name)
        if not m:
            res.errors.append(
                {
                    "type": "Document Invalid",
                    "message": f"El formato del documento es invalido: {base_name}",
                }
            )
            continue

        value = {
                "base_name": base_name,
                "is_group": m.group("group"),
                "year_publisher": m.group("year"),
                "sub_id": m.group("doc_id"),
                "type": m.group("kind"),
            }

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
                document = [e for e in res.entities if (e.label == "Documento" and source_stem == e.value["base_name"])]
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


def extract_proyect(res:ExtractionResult) -> Dict[str, Any]:
    datasets, res = associate_tables_with_documents(res)

    doc_by_id = {str(e.id): e for e in res.entities if e.label == "Documento"}
    print(datasets)
    for df in datasets:
        cols = df.columns
        id_col = df.columns[0]
        title_cols = [
            c for c in cols
            if "TITULO" in c.upper() or "TÍTULO" in c.upper()
        ]   
        if not title_cols:
            continue
        title_col = title_cols[0]
        small = df[[id_col, title_col]].copy()
        small = small.dropna(subset=[id_col, title_col])

        for _, row in small.iterrows():
            doc_id = str(row[id_col]).strip()
            title = str(row[title_col]).strip()
            
            doc = doc_by_id.get(doc_id)
            if doc is None:
                res.errors.append({
                    "type": "MissingDocument",
                    "message": f"No se encontró Documento con id={doc_id}",
                })
                continue
            
            # ACA YA SE OBTUVO EL TITULO , O PARTE DE EL, ENTONCES AHORA FALTA IR POR LOS CHUNKS Y ENCONTRAR EL TITULO COMPLETO AYUDANDOME DE LA FRACCION DEL TITULO OBTENIDA

        # 1) guardar el título en el documento (simple)
        #doc.value["title"] = title
    return res


def associate_tables_with_documents(res:ExtractionResult)->Tuple[List[pd.DataFrame], ExtractionResult]:

    if not TABLE_DIR.exists():
        res.errors.append(
            {
                "type": "MissingFolder",
                "message": f"No existe la carpeta: {TABLE_DIR.as_posix()}",
            }
        )
        return [], res
    
    datasets = []
    for path in sorted(TABLE_DIR.iterdir()):
        
        if not path.is_file():
            continue

        if path.suffix.lower() != ".parquet":
            continue
        
        m = PATTERN_TABLE.match(path.stem)
        if not m:
            res.errors.append(
                {
                    "type": "Table Invalid",
                    "message": f"El formato de la tabla es invalido: {path.stem}",
                }
            )
            continue
        group = m.group("group")
        year = m.group("year")
        documents = [e for e in res.entities if (e.label == "Documento" and group == e.value["is_group"] and year ==e.value["year_publisher"])]
        
        if documents == [] :
            res.errors.append({
                "type": "InvalidTable",
                "message": f"Table sin documentos asociado {path.name}",
            })
            continue

        #ids = str(doc.value["sub_id"]) for doc in documents]
        id_to_document = [(str(doc.value["sub_id"]), str(doc.id))for doc in documents]
        df = pd.read_parquet(path)
        if df.empty:
            res.errors.append(
                {
                    "type": "Table Invalid",
                    "message": f"la tabla esta vacia: {path.stem}",
                }
            )
            continue

        first_col = df.columns[0]

        if "ID" not in first_col.upper():
            res.errors.append({
                    "type": "Table Invalid",
                    "message": f"la tabla no tiene columna ID: {path.stem}",
            })
            continue

        df_filtered = expand_rows_by_id_mapping(df, id_to_document)
        datasets.append(df_filtered)
    return datasets, res


def expand_rows_by_id_mapping(df: pd.DataFrame, id_to_document: list[tuple[str, str]]) -> pd.DataFrame:
    """
    id_to_document: lista de (sub_id, doc_id). Puede tener sub_id repetidos.
    Devuelve un DF donde cada (sub_id, doc_id) genera una copia de las filas con sub_id,
    y reemplaza el valor de la primera columna (ID) por doc_id.
    """
    if df.empty:
        return df

    id_col = df.columns[0]

    by_sub: dict[str, pd.DataFrame] = {}
    s = df[id_col].astype(str)

    for sub_id, _doc_id in id_to_document:
        if sub_id not in by_sub:
            by_sub[sub_id] = df.loc[s == sub_id].copy()

    # construir salida, duplicando cuando haya sub_id repetidos
    out_parts: list[pd.DataFrame] = []

    for sub_id, doc_id in id_to_document:
        part = by_sub.get(sub_id)
        if part is None or part.empty:
            continue

        part2 = part.copy()
        part2[id_col] = doc_id  
        out_parts.append(part2)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=df.columns)



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
    res = extract_proyect(res)

    _write_output(res, "Entity_documents.json")

    print("Entidades:", len(res.entities))
    print("Relaciones:", len(res.relationships))