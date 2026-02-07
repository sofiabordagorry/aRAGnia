from pathlib import Path
from typing import List
from fastapi import UploadFile, File
from institutional_graphrag.ingest.embedder import E5Embedder
from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
from institutional_graphrag.ingest.file_namer import classify_pdf, PdfKind, save_temp_file
import json
from institutional_graphrag.extraction.ie import EntityExtractor
from institutional_graphrag.graph.builder import load_graph_json
from institutional_graphrag.graph.builder import GraphBuilder
from pathlib import Path
import os
from dotenv import load_dotenv
from institutional_graphrag.extraction.static_extractor import StaticExtractor
from institutional_graphrag.ingest.postprocess_entities import (
    consolidate_researchers,
    add_missing_evidence_text
)
from typing import Optional, Any, Dict, List

from institutional_graphrag.config import EMBED_MODEL_ID
from institutional_graphrag.ingest.table_extractors import extract_table
import re
from docling_core.types.doc import DoclingDocument
import numpy as np
from typing import Iterable
import asyncio
from io import BytesIO
from institutional_graphrag.ingest.docling_parser import (
    parse_corpus,
    parse_single_document,
    DEFAULT_DOCLING_DIR,
    DocumentAlreadyProcessed,
)

from institutional_graphrag.graph.schema import (
    DE_DOCUMENTO,
    ES_DESCRITO_POR,
    EVIDENCIA_DE,
    INICIO_EN,
    PARTICIPO_EN,
    PRIMER_CHUNK,
    SIGUIENTE_CHUNK,
    Anio,
    Chunk,
    Documento,
    Entity,
    GraphSchema,
    Investigador,
    Proyecto,
    Relationship,
)

from fastapi import File, Form, UploadFile
from typing import List
import json
from pydantic import BaseModel
from enum import Enum

class PdfKind(Enum):
    TABULAR = "tabular"
    NARRATIVE = "narrative"

def looks_tabular(fileName: str) -> bool:
    if any(k in fileName for k in ("informe", "propuesta", "resumen")):
        return False
    return True

PATTERN_DOCUMENT = re.compile(
    r"^(?P<group>[A-Za-z]+)_(?P<year>\d{4})_(?P<doc_id>\d+)_(?P<kind>informe|propuesta|resumen)$",
    re.IGNORECASE,
)

PATTERN_TABLE = re.compile(r"^(?P<group>[^_]+)_(?P<year>\d{4})_.*$", re.IGNORECASE)


def classify_pdf(fileName: str) -> PdfKind:
    return PdfKind.TABULAR if looks_tabular(fileName) else PdfKind.NARRATIVE
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def make_uploadfiles(paths: Iterable[str | Path]) -> List[UploadFile]:
    files: List[UploadFile] = []
    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"No existe o no es archivo: {path}")

        data = path.read_bytes()  # lee el archivo real del disco
        uf = UploadFile(filename=path.name, file=BytesIO(data))
        files.append(uf)

    return files

class FileMeta(BaseModel):
    logical_path: str  # ej: "\1_GRUPOS I+D_2010_2014_2018\2010_Informes...\Negreira.pdf"
    group: str
    year: int
    identifier: str   # ej: "1445"
    doc_type: str     # ej: "Resumen_publicable"

async def main(
    files: List[UploadFile],
    meta_dict: Optional[Dict[str, Any]] = None,
):
    """
    Recibe multipart/form-data con field 'files' (mismo que tu JS)
    Guarda en data/uploads y devuelve resumen.
    """

    print("\n" + "=" * 80)
    print("INICIO UPLOAD PIPELINE")
    print("=" * 80)

    # -------------------------------------------------
    # Crear carpetas
    # -------------------------------------------------
    corpus_dir = DATA_DIR / "corpus"
    docling_dir = DATA_DIR / "docling"
    chunks_dir = DATA_DIR / "chunks"
    embedding_dir = DATA_DIR / "embeddings"

    for d in [corpus_dir, docling_dir, chunks_dir, embedding_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Dir asegurado: {d}")

    tokenizer = EMBED_MODEL_ID
    print(f"Tokenizer: {tokenizer}")

    shared_chunker = get_native_chunker(tokenizer=tokenizer)
    embedder = E5Embedder()

    if not files:
        print("No se enviaron archivos")
        return {"error": "No se enviaron archivos"}

    print(f"\nArchivos recibidos: {len(files)}")

    saved = []
    output_dir_tables = DATA_DIR / "tables"
    static_extractor = StaticExtractor()
    entity_extractor = EntityExtractor()
    # =========================================================
    # LOOP PRINCIPAL
    # =========================================================
    for f in files:
        if f.filename not in meta_dict:
            return {"error": f"Falta metadata para {f.filename}"}

        m = FileMeta.model_validate(meta_dict[f.filename])
        print("\n" + "-" * 60)
        print(f"Procesando: {f.filename}")
        print("-" * 60)


        # -------------------------------------------------
        # Generar nombre del archivo
        # -------------------------------------------------
        content = await f.read()
        new_filename = f"{m.group}_{m.year}_{m.identifier}_{m.doc_type}{Path(f.filename).suffix}"
        file_path = corpus_dir / new_filename
        print(f"➡ Guardando en: {file_path}")

        if file_path.exists():
            print("Ya existe → se omite")
            continue

        tmp_path = save_temp_file(content, new_filename)
        try:
            kind = classify_pdf(new_filename)

            if kind == PdfKind.TABULAR:
                print(f"Descargando: {f.filename}")
                extract_table(tmp_path, output_dir_tables)
            else:
                print(f"Descargando: {f.filename} -> Guardando como: {new_filename}")
                with file_path.open("wb") as out:
                    out.write(content)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        print("Guardado en corpus")
        if kind == PdfKind.NARRATIVE:
            # -------------------------------------------------
            # DOC LING
            # -------------------------------------------------
            print("Ejecutando Docling...")
            
            doc_dict = parse_single_document(file_path)

            filename = doc_dict.get("name", "sin_nombre")
            print(f"Docling name: {filename}")

            output_path = docling_dir / f"{filename}.json"

            json_output = json.dumps(doc_dict, indent=2, ensure_ascii=False)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_output)

            print(f"Docling guardado → {output_path}")

            # -------------------------------------------------
            # CHUNKING
            # -------------------------------------------------
            print("Chunking...")

            doc = DoclingDocument.model_validate(doc_dict)
            chunks = chunk_document(doc=doc, chunker=shared_chunker)

            print(f"Total chunks generados: {len(chunks)}")

            output_file = chunks_dir / f"{filename}_chunks.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": doc.name,
                        "total_chunks": len(chunks),
                        "tokenizer": tokenizer,
                        "chunks": chunks,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"✅ Chunks guardados → {output_file}")

        # -------------------------------------------------
        # EMBEDDINGS
        # -------------------------------------------------
        print("Generando embeddings...")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks = data.get("chunks", [])
            if not chunks:
                print(f"{output_file.name}: no tiene chunks")
                continue

            chunk_content = [c["text"] for c in chunks]

            embeddings = embedder.embed_passages(chunk_content)

            # Guardar embeddings
            np.save(embedding_dir / f"{filename}.npy", embeddings)

            # Se guardan tambien los chunks como metadata de los embeddings
            # Esto evita referencias erroneas y hace que el retrieval requiera menos parseo
            with open(embedding_dir / f"{filename}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            print(f" Embeddings guardados → {embedding_dir}")

        except Exception as e:
            print(f"✗ {filename}: {e}")


        # -------------------------------------------------
        # EXTRACCION DE ENTIDADES
        # -------------------------------------------------
            # Extraer Documento
            meta = {
                "base_name": filename,
                "is_group": m.group,
                "year_publisher": m.year,
                "sub_id": m.identifier,
                "type": m.doc_type,
            }
            pattern = PATTERN_TABLE
            if kind == PdfKind.NARRATIVE:
                pattern = PATTERN_DOCUMENT
            
            res = static_extractor.extract_document(
                path=file_path,
                pattern=pattern,
                value_builder=lambda *_: meta,
                create_year_entity=True,
            )
            entity_extractor.add_entities(res.entities)
            entity_extractor.add_relationship(res.relationships)
            entity_extractor.res.errors.extend(res.errors)
            # Extraer Chunk
            entity_extractor._build_doc_indexes()
            res = static_extractor.extract_chunk(file_path, entity_extractor.doc_by_basename)
            entity_extractor.add_entities(res.entities)
            entity_extractor.add_relationship(res.relationships)
            entity_extractor.res.errors.extend(res.errors)
            saved.append(filename)

    # Extraer Proyecto y responsable
    res = static_extractor.associate_tables_with_documents(entity_extractor.docs_by_group_year)
    entity_extractor.res.errors.extend(res.errors)

    # Extraer con LLMS 
    
    res = static_extractor.extract_projects_and_responsible_from_tables(entity_extractor.doc_by_id, entity_extractor.chunks_dir)
    entity_extractor.add_entities(res.entities)
    entity_extractor.add_relationship(res.relationships)
    entity_extractor.res.errors.extend(res.errors)
    entity_extractor._extract_with_llm()

    # Normalizar entidades
    entities, relationships, _ = consolidate_researchers(
        entity_extractor.res.entities,
        entity_extractor.res.relationships
    )
    relationships, _ = add_missing_evidence_text(relationships)
    entity_extractor.res.entities = entities
    entity_extractor.res.relationships = relationships
    # Guardado temporal para ver errores
    first_key, _ = next(iter(entity_extractor.doc_by_id.items()))
    filename = f"entity_extraction_web_{first_key}"
    entity_extractor.save_in_file(filename)

    filename = f"{filename}.json"


    # GUARDAR ENTIDAD EN BASE DE DATOS. 
    neo4j_host = os.getenv("NEO4J_HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")

    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"
    grafo = GraphBuilder(
    neo4j_uri,
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASSWORD"),
    )
    path = DATA_DIR / "entities_relations" / filename
    entities, relationships, db_merge = load_graph_json(path)
    grafo.ingest(entities=entities, relationships=relationships)

    for old_id, new_id in db_merge:
        stats = grafo.backend.merge_node_id(
            label="Investigador",
            old_id=old_id,
            new_id=new_id,
            sample_ids=10,
        )

        print(
            "MERGE Investigador:",
            f"{old_id} -> {new_id}",
            "nodes_created=", stats.nodes_created,
            "nodes_deleted=", stats.nodes_deleted,
            "rels_created=", stats.relationships_created,
            "rels_deleted=", stats.relationships_deleted,
            "props_set=", stats.properties_set,
            "sample_node_eids=", stats.node_eids[:10],
            "sample_rel_eids=", stats.rel_eids[:10],
        )
    print("\n" + "=" * 80)
    print("PIPELINE FINALIZADO")
    print("Archivos procesados:", saved)
    print("=" * 80)

    return {"processed": saved}

if __name__ == "__main__":
    paths = [
        DATA_DIR / "prueba" / "624_documento_documentosrequeridos.pdf",
        DATA_DIR / "prueba" / "2465_documento_avales.pdf"
    ]

    meta_dict = {
        "2465_documento_avales.pdf": {
            "logical_path": r"\...\2465_documento_avales.pdf",
            "group": "Proyecto",
            "year": 2020,
            "identifier": "69",
            "doc_type": "propuesta",
        },
        "624_documento_documentosrequeridos.pdf": {
            "logical_path": r"\...\624_documento_documentosrequeridos.pdf",
            "group": "Proyecto",
            "year": 2020,
            "identifier": "69",
            "doc_type": "informe",
        },
    }

    files = make_uploadfiles(paths)

    res = asyncio.run(main(files, meta_dict=meta_dict))
    print(res)