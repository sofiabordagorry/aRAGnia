import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from institutional_graphrag.graph.schema import (
    DE_DOCUMENTO,
    ES_DESCRITO_POR,
    PRIMER_CHUNK,
    SIGUIENTE_CHUNK,
    Anio,
    Chunk,
    Documento,
    Entity,
    Relationship,
)


@dataclass
class ExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


@dataclass
class ReadJsonResult:
    data: Optional[dict[str, Any]]
    errors: dict[str, Any]


ALLOWED_SUFFIXES = {".csv", ".pdf"}

TYPE_TABLE_NAME = {
    "APELLIDO",
    "NOMBRES",
    "RESPONSABLE",
    "NOMBRE RESPONSABLE",
}

TYPE_PRECEDENCIA = {
    "resumen": 1,
    "informe": 2,
    "propuesta": 3,
}

TABLE_DIR = Path(__file__).resolve().parents[4] / "data" / "tables"


class RuleBasedExtractor:
    def __init__(self):
        self.datasets: list[pd.DataFrame] = []
        self.paths: list[Path] = list(Path(TABLE_DIR).glob("*.csv"))

    def cleanup(self):
        self.datasets = []

    def extract_document(
        self, path: Path, pattern, value_builder, create_year_entity: bool = False
    ) -> ExtractionResult:
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        errors: list[dict[str, Any]] = []
        if path.suffix.lower() in ALLOWED_SUFFIXES:
            base_name = path.stem

            m = pattern.match(base_name)
            if not m:
                errors.append(
                    {
                        "type": "Document Invalid",
                        "message": f"El documento no se encuentra en las tablas: {base_name}",
                    }
                )
                return ExtractionResult(entities, relationships, errors)
            groupdict = m.groupdict()
            if {"group", "year", "doc_id"}.issubset(groupdict):
                # En este punto ya se revisaron los csv
                key_project = f'{m.group("group")}_{m.group("year")}_{m.group("doc_id")}'
                in_csv = False
                for csv_file in self.paths:

                    df = pd.read_csv(csv_file)
                    if key_project in df["file_id"].astype(str).values:
                        in_csv = True
                        break

                if not in_csv:
                    errors.append(
                        {
                            "type": "Document Invalid",
                            "message": f"El identificador del documento es invalido: {base_name} y  el proyecto es {key_project}",
                        }
                    )
                    return ExtractionResult(entities, relationships, errors)

                entities.append(Documento(id=base_name, value=value_builder(base_name, m)))
            else:
                entities.append(Documento(id=base_name, value=value_builder(base_name, m)))
                key_project = base_name
                return ExtractionResult(entities, relationships, errors)

            relationships.append(ES_DESCRITO_POR(key_project, base_name))
            if create_year_entity:
                year = m.group("year")
                anio = Anio(
                    id=year,
                    value={"anio": year},
                )
                entities.append(anio)
        else:
            errors.append(
                {
                    "type": "Document Invalid",
                    "message": f"La extension del documento es invalido: {path.name}",
                }
            )

        return ExtractionResult(entities, relationships, errors)

    def extract_chunk(self, path: Path, doc_by_basename: dict[str, Documento]) -> ExtractionResult:
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        errors: list[dict[str, Any]] = []
        if not path.is_file() or path.suffix.lower() != ".json":
            return ExtractionResult(entities, relationships, errors)
        res = self._read_json(path)
        if res.errors:
            errors.append(res.errors)
            return ExtractionResult(entities, relationships, errors)

        payload = res.data
        if payload is None:
            return ExtractionResult(entities, relationships, errors)

        source = payload.get("source", "")
        chunks_list = payload.get("chunks", [])

        if not isinstance(source, str) or not isinstance(chunks_list, list):
            errors.append(
                {
                    "type": "InvalidChunksFile",
                    "message": f"'chunks' no es lista en {path.name}",
                }
            )
            return ExtractionResult(entities, relationships, errors)
        source_stem = Path(source).stem
        doc = doc_by_basename.get(source_stem)
        if doc is None:
            errors.append(
                {
                    "type": "MissingDocumentForChunks",
                    "message": f"{path.name} sin documento {source_stem}",
                }
            )
            return ExtractionResult(entities, relationships, errors)

        prev_chunk_id: Optional[str] = None
        for c in chunks_list:

            if not isinstance(c, dict):
                errors.append(
                    {
                        "type": "InvalidChunk",
                        "message": f"Chunk no es dict en {path.name}: {repr(c)[:200]}",
                    }
                )
                return ExtractionResult(entities, relationships, errors)

            chunk_id = c.get("chunk_id")
            if not isinstance(chunk_id, str):
                errors.append(
                    {
                        "type": "InvalidChunkIndex",
                        "message": f"Chunk sin index int en {path.name}: {repr(c)[:200]}",
                    }
                )
                return ExtractionResult(entities, relationships, errors)

            if prev_chunk_id is None:
                relationships.append(PRIMER_CHUNK(doc.id, chunk_id))
            else:
                relationships.append(SIGUIENTE_CHUNK(prev_chunk_id, chunk_id))

            prev_chunk_id = chunk_id

            relationships.append(DE_DOCUMENTO(chunk_id, doc.id))
            meta = c.get("metadata", {})
            if "page_numbers" in meta:
                meta["paginas"] = meta.pop("page_numbers")
            meta["texto"] = c.get("text", "")
            entities.append(Chunk(id=chunk_id, value=meta))
        return ExtractionResult(entities, relationships, errors)

    def _read_json(self, path: Path) -> ReadJsonResult:
        errors: dict[str, Any] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            errors = {"type": "InvalidJson", "message": f"{path.name}: {e}"}
            return ReadJsonResult(None, errors)

        if not isinstance(data, dict):
            errors = {"type": "InvalidJson", "message": f"{path.name}: JSON root is not an object"}
            return ReadJsonResult(None, errors)

        return ReadJsonResult(data, errors)
