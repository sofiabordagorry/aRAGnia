# src/extraction/ie.py
from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

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

DATA_DIR = Path(__file__).resolve().parents[4] / "data"

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

PATTERN_DOCUMENT = re.compile(
    r"^(?P<group>[A-Za-z]+)_(?P<year>\d{4})_(?P<doc_id>\d+)_(?P<kind>informe|propuesta|resumen)$",
    re.IGNORECASE,
)

PATTERN_TABLE = re.compile(r"^(?P<group>[^_]+)_(?P<year>\d{4})_.*$", re.IGNORECASE)


@dataclass
class ExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


class EntityExtractor:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.documents_dir = DATA_DIR / "corpus"
        self.chunks_dir = DATA_DIR / "chunks"
        self.table_dir = DATA_DIR / "tables"
        self.input_dir = DATA_DIR / "entities_relations"
        self.res: ExtractionResult = ExtractionResult([], [], [])

        self.doc_by_basename: dict[str, Documento] = {}
        self.doc_by_id: dict[str, Documento] = {}
        self.docs_by_group_year: dict[tuple[str, str], list[Documento]] = defaultdict(list)

    def run(self) -> ExtractionResult:
        self.extract_documents()
        self._build_doc_indexes()
        self.extract_chunks()
        self.extract_projects()
        self.extract_responsible()
        return self.res

    def _build_doc_indexes(self) -> None:
        docs = [cast(Documento, e) for e in self.res.entities if e.label == "Documento"]
        self.doc_by_id = {str(d.id): d for d in docs}

        self.doc_by_basename = {}
        self.docs_by_group_year = defaultdict(list)
        for d in docs:
            base = d.value["base_name"]
            self.doc_by_basename[base] = d
            key = (d.value["is_group"], d.value["year_publisher"])
            self.docs_by_group_year[key].append(d)

    def extract_documents(self) -> None:
        if not self.documents_dir.exists():
            self.res.errors.append(
                {"type": "MissingFolder", "message": f"No existe la carpeta: {self.documents_dir}"}
            )
            return

        for path in sorted(self.documents_dir.iterdir()):
            if not path.is_file():
                continue

            base_name = path.stem
            m = PATTERN_DOCUMENT.match(base_name)

            if not m:
                self.res.errors.append(
                    {
                        "type": "Document Invalid",
                        "message": f"El formato del documento es invalido: {base_name}",
                    }
                )
                continue

            doc_id = f"{base_name}"
            value = {
                "base_name": base_name,
                "is_group": m.group("group"),
                "year_publisher": m.group("year"),
                "sub_id": m.group("doc_id"),
                "type": m.group("kind"),
            }

            self.res.entities.append(
                Documento(
                    id=doc_id,
                    value=value,
                )
            )

    def extract_chunks(self) -> None:
        if not self.chunks_dir.exists():
            self.res.errors.append(
                {
                    "type": "MissingFolder",
                    "message": f"No existe la carpeta: {self.chunks_dir.as_posix()}",
                }
            )
            return
        for path in sorted(self.chunks_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".json":
                continue

            payload = self._read_json(path)
            if payload is None:
                continue

            source = payload.get("source", "")
            chunks_list = payload.get("chunks", [])

            if not isinstance(source, str) or not isinstance(chunks_list, list):
                self.res.errors.append(
                    {
                        "type": "InvalidChunksFile",
                        "message": f"'chunks' no es lista en {path.name}",
                    }
                )
                continue

            source_stem = Path(source).stem
            doc = self.doc_by_basename.get(source_stem)
            if doc is None:
                self.res.errors.append(
                    {
                        "type": "MissingDocumentForChunks",
                        "message": f"{path.name} sin documento {source_stem}",
                    }
                )
                continue

            prev_chunk_id: Optional[str] = None
            for c in chunks_list:
                if not isinstance(c, dict):
                    self.res.errors.append(
                        {
                            "type": "InvalidChunk",
                            "message": f"Chunk no es dict en {path.name}: {repr(c)[:200]}",
                        }
                    )
                    continue

                chunk_id = c.get("chunk_id")
                if not isinstance(chunk_id, str):
                    self.res.errors.append(
                        {
                            "type": "InvalidChunkIndex",
                            "message": f"Chunk sin index int en {path.name}: {repr(c)[:200]}",
                        }
                    )
                    continue

                if prev_chunk_id is None:
                    self.res.relationships.append(PRIMER_CHUNK(doc.id, chunk_id))
                else:
                    self.res.relationships.append(SIGUIENTE_CHUNK(prev_chunk_id, chunk_id))

                prev_chunk_id = chunk_id

                self.res.relationships.append(DE_DOCUMENTO(chunk_id, doc.id))

                meta = c.get("metadata", {})

                value = {
                    "section_heading": meta.get("headings"),
                    "parent_doc": meta.get("parent_doc"),
                    "element_type": meta.get("element_type"),
                    "size_chars": meta.get("token_count"),
                    "page_content": meta.get("page_numbers"),
                }

                self.res.entities.append(Chunk(id=chunk_id, value=value))

    def extract_projects(self) -> None:
        datasets = self._associate_tables_with_documents()

        for df in datasets:
            # diccionario_proyectos = {}
            # cols = df.columns
            id_col = df.columns[0]
            title_col = next(
                (c for c in df.columns if "TITULO" in c.upper() or "TÍTULO" in c.upper()), None
            )
            if not title_col:
                continue

            small = df[[id_col, title_col]].dropna(subset=[id_col, title_col])
            projects: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for _, row in small.iterrows():
                doc_id = str(row[id_col]).strip()
                frac_title = str(row[title_col]).strip()

                doc = self.doc_by_id.get(doc_id)
                if doc is None:
                    self.res.errors.append(
                        {
                            "type": "MissingDocument",
                            "message": f"No se encontró Documento con id={doc_id}",
                        }
                    )
                    continue

                project_id = (
                    f"{doc.value['is_group']}_{doc.value['year_publisher']}_{doc.value['sub_id']}"
                )
                self.res.relationships.append(ES_DESCRITO_POR(project_id, doc_id))

                chunk_file = self.chunks_dir / f"{doc.value['base_name']}_chunks.json"
                candidate = self._search_title(chunk_file, frac_title)
                if candidate:
                    candidate["type"] = doc.value["type"]
                    candidate["year"] = doc.value["year_publisher"]
                    projects[project_id].append(candidate)

            for project_id, candidates in projects.items():
                best = self._pick_best_candidate(candidates)
                if best is None:
                    self.res.errors.append(
                        {
                            "type": "MissingCandidate",
                            "message": f"No candidates for project_id={project_id}",
                        }
                    )
                    continue
                self.res.entities.append(Proyecto(id=project_id, value=best["candidate_title"]))
                year = best.get("year")

                if not year:
                    self.res.errors.append(
                        {
                            "type": "MissingYear",
                            "message": f"El proyecto {project_id} no tiene anio de publicacion",
                        }
                    )
                else:
                    self.res.entities.append(
                        Anio(
                            id=year,
                            value=year,
                        )
                    )
                    self.res.relationships.append(INICIO_EN(project_id, year))

                self.res.relationships.append(EVIDENCIA_DE(best["chunk_id"], project_id))

        related_docs = {
            rel.target_id for rel in self.res.relationships if rel.type == "ES_DESCRITO_POR"
        }

        unrelated_docs = [doc_id for doc_id in self.doc_by_id.keys() if doc_id not in related_docs]

        candidates_for_projects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc_id in unrelated_docs:
            doc = self.doc_by_id.get(doc_id)
            if doc is None:
                self.res.errors.append(
                    {"type": "MissingDocument", "message": f"doc_by_id no contiene doc_id={doc_id}"}
                )
                continue

            chunk_path = self.chunks_dir / f"{doc.value['base_name']}_chunks.json"

            fallback_result = self._search_title(chunk_path, None)
            if not fallback_result:
                continue

            # metadata del doc
            project_id = (
                f"{doc.value['is_group']}_{doc.value['year_publisher']}_{doc.value['sub_id']}"
            )
            self.res.relationships.append(ES_DESCRITO_POR(project_id, doc_id))

            candidates_for_projects[project_id].append(
                {
                    **fallback_result,
                    "type": doc.value.get("type"),
                    "year": doc.value.get("year_publisher"),
                }
            )

        for project_id, candidates in candidates_for_projects.items():
            best = self._pick_best_candidate(candidates)
            if best is None:
                continue
            title = best.get("candidate_title")
            chunk_id = best.get("chunk_id")
            year = best.get("year")
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            self.res.entities.append(Proyecto(id=project_id, value=title.strip()))

            if year:
                self.res.entities.append(Anio(id=best["year"], value=best["year"]))
                self.res.relationships.append(INICIO_EN(project_id, best["year"]))
            else:
                self.res.errors.append(
                    {
                        "type": "MissingYear",
                        "message": f"Proyecto {project_id} sin year_publisher (fallback)",
                    }
                )

            self.res.relationships.append(EVIDENCIA_DE(chunk_id, project_id))

    def _search_title(self, path: Path, title: Optional[str]) -> Optional[dict[str, Any]]:
        if not path.exists():
            self.res.errors.append(
                {"type": "MissingFile", "message": f"No existe el archivo : {path}"}
            )
            return None

        payload = self._read_json(path)
        if payload is None:
            return None
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return None

        results: list[dict[str, Any]] = []

        if title:
            title_esc = re.escape(title)
            rx1 = re.compile(
                rf"(Titulo|Título).*?[:,\n]\s*.*?({title_esc}.*?)(?:\.|\n|$)", re.I | re.S
            )
            rx2 = re.compile(r'(Titulo|Título).*?[:,\n]\s*([^:,.\n\'"]+)', re.I)

        for i, chunk in enumerate(chunks[:4]):
            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            meta = chunk.get("metadata") or {}
            headings = meta.get("headings") or []

            if i == 0 and headings:
                results.append(
                    {"chunk_id": chunk_id, "candidate_title": headings[0], "best_grade": 6}
                )

            if title and isinstance(text, str):
                match = rx1.search(text)
                if match:
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "candidate_title": match.group(2).strip(),
                            "best_grade": 1,
                        }
                    )

                match = rx2.search(text)
                if match:
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "candidate_title": match.group(2).strip(),
                            "best_grade": 2,
                        }
                    )

                for heading in headings:
                    if title in heading:
                        results.append(
                            {"chunk_id": chunk_id, "candidate_title": heading, "best_grade": 3}
                        )

                if title in text:
                    # Si title está en el texto, capturar hasta la siguiente coma, punto o salto de línea
                    text_match = re.search(rf"{re.escape(title)}[^\n.,]*[.,\n]", text)
                    if text_match:
                        results.append(
                            {
                                "chunk_id": chunk_id,
                                "candidate_title": text_match.group(0).strip(),
                                "best_grade": 4,
                            }
                        )

                results.append({"chunk_id": chunk_id, "candidate_title": title, "best_grade": 5})

        return min(results, key=lambda x: x["best_grade"]) if results else None

    def _pick_best_candidate(self, candidatos: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not candidatos:
            return None

        def _score(x: dict[str, Any]) -> tuple[int, int]:
            # menor = mejor
            grado = int(x.get("best_grade", 99))
            tipo = TYPE_PRECEDENCIA.get(str(x.get("type", "")).lower(), 99)
            return (grado, tipo)

        return min(candidatos, key=_score)

    def extract_responsible(self) -> None:
        datasets = self._associate_tables_with_documents()
        if not datasets:
            return

        # índices globales (para performance)
        inv_ids_by_project = self._build_indexes()

        normalized_targets = {self._normalize_col(x) for x in TYPE_TABLE_NAME}

        for df in datasets:
            id_col = df.columns[0]

            responsible_cols = [
                c
                for c in df.columns
                if any(t in self._normalize_col(c) for t in normalized_targets)
            ]
            if not responsible_cols:
                continue

            # df reducido
            small = df[[id_col, *responsible_cols]].dropna(subset=[id_col])
            cols = list(small.columns)

            for _, row in small.iterrows():
                doc_id = str(row[id_col]).strip()

                doc = self.doc_by_id.get(doc_id)
                if doc is None:
                    self.res.errors.append(
                        {
                            "type": "MissingDocument",
                            "message": f"No se encontró Documento con id={doc_id}",
                        }
                    )
                    continue

                project_id = (
                    f"{doc.value['is_group']}_{doc.value['year_publisher']}_{doc.value['sub_id']}"
                )

                people = self._extract_up_to_people(row, cols, id_col)

                for full_name, fallback in people:
                    # buscar en chunks (primero full, luego fallback)
                    candidate_in_text = None
                    if full_name:
                        candidate_in_text = full_name

                    if not candidate_in_text and fallback:
                        candidate_in_text = fallback

                    if not candidate_in_text:
                        continue

                    candidate_id = str(uuid.uuid4())

                    if (
                        fallback
                        and any(fallback == value for value, _ in inv_ids_by_project[project_id])
                        and candidate_in_text == fallback
                    ) or (
                        full_name
                        and any(full_name == value for value, _ in inv_ids_by_project[project_id])
                    ):
                        continue

                    if (fallback and candidate_in_text == fallback) or (
                        full_name
                        and any(value == full_name for _, value in inv_ids_by_project[project_id])
                    ):
                        continue

                    # si existe el investigador con un nombre pero ahora aparece con nombre+apellido elimino la entidad anterior
                    if fallback:
                        to_remove = {
                            item for item in inv_ids_by_project[project_id] if item[1] == fallback
                        }
                        if to_remove:
                            inv_ids_by_project[project_id] -= to_remove
                            candidate_to_remove = next(iter(to_remove), None)
                            if candidate_to_remove is not None:
                                inv_id = candidate_to_remove[0]
                                self.res.entities = [
                                    e
                                    for e in self.res.entities
                                    if not (e.label == "Investigador" and e.id == inv_id)
                                ]
                                self.res.relationships = [
                                    r
                                    for r in self.res.relationships
                                    if r.source_id != inv_id and r.target_id != project_id
                                ]

                    self.res.entities.append(Investigador(id=candidate_id, value=candidate_in_text))
                    self.res.relationships.append(PARTICIPO_EN(candidate_id, project_id))
                    inv_ids_by_project[project_id].add((candidate_id, candidate_in_text))

    def _build_indexes(self):
        inv_ids_by_project: dict[str, set[str]] = defaultdict(set)
        for r in self.res.relationships:
            if r.type == "PARTICIPO_EN":
                responsible_by_id = {
                    str(e.value): e
                    for e in self.res.entities
                    if (e.label == "Investigador" and e.id == r.source_id)
                }
                investigador = next(iter(responsible_by_id.values()), None)
                inv_ids_by_project[str(r.target_id)].add(investigador)

        return inv_ids_by_project

    def _clean_pair(
        self, full: Optional[str], fallback: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        if full is not None:
            full = full.strip()
            if not full:
                full = None
        if fallback is not None:
            fallback = fallback.strip()
            if not fallback:
                fallback = None
        return full, fallback

    def _extract_up_to_people(
        self, row, cols: list[str], id_col: str
    ) -> list[tuple[Optional[str], Optional[str]]]:
        out: list[tuple[Optional[str], Optional[str]]] = []
        i = 0

        while i < len(cols):
            c1 = cols[i]
            if c1 == id_col:
                i += 1
                continue

            c2 = cols[i + 1] if i + 1 < len(cols) else None

            v1 = self.cell_str(row.get(c1))
            v2 = self.cell_str(row.get(c2)) if c2 else None

            # Nombre + Apellido (orden normal)
            if c2 and self.is_name_col(c1) and self.is_lastname_col(c2):
                full = f"{v1} {v2}" if (v1 and v2) else None
                fallback = v2 or v1
                out.append(self._clean_pair(full, fallback))
                i += 2
                continue

            # Apellido + Nombre (orden invertido)
            if c2 and self.is_lastname_col(c1) and self.is_name_col(c2):
                full = f"{v2} {v1}" if (v1 and v2) else None
                fallback = v1 or v2
                out.append(self._clean_pair(full, fallback))
                i += 2
                continue

            # Sueltos (Responsable / Nombre / Apellido)
            if (
                self.is_responsable_col(c1) or self.is_name_col(c1) or self.is_lastname_col(c1)
            ) and v1:
                out.append(self._clean_pair(None, v1))

            i += 1

        # eliminar pares vacíos y duplicados básicos
        out = [(a, b) for (a, b) in out if a or b]
        return out

    def _normalize_col(self, name: str) -> str:
        # mayúsculas + sin acentos + espacios simples
        if not isinstance(name, str):
            return ""

        name = name.upper()
        name = "".join(
            c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn"
        )
        return " ".join(name.split())

    def is_name_col(self, col: str) -> bool:
        c = self._normalize_col(col).upper()
        return "NOMBRE" in c or "NOMBRES" in c

    def is_lastname_col(self, col: str) -> bool:
        c = self._normalize_col(col).upper()
        return "APELLIDO" in c

    def is_responsable_col(self, col: str) -> bool:
        c = self._normalize_col(col).upper()
        return "RESPONSABLE" in c

    def cell_str(self, v) -> Optional[str]:
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        s = str(v).strip()
        return s if s else None

    def _associate_tables_with_documents(self) -> list[pd.DataFrame]:

        if not self.table_dir.exists():
            self.res.errors.append(
                {"type": "MissingFolder", "message": f"No existe: {self.table_dir}"}
            )
            return []

        datasets: list[pd.DataFrame] = []
        for path in sorted(self.table_dir.iterdir()):

            if not path.is_file() or path.suffix.lower() != ".parquet":
                continue

            match = PATTERN_TABLE.match(path.stem)
            if not match:
                self.res.errors.append(
                    {
                        "type": "Table Invalid",
                        "message": f"El formato de la tabla es invalido: {path.stem}",
                    }
                )
                continue

            key = (match.group("group"), match.group("year"))
            docs = self.docs_by_group_year.get(key)

            if not docs:
                self.res.errors.append(
                    {"type": "InvalidTable", "message": f"{path.name} sin docs {key}"}
                )
                continue

            df = pd.read_parquet(path)
            if df.empty:
                continue

            id_col = df.columns[0]

            if "ID" not in id_col.upper():
                self.res.errors.append(
                    {
                        "type": "Table Invalid",
                        "message": f"la tabla no tiene columna ID: {path.stem}",
                    }
                )
                continue

            id_to_doc = [(str(d.value["sub_id"]), str(d.id)) for d in docs]
            datasets.append(self._expand_rows_by_id_mapping(df, id_to_doc))

        return datasets

    def _expand_rows_by_id_mapping(
        self, df: pd.DataFrame, id_to_document: list[tuple[str, str]]
    ) -> pd.DataFrame:
        if df.empty:
            return df

        id_col = df.columns[0]

        by_sub: dict[str, pd.DataFrame] = {}
        s = df[id_col].astype(str)

        for sub_id, _ in id_to_document:
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

        return (
            pd.concat(out_parts, ignore_index=True)
            if out_parts
            else pd.DataFrame(columns=df.columns)
        )

    def _result_to_json(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.res.entities],
            "relationships": [r.to_dict() for r in self.res.relationships],
            "errors": self.res.errors,
        }

    def _read_json(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.res.errors.append({"type": "InvalidJson", "message": f"{path.name}: {e}"})
            return None

        if not isinstance(data, dict):
            self.res.errors.append(
                {"type": "InvalidJson", "message": f"{path.name}: JSON root is not an object"}
            )
            return None

        return cast(dict[str, Any], data)

    def save_in_file(self, filename: str) -> None:

        self.input_dir.mkdir(parents=True, exist_ok=True)

        out = self._result_to_json()
        out_path = self.input_dir / filename
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_from_json(self, filename: str) -> Optional[ExtractionResult]:
        path = self.input_dir / filename
        if not path.exists():
            return None

        data = self._read_json(path)
        if data is None:
            return None

        entities_data = data.get("entities", [])
        relationships_data = data.get("relationships", [])
        errors_data = data.get("errors", [])

        entities: list[Entity] = []
        relationships: list[Relationship] = []
        errors: list[dict[str, Any]] = errors_data if isinstance(errors_data, list) else []

        if not isinstance(entities_data, list):
            errors.append({"type": "InvalidFormat", "message": "'entities' no es lista"})
            return ExtractionResult(entities=entities, relationships=relationships, errors=errors)

        # --- reconstruir entidades ---
        for e in entities_data:
            if not isinstance(e, dict):
                errors.append(
                    {"type": "InvalidEntity", "message": f"Entidad no es dict: {repr(e)[:200]}"}
                )
                continue

            entity_id = e.get("id")
            label = e.get("label")
            value = e.get("value")
            if not isinstance(entity_id, str) or not entity_id:
                errors.append(
                    {"type": "InvalidEntity", "message": f"Entidad sin id válido: {repr(e)[:200]}"}
                )
                continue
            if not isinstance(label, str) or not label:
                errors.append(
                    {
                        "type": "InvalidEntity",
                        "message": f"Entidad sin label válido: {repr(e)[:200]}",
                    }
                )
                continue

            cls = GraphSchema.ENTITIES.get(label)

            if cls is None:
                errors.append(
                    {"type": "UnknownEntityType", "message": f"Label desconocido: {label}"}
                )
                continue

            try:
                entities.append(cls(id=entity_id, value=value))
            except Exception as exc:
                errors.append({"type": "InvalidEntity", "message": f"{label}({entity_id}): {exc}"})

        # --- reconstruir relaciones ---
        if not isinstance(relationships_data, list):
            self.res.errors.append(
                {"type": "InvalidFormat", "message": "'relationships' no es lista"}
            )
            return None

        for r in relationships_data:
            if not isinstance(r, dict):
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"Relación no es dict: {repr(r)[:200]}",
                    }
                )
                continue
            rel_type = r.get("type")
            source_id = r.get("source_id")
            target_id = r.get("target_id")
            props = r.get("properties", {})

            if not isinstance(rel_type, str) or not rel_type:
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"Relación sin type válido: {repr(r)[:200]}",
                    }
                )
                continue
            if (
                not isinstance(source_id, str)
                or not source_id
                or not isinstance(target_id, str)
                or not target_id
            ):
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"Relación sin endpoints: {repr(r)[:200]}",
                    }
                )
                continue
            if not isinstance(props, dict):
                props = {}  # normalizar

            try:
                relationships.append(
                    Relationship(
                        type=rel_type, source_id=source_id, target_id=target_id, properties=props
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"{rel_type}({source_id}->{target_id}): {exc}",
                    }
                )

        return ExtractionResult(entities=entities, relationships=relationships, errors=errors)
