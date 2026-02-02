# src/extraction/ie.py
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from institutional_graphrag.extraction.llm_extractor import (
    LLMEntityExtractor,
    create_entities_and_relationships_from_llm_extraction,
    create_topics_from_llm_extraction,
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

logger = logging.getLogger(__name__)

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

ALLOWED_SUFFIXES = {".parquet", ".pdf"}


@dataclass
class ExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


class EntityExtractor:
    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
    ):
        self.data_dir = DATA_DIR
        self.documents_dir = DATA_DIR / "corpus"
        self.chunks_dir = DATA_DIR / "chunks"
        self.table_dir = DATA_DIR / "tables"
        self.input_dir = DATA_DIR / "entities_relations"
        self.res: ExtractionResult = ExtractionResult([], [], [])

        self.doc_by_basename: dict[str, Documento] = {}
        self.doc_by_id: dict[str, Documento] = {}
        self.docs_by_group_year: dict[tuple[str, str], list[Documento]] = defaultdict(list)

        # Configuración LLM
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        self._seen_entities: set[tuple[str, str]] = set()
        self._seen_rels: set[tuple[str, str, str, str]] = set()
        self._rel_index: dict[tuple[str, str, str], int] = {}

    def run(
        self,
        max_docs: int | None = None,
    ) -> ExtractionResult:
        entities_json = DATA_DIR / "entities_relations" / "entity_documents.json"
        if entities_json.exists():
            self.load_subset_from_graph_json(
                entities_json,
                label="Investigador",
                value_filter={"source": "llm"},
            )
            self.load_subset_from_graph_json(entities_json, label="Topico")

        self.extract_documents()
        self._build_doc_indexes()
        self.extract_chunks()
        self.extract_projects()
        self.extract_responsible()
        self.extract_researchers_llm(max_docs=max_docs)
        self.extract_topics_llm(max_docs=max_docs)

        return self.res

    def load_subset_from_graph_json(
        self,
        json_path: str | Path,
        *,
        label: str,
        value_filter: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Carga al self un subgrafo del JSON:
        - Entidades con `label` y (opcional) filtros exactos dentro de `value`
        - Todas las relaciones donde aparezcan esas entidades (como source o target)

        Ejemplos:
        label="Investigador", value_filter={"source": "llm"}
        label="Topico"  (sin value_filter)
        """
        json_path = Path(json_path).resolve()
        if not json_path.is_file():
            self.res.errors.append(
                {"type": "MissingFile", "message": f"No existe el archivo: {json_path}"}
            )
            return

        with json_path.open(encoding="utf-8") as f:
            payload = json.load(f)

        entities_raw = payload.get("entities", [])
        rels_raw = payload.get("relationships", [])
        errors_raw = payload.get("errors", [])

        if isinstance(errors_raw, list):
            self.res.errors.extend([e for e in errors_raw if isinstance(e, dict)])

        if not isinstance(entities_raw, list) or not isinstance(rels_raw, list):
            self.res.errors.append(
                {
                    "type": "InvalidJson",
                    "message": "Formato inválido: entities/relationships no son listas",
                }
            )
            return

        # ---- 1) Filtrar entidades target ----
        matched_entities: list[Entity] = []
        matched_ids: set[str] = set()

        for raw in entities_raw:
            if not isinstance(raw, dict):
                continue
            if raw.get("label") != label:
                continue

            v = raw.get("value")
            if value_filter is not None:
                if not isinstance(v, dict):
                    continue
                ok = all(v.get(k) == expected for k, expected in value_filter.items())
                if not ok:
                    continue

            entity_id = raw.get("id")
            if not isinstance(entity_id, str) or not entity_id:
                continue

            cls = GraphSchema.ENTITIES.get(label)
            if cls is None:
                self.res.errors.append(
                    {"type": "UnknownEntityType", "message": f"Label desconocido: {label}"}
                )
                return

            try:
                ent = cls(id=entity_id, value=v)
            except Exception as exc:
                self.res.errors.append(
                    {"type": "InvalidEntity", "message": f"{label}({entity_id}): {exc}"}
                )
                continue

            matched_entities.append(ent)
            matched_ids.add(entity_id)

        for e in matched_entities:
            self.add_entity(e)

        for raw in rels_raw:
            if not isinstance(raw, dict):
                continue
            rel_type = raw.get("type")
            source_id = raw.get("source_id")
            target_id = raw.get("target_id")
            props = raw.get("properties") or {}

            if (
                not isinstance(rel_type, str)
                or not isinstance(source_id, str)
                or not isinstance(target_id, str)
            ):
                continue
            if source_id not in matched_ids and target_id not in matched_ids:
                continue
            if not isinstance(props, dict):
                props = {}

            try:
                self.add_relationship(
                    Relationship(
                        type=rel_type, source_id=source_id, target_id=target_id, properties=props
                    )
                )
            except Exception as exc:
                self.res.errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"{rel_type}({source_id}->{target_id}): {exc}",
                    }
                )

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

    def add_entity(self, e: Entity) -> bool:
        key = (e.label, str(e.id))

        if key in self._seen_entities:
            # reemplazar la entidad existente
            for i, existing in enumerate(self.res.entities):
                if existing.label == e.label and str(existing.id) == str(e.id):
                    self.res.entities[i] = e
                    return True

            # fallback raro (no debería pasar)
            return False

        self._seen_entities.add(key)
        self.res.entities.append(e)
        return True

    def add_relationship(self, r: Relationship) -> bool:
        key = (r.type, str(r.source_id), str(r.target_id))

        if key in self._rel_index:
            idx = self._rel_index[key]
            self.res.relationships[idx] = r
            return True

        self._rel_index[key] = len(self.res.relationships)
        self.res.relationships.append(r)
        return True

    def extract_documents(self) -> None:

        def ensure_dir(d: Path) -> bool:
            if d.exists():
                return True
            self.res.errors.append(
                {"type": "MissingFolder", "message": f"No existe la carpeta: {d}"}
            )
            return False

        def add_docs_from_dir(
            d: Path, pattern, value_builder, create_year_entity: bool = False
        ) -> None:
            for path in sorted(p for p in d.iterdir() if p.is_file()):
                if path.suffix.lower() in ALLOWED_SUFFIXES:
                    base_name = path.stem
                    m = pattern.match(base_name)
                    if not m:
                        self.res.errors.append(
                            {
                                "type": "Document Invalid",
                                "message": f"El formato del documento es invalido: {base_name}",
                            }
                        )
                        continue
                    self.add_entity(Documento(id=base_name, value=value_builder(base_name, m)))
                else:
                    self.res.errors.append(
                        {
                            "type": "Document Invalid",
                            "message": f"La extension del documento es invalido: {path.name}",
                        }
                    )
                if create_year_entity:
                    year = m.group("year")
                    anio = Anio(
                        id=year,
                        value={"year": year},
                    )
                    self.add_entity(anio)

        if not ensure_dir(self.documents_dir) or not ensure_dir(self.table_dir):
            return

        add_docs_from_dir(
            self.documents_dir,
            PATTERN_DOCUMENT,
            lambda base, m: {
                "base_name": base,
                "is_group": m.group("group"),
                "year_publisher": m.group("year"),
                "sub_id": m.group("doc_id"),
                "type": m.group("kind"),
            },
        )

        add_docs_from_dir(
            self.table_dir,
            PATTERN_TABLE,
            lambda base, m: {
                "base_name": base,
                "is_group": m.group("group"),
                "year_publisher": m.group("year"),
            },
            create_year_entity=True,
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

        all_paths = sorted(
            [p for p in self.chunks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
        )

        for path in all_paths:
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
                    self.add_relationship(PRIMER_CHUNK(doc.id, chunk_id))
                else:
                    self.add_relationship(SIGUIENTE_CHUNK(prev_chunk_id, chunk_id))

                prev_chunk_id = chunk_id

                self.add_relationship(DE_DOCUMENTO(chunk_id, doc.id))

                meta = c.get("metadata", {})

                self.add_entity(Chunk(id=chunk_id, value=meta))

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
                table_chunk_id = f"{doc.value['is_group']}_{doc.value['year_publisher']}_table_{doc.value['sub_id']}"
                self.add_relationship(ES_DESCRITO_POR(project_id, doc_id))

                chunk_file = self.chunks_dir / f"{doc.value['base_name']}_chunks.json"
                candidate = self._search_title(chunk_file, frac_title)
                if candidate:
                    candidate["type"] = doc.value["type"]
                    candidate["year"] = doc.value["year_publisher"]
                    candidate["table_chunk_id"] = table_chunk_id
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
                self.add_entity(Proyecto(id=project_id, value=best["candidate_title"]))
                year = best.get("year")

                if not year:
                    self.res.errors.append(
                        {
                            "type": "MissingYear",
                            "message": f"El proyecto {project_id} no tiene anio de publicacion",
                        }
                    )
                else:
                    self.add_relationship(INICIO_EN(project_id, year))
                self.add_relationship(
                    EVIDENCIA_DE(
                        best["chunk_id"],
                        project_id,
                        properties={
                            "evidence_text": f"Proyecto identificado en chunk {best['chunk_id']}"
                        },
                    )
                )
                self.add_relationship(
                    EVIDENCIA_DE(
                        best["table_chunk_id"],
                        project_id,
                        properties={"evidence_text": "Proyecto identificado en tabla"},
                    )
                )

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

            self.add_relationship(ES_DESCRITO_POR(project_id, doc_id))

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
            self.add_entity(Proyecto(id=project_id, value=title.strip()))

            if year:
                self.add_relationship(INICIO_EN(project_id, best["year"]))
            else:
                self.res.errors.append(
                    {
                        "type": "MissingYear",
                        "message": f"Proyecto {project_id} sin year_publisher (fallback)",
                    }
                )

            self.add_relationship(
                EVIDENCIA_DE(
                    chunk_id,
                    project_id,
                    properties={"evidence_text": f"Proyecto mencionado en {chunk_id}"},
                )
            )

        # Agregar documento tipo tabla a la relacion del proyecto

        key_re = re.compile(r"((?:gi|proy)_\d{4})_\d+")

        projects_table: list[Proyecto] = [
            cast(Proyecto, e) for e in self.res.entities if e.label == "Proyecto"
        ]

        projects_by_key: defaultdict[str, list[Proyecto]] = defaultdict(list)
        for p in projects_table:
            m = key_re.search(p.id)
            if m:
                projects_by_key[m.group(1)].append(p)

        for doc_id in unrelated_docs:
            if not doc_id.endswith("_table"):
                continue

            base_id = doc_id.removesuffix("_table")
            for p in projects_by_key.get(base_id, []):
                self.add_relationship(ES_DESCRITO_POR(p.id, doc_id))

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

                table_chunk_id = f"{doc.value['is_group']}_{doc.value['year_publisher']}_table_{doc.value['sub_id']}"

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

                    # Validar que no sea un valor inválido
                    invalid_values = ["--", "unnamed:", "n/a", "na", "s/d"]
                    if any(inv in candidate_in_text.lower() for inv in invalid_values):
                        continue

                    if (
                        fallback
                        and any(fallback == value for _, value in inv_ids_by_project[project_id])
                        and candidate_in_text == fallback
                    ) or (
                        full_name
                        and any(full_name == value for _, value in inv_ids_by_project[project_id])
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
                                self.res.relationships = [
                                    r
                                    for r in self.res.relationships
                                    if r.source_id != table_chunk_id and r.target_id != inv_id
                                ]
                    candidate_id = self.make_candidate_id(candidate_in_text)
                    if candidate_id:
                        self.add_entity(
                            Investigador(
                                id=candidate_id,
                                value={
                                    "name": candidate_in_text,
                                    "source": "static",
                                },
                            )
                        )
                        self.add_relationship(PARTICIPO_EN(candidate_id, project_id))
                        self.add_relationship(
                            EVIDENCIA_DE(
                                table_chunk_id,
                                candidate_id,
                                properties={
                                    "evidence_text": f"Investigador extraído de tabla: {candidate_in_text}"
                                },
                            )
                        )
                        inv_ids_by_project[project_id].add((candidate_id, candidate_in_text))

    def make_candidate_id(self, name: str) -> str:
        # 1) pasar a minúsculas
        s = name.lower()

        # 2) quitar acentos
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))

        # 3) reemplazar cualquier cosa que no sea letra o número por _
        s = re.sub(r"[^a-z0-9]+", "_", s)

        # 4) limpiar _ al inicio/final
        s = s.strip("_")

        return s

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

            id_to_doc = [(str(d.value["sub_id"]), str(d.id)) for d in docs if "sub_id" in d.value]
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

    def load_registry(self, path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def atomic_write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.reg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def already_run(self, path: Path, doc_id: str, entity_label: str) -> bool:
        self.reg = self.load_registry(path)
        return entity_label in self.reg.get(doc_id, [])

    def mark_success(self, path: Path, doc_id: str, entity_label: str) -> None:
        self.reg = self.load_registry(path)
        self.reg.setdefault(doc_id, [])
        if entity_label not in self.reg[doc_id]:
            self.reg[doc_id].append(entity_label)
            self.reg[doc_id].sort()
            self.atomic_write(path)

    def extract_researchers_llm(self, max_docs: int | None = None) -> None:
        """Extraer investigadores usando LLM con deduplicación por proyecto.

        Args:
            max_docs: Límite opcional de documentos a procesar.
        """
        llm_extractor = LLMEntityExtractor(
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            temperature=0.1,
            max_tokens=1024,
        )

        # Procesar cada proyecto
        projects = [e for e in self.res.entities if e.label == "Proyecto"]

        docs_processed = 0

        for project in projects:
            project_id = project.id

            # Inicializar set vacío para este proyecto
            existing_researcher_ids: set[str] = set()

            # Obtener documentos del proyecto
            project_docs = [
                r.target_id
                for r in self.res.relationships
                if r.type == "ES_DESCRITO_POR" and r.source_id == project_id
            ]

            if not project_docs:
                continue

            # Procesar chunks de cada documento del proyecto
            for doc_id in project_docs:
                # Verificar límite de documentos
                if max_docs is not None and docs_processed >= max_docs:
                    logger.info(f"[LLM Researchers] Límite de {max_docs} documentos alcanzado")
                    return

                # Skipear documentos de tabla
                if doc_id.endswith("_table"):
                    continue

                doc = self.doc_by_id.get(doc_id)
                if doc is None:
                    continue
                if self.already_run(DATA_DIR / "llm_registry.json", doc_id, "Investigador"):
                    logger.info(f"[LLM Researchers] Archivo en cache: {doc_id}")
                else:
                    # Cargar chunks del documento
                    base_name = doc.value.get("base_name", "")
                    if not base_name:
                        continue

                    chunks_file = self.chunks_dir / f"{base_name}_chunks.json"
                    if not chunks_file.exists():
                        continue

                    try:
                        # Cargar chunks del archivo
                        payload = self._read_json(chunks_file)
                        if payload is None:
                            continue

                        chunks = payload.get("chunks", [])
                        if not isinstance(chunks, list):
                            continue

                        # Extraer investigadores usando LLM de todos los chunks
                        logger.info(
                            f"[LLM Researchers] Procesando {len(chunks)} chunks de {base_name}..."
                        )
                        llm_result = llm_extractor.extract_researchers_from_chunks(
                            chunks, max_chunks=None
                        )
                        logger.info(
                            f"[LLM Researchers] ✓ {base_name}: encontrados {len(llm_result.researchers)} investigadores, {len(llm_result.errors)} errores"
                        )

                        # Agregar errores
                        self.res.errors.extend(llm_result.errors)

                        # Crear entidades y relaciones
                        # NOTA: existing_researcher_ids resetea por proyecto
                        # Mismo investigador en docs del mismo proyecto = misma entidad
                        # Mismo investigador en diferentes proyectos = entidades distintas
                        new_entities, new_relationships = (
                            create_entities_and_relationships_from_llm_extraction(
                                llm_result, project_id, existing_researcher_ids
                            )
                        )

                        # Agregar al resultado
                        for e in new_entities:
                            self.add_entity(e)

                        for r in new_relationships:
                            self.add_relationship(r)

                        # Actualizar el set de IDs existentes para este proyecto
                        existing_researcher_ids.update(e.id for e in new_entities)

                        # Incrementar contador de documentos procesados
                        docs_processed += 1
                        self.mark_success(DATA_DIR / "llm_registry.json", doc_id, "Investigador")

                    except Exception as e:
                        self.res.errors.append(
                            {
                                "type": "LLMExtractionError",
                                "document": base_name,
                                "message": f"Error procesando documento con LLM: {str(e)}",
                            }
                        )
                        docs_processed += 1

    def extract_topics_llm(self, max_docs: int | None = None) -> None:
        """Extraer tópicos usando LLM.

        Los tópicos se extraen a nivel de chunk (Chunk TIENE_TOPICO Topico).
        Después se agregan a nivel de proyecto según frecuencia en chunks.

        Args:
            max_docs: Límite opcional de documentos a procesar.
        """
        # Obtener IDs de tópicos ya existentes globalmente
        existing_topic_ids = {e.id for e in self.res.entities if e.label == "Topico"}

        llm_extractor = LLMEntityExtractor(
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            temperature=0.1,
            max_tokens=1024,
        )

        # Procesar cada proyecto
        projects = [e for e in self.res.entities if e.label == "Proyecto"]

        docs_processed = 0

        for project in projects:
            project_id = project.id

            # Obtener documentos del proyecto
            project_docs = [
                r.target_id
                for r in self.res.relationships
                if r.type == "ES_DESCRITO_POR" and r.source_id == project_id
            ]

            if not project_docs:
                continue

            # Procesar chunks de cada documento
            for doc_id in project_docs:
                # Verificar límite de documentos
                if max_docs is not None and docs_processed >= max_docs:
                    logger.info(f"[LLM Topics] Límite de {max_docs} documentos alcanzado")
                    return

                # Skipear documentos de tabla
                if doc_id.endswith("_table"):
                    continue

                doc = self.doc_by_id.get(doc_id)
                if doc is None:
                    continue
                if self.already_run(DATA_DIR / "llm_registry.json", doc_id, "Topico"):
                    logger.info(f"[LLM Topics] Archivo en cache: {doc_id}")
                else:
                    # Cargar chunks del documento
                    base_name = doc.value.get("base_name", "")
                    if not base_name:
                        continue

                    chunks_file = self.chunks_dir / f"{base_name}_chunks.json"
                    if not chunks_file.exists():
                        continue

                    try:
                        # Cargar chunks del archivo
                        payload = self._read_json(chunks_file)
                        if payload is None:
                            continue

                        chunks = payload.get("chunks", [])
                        if not isinstance(chunks, list):
                            continue

                        # Extraer tópicos usando LLM de todos los chunks
                        logger.info(
                            f"[LLM Topics] Procesando {len(chunks)} chunks de {base_name}..."
                        )
                        llm_result = llm_extractor.extract_topics_from_chunks(
                            chunks, max_chunks=None
                        )
                        logger.info(
                            f"[LLM Topics] ✓ {base_name}: encontrados {len(llm_result.topics)} tópicos, {len(llm_result.errors)} errores"
                        )

                        # Agregar errores
                        self.res.errors.extend(llm_result.errors)

                        # Crear entidades y relaciones chunk->topico
                        new_entities, new_relationships = create_topics_from_llm_extraction(
                            llm_result, existing_topic_ids
                        )

                        # Agregar al resultado
                        for e in new_entities:
                            self.add_entity(e)

                        for r in new_relationships:
                            self.add_relationship(r)

                        # Actualizar el set de IDs globales
                        existing_topic_ids.update(e.id for e in new_entities)

                        # Incrementar contador de documentos procesados
                        docs_processed += 1
                        self.mark_success(DATA_DIR / "llm_registry.json", doc_id, "Topico")

                    except Exception as e:
                        self.res.errors.append(
                            {
                                "type": "LLMExtractionError",
                                "document": base_name,
                                "message": f"Error procesando tópicos con LLM: {str(e)}",
                            }
                        )
                        docs_processed += 1

            # Agregar relaciones proyecto->topico basadas en los chunks del proyecto
            self._aggregate_topics_for_project(project_id)

    def _aggregate_topics_for_project(self, project_id: str) -> None:
        """Agregar tópicos a nivel de proyecto basándose en los chunks.

        Cuenta los tópicos de todos los chunks del proyecto y crea
        relaciones Proyecto TIENE_TOPICO Topico para todos los tópicos mencionados.
        """
        # Obtener todos los chunks del proyecto (a través de documentos)
        project_docs = [
            r.target_id
            for r in self.res.relationships
            if r.type == "ES_DESCRITO_POR" and r.source_id == project_id
        ]

        project_chunks = set()
        for doc_id in project_docs:
            doc_chunks = [
                r.source_id  # chunk_id es el source, doc_id es el target
                for r in self.res.relationships
                if r.type == "DE_DOCUMENTO" and r.target_id == doc_id
            ]
            project_chunks.update(doc_chunks)

        # Contar tópicos de los chunks del proyecto
        topic_counts: Counter[str] = Counter()
        for chunk_id in project_chunks:
            chunk_topics = [
                r.target_id
                for r in self.res.relationships
                if r.type == "EVIDENCIA_DE" and r.source_id == chunk_id
            ]

            topic_ids = [
                tid
                for tid in chunk_topics
                if any(e.id == tid and e.label == "Topico" for e in self.res.entities)
            ]
            topic_counts.update(topic_ids)

        # Crear relaciones proyecto->topico para los top 3 tópicos más mencionados
        top_topics = topic_counts.most_common(3)

        for topic_id, count in top_topics:
            self.add_relationship(
                Relationship(
                    type="TIENE_TOPICO",
                    source_id=project_id,
                    target_id=topic_id,
                    properties={"mention_count": count},
                )
            )
