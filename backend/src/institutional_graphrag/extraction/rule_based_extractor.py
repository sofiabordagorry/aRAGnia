import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from institutional_graphrag.document_naming import (
    DOCUMENT_KIND_PRIORITY,
    PATTERN_TABLE,
    PROJECT_KEY_RE,
    build_project_id,
    build_table_chunk_id,
)
from institutional_graphrag.graph.schema import (
    DE_DOCUMENTO,
    ES_DESCRITO_POR,
    EXTRAIDO_DE,
    INICIO_EN,
    RESPONSABLE_DE,
    PARTICIPO_EN,
    PRIMER_CHUNK,
    SIGUIENTE_CHUNK,
    TITULO_EXTRAIDO_DE,
    Anio,
    Chunk,
    Documento,
    Entity,
    Investigador,
    Proyecto,
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


ALLOWED_SUFFIXES = {".parquet", ".pdf"}

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


class RuleBasedExtractor:
    def __init__(self):
        self.datasets: list[pd.DataFrame] = []

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
                        "message": f"El formato del documento es invalido: {base_name}",
                    }
                )
                return ExtractionResult(entities, relationships, errors)
            entities.append(Documento(id=base_name, value=value_builder(base_name, m)))
            if create_year_entity:
                year = m.group("year")
                anio = Anio(
                    id=year,
                    value={"year": year},
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
            meta["text"] = c.get("text", "")
            entities.append(Chunk(id=chunk_id, value=meta))
        return ExtractionResult(entities, relationships, errors)

    #  LEER TABLAS ASOCIADAS A UN CONJUNTO DE DOCUMENTOS
    def associate_tables_with_documents(
        self, docs_by_group_year: dict[tuple[str, str], list[Documento]], table_dir: Path
    ) -> ExtractionResult:
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        errors: list[dict[str, Any]] = []
        if not table_dir.exists():
            errors.append({"type": "MissingFolder", "message": f"No existe: {table_dir}"})
            return ExtractionResult(entities, relationships, errors)

        for path in sorted(table_dir.iterdir()):

            if not path.is_file() or path.suffix.lower() != ".parquet":
                continue

            match = PATTERN_TABLE.match(path.stem)
            if not match:
                errors.append(
                    {
                        "type": "Table Invalid",
                        "message": f"El formato de la tabla es invalido: {path.stem}",
                    }
                )
                continue

            key = (match.group("group"), match.group("year"))
            docs = docs_by_group_year.get(key)

            if not docs:
                errors.append({"type": "InvalidTable", "message": f"{path.name} sin docs {key}"})
                continue

            df = pd.read_parquet(path)
            if df.empty:
                continue

            id_col = df.columns[0]

            if "ID" not in id_col.upper():
                errors.append(
                    {
                        "type": "Table Invalid",
                        "message": f"la tabla no tiene columna ID: {path.stem}",
                    }
                )
                continue

            id_to_doc = [(str(d.value["sub_id"]), str(d.id)) for d in docs if "sub_id" in d.value]
            self.datasets.append(self._expand_rows_by_id_mapping(df, id_to_doc))

        return ExtractionResult(entities, relationships, errors)

    # def extract_projects(self) -> ExtractionResult:
    def extract_projects_and_responsible_from_tables(
        self, doc_by_id: dict[str, Documento], chunk_dir: Path
    ) -> ExtractionResult:
        self.res: ExtractionResult = ExtractionResult([], [], [])
        all_projects_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        inv_ids_by_project = self._build_indexes()
        normalized_targets = {self._normalize_col(x) for x in TYPE_TABLE_NAME}
        self.doc_by_id = doc_by_id

        for df in self.datasets:
            title_col = next(
                (c for c in df.columns if "TITULO" in c.upper() or "TÍTULO" in c.upper()), None
            )
            if not title_col:
                continue
            responsible_cols = [
                c
                for c in df.columns
                if any(t in self._normalize_col(c) for t in normalized_targets)
            ]
            if not responsible_cols:
                continue

            chunk = self._collect_candidates_from_table_df(
                df, title_col, responsible_cols, chunk_dir, inv_ids_by_project
            )
            for project_id, candidates in chunk.items():
                all_projects_candidates[project_id].extend(candidates)

        self._apply_projects_from_candidates(all_projects_candidates)

        # 2) fallback para docs no relacionados
        related_docs = {
            rel.target_id for rel in self.res.relationships if rel.type == "ES_DESCRITO_POR"
        }
        unrelated_docs = [doc_id for doc_id in self.doc_by_id.keys() if doc_id not in related_docs]

        fallback_candidates = self._collect_fallback_candidates_for_unrelated_docs(
            unrelated_docs, chunk_dir
        )
        self._apply_fallback_projects(fallback_candidates)

        # 3) link doc_table -> proyectos
        self._link_table_docs_to_projects(unrelated_docs)

        return self.res

    ##############################
    #       AUXILIARES
    ##############################

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

    def _expand_rows_by_id_mapping(
        self, df: pd.DataFrame, id_to_document: list[tuple[str, str]]
    ) -> pd.DataFrame:
        if df.empty:
            return df

        id_col = df.columns[0]

        by_sub: dict[str, pd.DataFrame] = {}
        s = df[id_col].astype(str)
        # eliminar .0 al final de la id en la tabla (agregado automaticamente por pd)
        s = s.str.replace(r"\.0$", "", regex=True).str.strip()

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

    #############################
    # Auxiliares para proyectos #
    #############################
    def _collect_candidates_from_table_df(
        self,
        df: pd.DataFrame,
        title_col: str,
        responsible_cols: List[str],
        chunk_dir: Path,
        inv_ids_by_project: dict[str, set[tuple[str, str]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        De un df de tabla:
        - encuentra columna titulo
        - para cada fila: mapea doc, arma project_id, busca candidate en chunks
        - devuelve dict project_id -> list[candidates]
        """
        projects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        id_col = df.columns[0]
        small = df[[id_col, title_col] + responsible_cols].dropna(subset=[id_col])
        cols = list(small.columns)
        for _, row in small.iterrows():
            doc_id = str(row[id_col]).strip()
            frac_title = self.cell_str(row.get(title_col))

            doc = self.doc_by_id.get(doc_id)
            if doc is None:
                self.res.errors.append(
                    {
                        "type": "MissingDocument",
                        "message": f"No se encontró Documento con id={doc_id}",
                    }
                )
                continue

            project_id = build_project_id(
                str(doc.value["is_group"]),
                str(doc.value["year_publisher"]),
                str(doc.value["sub_id"]),
            )
            table_chunk_id = build_table_chunk_id(
                str(doc.value["is_group"]),
                str(doc.value["year_publisher"]),
                str(doc.value["sub_id"]),
            )

            self.res.relationships.append(ES_DESCRITO_POR(project_id, doc_id))
            chunk_file = chunk_dir / f"{doc.value['base_name']}_chunks.json"
            candidate = self._search_title(chunk_file, frac_title)
            if candidate:
                candidate["type"] = doc.value["type"]
                candidate["year"] = doc.value["year_publisher"]
                candidate["table_chunk_id"] = table_chunk_id
                projects[project_id].append(candidate)
            ####
            # Investigador
            ####
            people = self._extract_up_to_people(row, cols, id_col)
            self._process_table_investigators(
                people, inv_ids_by_project, project_id, table_chunk_id
            )
        return projects

    def _search_title(self, path: Path, title: Optional[str]) -> Optional[dict[str, Any]]:
        if not path.exists():
            self.res.errors.append(
                {"type": "MissingFile", "message": f"No existe el archivo : {path}"}
            )
            return None

        res = self._read_json(path)
        if res.errors:
            self.res.errors.append(res.errors)
            return None

        payload = res.data
        if payload is None:
            return None

        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return None

        results: list[dict[str, Any]] = []

        ### Funcion para agregar un titulo candidato
        def add(chunk_id: Any, candidate_title: Any, grade: int) -> None:
            if chunk_id is None or candidate_title is None:
                return
            results.append(
                {"chunk_id": chunk_id, "candidate_title": candidate_title, "best_grade": grade}
            )

        ###

        rx1 = rx2 = None
        title_esc = None
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
                add(chunk_id, headings[0], 6)

            if not (title and isinstance(text, str)):
                continue

            m = rx1.search(text) if rx1 else None
            if m:
                add(chunk_id, m.group(2).strip(), 1)

            m = rx2.search(text) if rx2 else None
            if m:
                add(chunk_id, m.group(2).strip(), 2)

            for h in headings:
                if title in h:
                    add(chunk_id, h, 3)

            if title in text:
                tm = re.search(rf"{title_esc}[^\n.,]*[.,\n]", text) if title_esc else None
                if tm:
                    add(chunk_id, tm.group(0).strip(), 4)

            add(chunk_id, title, 5)

        return min(results, key=lambda x: x["best_grade"]) if results else None

    def _apply_projects_from_candidates(self, projects: dict[str, list[dict[str, Any]]]) -> None:
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
            self._add_project_from_best(project_id, best)

    def _pick_best_candidate(self, candidatos: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not candidatos:
            return None

        def _score(x: dict[str, Any]) -> tuple[int, int]:
            # menor = mejor
            grado = int(x.get("best_grade", 99))
            tipo = DOCUMENT_KIND_PRIORITY.get(str(x.get("type", "")).lower(), 99)
            return (grado, tipo)

        return min(candidatos, key=_score)

    def _add_project_from_best(self, project_id: str, best: dict[str, Any]) -> None:
        # mismo comportamiento que tu bloque
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
            self.res.relationships.append(INICIO_EN(project_id, year))

        self.res.relationships.append(
            TITULO_EXTRAIDO_DE(
                project_id,
                best["chunk_id"],
                properties={"evidence_text": f"Proyecto identificado en chunk {best['chunk_id']}"},
            )
        )
        self.res.relationships.append(
            TITULO_EXTRAIDO_DE(
                project_id,
                best["table_chunk_id"],
                properties={"evidence_text": "Proyecto identificado en tabla"},
            )
        )

    def _collect_fallback_candidates_for_unrelated_docs(
        self,
        unrelated_docs: list[str],
        chunk_dir: Path,
    ) -> dict[str, list[dict[str, Any]]]:
        candidates_for_projects: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for doc_id in unrelated_docs:
            doc = self.doc_by_id.get(doc_id)
            if doc is None:
                self.res.errors.append(
                    {"type": "MissingDocument", "message": f"doc_by_id no contiene doc_id={doc_id}"}
                )
                continue
            chunk_path = chunk_dir / f"{doc.value['base_name']}_chunks.json"

            fallback_result = self._search_title(chunk_path, None)
            if not fallback_result:
                continue

            project_id = build_project_id(
                str(doc.value["is_group"]),
                str(doc.value["year_publisher"]),
                str(doc.value["sub_id"]),
            )

            # misma relación que tenías
            self.res.relationships.append(ES_DESCRITO_POR(project_id, doc_id))

            candidates_for_projects[project_id].append(
                {
                    **fallback_result,
                    "type": doc.value.get("type"),
                    "year": doc.value.get("year_publisher"),
                }
            )

        return candidates_for_projects

    def _apply_fallback_projects(
        self, candidates_for_projects: dict[str, list[dict[str, Any]]]
    ) -> None:
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
                self.res.relationships.append(INICIO_EN(project_id, best["year"]))
            else:
                self.res.errors.append(
                    {
                        "type": "MissingYear",
                        "message": f"Proyecto {project_id} sin year_publisher (fallback)",
                    }
                )

            self.res.relationships.append(
                TITULO_EXTRAIDO_DE(
                    project_id,
                    chunk_id,
                    properties={"evidence_text": f"Proyecto mencionado en {chunk_id}"},
                )
            )

    def _link_table_docs_to_projects(self, unrelated_docs: list[str]) -> None:
        projects_table: list[Proyecto] = [
            cast(Proyecto, e) for e in self.res.entities if e.label == "Proyecto"
        ]

        projects_by_key: defaultdict[str, list[Proyecto]] = defaultdict(list)
        for p in projects_table:
            m = PROJECT_KEY_RE.search(p.id)
            if m:
                projects_by_key[m.group(1)].append(p)

        for doc_id in unrelated_docs:
            if not doc_id.endswith("_table"):
                continue

            base_id = doc_id.removesuffix("_table")
            for p in projects_by_key.get(base_id, []):
                self.res.relationships.append(ES_DESCRITO_POR(p.id, doc_id))

    ################################
    # Auxiliares para Responsables #
    ################################

    def _build_indexes(self) -> dict[str, set[tuple[str, str]]]:
        inv_ids_by_project: dict[str, set[tuple[str, str]]] = defaultdict(set)

        # Si no hay investigadores aún, va a quedar vacío. Está bien.
        investigators_by_id: dict[str, str] = {
            str(e.id): str(e.value.get("name", "")) if isinstance(e.value, dict) else str(e.value)
            for e in self.res.entities
            if e.label == "Investigador"
        }

        for r in self.res.relationships:
            if r.type not in ["PARTICIPO_EN", "RESPONSABLE_DE"]:
                continue

            inv_id = str(r.source_id)
            proj_id = str(r.target_id)

            name = investigators_by_id.get(inv_id)
            if name:
                inv_ids_by_project[proj_id].add((inv_id, name))

        return dict(inv_ids_by_project)

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
        return "NOMBRE" in c

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

    def _process_table_investigators(
        self,
        people: list[tuple[Optional[str], Optional[str]]],
        inv_ids_by_project: dict[str, set[tuple[str, str]]],
        project_id: str,
        table_chunk_id: str,
    ):
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
            invalid_values = ["--", "n/a", "na", "s/d"]
            invalid_contains = ["unnamed:"]
            candidate_lower = candidate_in_text.lower()
            if candidate_lower in invalid_values or any(
                sub in candidate_lower for sub in invalid_contains
            ):
                continue

            current = inv_ids_by_project.get(project_id, set())
            if (
                fallback
                and any(name == fallback for _, name in current)
                and candidate_in_text == fallback
            ) or (full_name and any(name == full_name for _, name in current)):
                continue

            # si existe el investigador con un nombre pero ahora aparece con nombre+apellido elimino la entidad anterior
            if fallback:
                current = inv_ids_by_project.setdefault(project_id, set())
                to_remove = {item for item in current if item[1] == fallback}
                if to_remove:
                    inv_ids_by_project[project_id] -= to_remove

                    for candidate_to_remove in to_remove:
                        inv_id = candidate_to_remove[0]

                        self.res.entities = [
                            e
                            for e in self.res.entities
                            if not (e.label == "Investigador" and e.id == inv_id)
                        ]
                        self.res.relationships = [
                            r
                            for r in self.res.relationships
                            if not (r.source_id == inv_id and r.target_id == project_id)
                        ]
                        self.res.relationships = [
                            r
                            for r in self.res.relationships
                            if not (r.source_id == table_chunk_id and r.target_id == inv_id)
                        ]

            candidate_id = self.make_candidate_id(candidate_in_text)
            if candidate_id:
                self.res.entities.append(
                    Investigador(
                        id=candidate_id,
                        value={
                            "name": candidate_in_text,
                            "source": "rule_based",
                        },
                    )
                )
                self.res.relationships.append(PARTICIPO_EN(candidate_id, project_id))
                self.res.relationships.append(RESPONSABLE_DE(candidate_id, project_id))
                self.res.relationships.append(
                    EXTRAIDO_DE(
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
