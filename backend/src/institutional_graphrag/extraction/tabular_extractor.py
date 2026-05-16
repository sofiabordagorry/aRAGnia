"""Extracción de investigadores desde datos tabulares CSV."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from institutional_graphrag.document_naming import build_project_id
from institutional_graphrag.graph.schema import (
    EXTRAIDO_DE,
    PARTICIPO_EN,
    Entity,
    Investigador,
    Proyecto,
    Relationship,
)

@dataclass
class TabularExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


class TabularExtractor:
    def extract_from_directory(self, table_dir: Path) -> TabularExtractionResult:
        """Extrae investigadores y proyectos de todos los CSV del directorio.

        Si hay varios CSVs (ej: distintos años), mergea los resultados deduplicando
        por ID: un mismo investigador presente en dos CSVs queda como una sola entidad
        con todas sus relaciones PARTICIPO_EN acumuladas.
        """
        relationships: List[Relationship] = []
        errors: List[Dict[str, Any]] = []
        seen_investigators: dict[str, Investigador] = {}
        seen_projects: dict[str, Proyecto] = {}
        seen_rel_keys: set[tuple] = set()

        if not table_dir.exists():
            errors.append({"type": "MissingFolder", "message": f"No existe: {table_dir}"})
            return TabularExtractionResult([], relationships, errors)

        csv_files = sorted(
            p for p in table_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
        )

        for csv_path in csv_files:
            errors.extend(
                self._process_csv(
                    csv_path, seen_investigators, seen_projects, relationships, seen_rel_keys
                )
            )

        entities: List[Entity] = []
        entities.extend(seen_investigators.values())
        entities.extend(seen_projects.values())
        return TabularExtractionResult(entities, relationships, errors)

    def extract_from_csv(self, csv_path: Path) -> TabularExtractionResult:
        """Extrae de un único CSV. Para mergear varios usar extract_from_directory."""
        relationships: List[Relationship] = []
        seen_investigators: dict[str, Investigador] = {}
        seen_projects: dict[str, Proyecto] = {}
        seen_rel_keys: set[tuple] = set()
        errors = self._process_csv(
            csv_path, seen_investigators, seen_projects, relationships, seen_rel_keys
        )
        entities: List[Entity] = []
        entities.extend(seen_investigators.values())
        entities.extend(seen_projects.values())
        return TabularExtractionResult(entities, relationships, errors)

    def _process_csv(
        self,
        csv_path: Path,
        seen_investigators: dict[str, Investigador],
        seen_projects: dict[str, Proyecto],
        relationships: List[Relationship],
        seen_rel_keys: set[tuple],
    ) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        try:
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row_number, row in enumerate(reader, start=2):
                    errors.extend(
                        self._process_row(
                            row,
                            row_number,
                            seen_investigators,
                            seen_projects,
                            relationships,
                            seen_rel_keys,
                        )
                    )
        except Exception as exc:
            errors.append({"type": "CSVReadError", "message": f"{csv_path.name}: {exc}"})
        return errors

    def _process_row(
        self,
        row: dict[str, Optional[str]],
        row_number: int,
        seen_investigators: dict[str, Investigador],
        seen_projects: dict[str, Proyecto],
        relationships: list[Relationship],
        seen_rel_keys: set[tuple],
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []

        inv_id = self._get_or_create_investigador(row, row_number, seen_investigators, errors)
        if inv_id is None:
            return errors

        project_id = self._get_or_create_proyecto(row, row_number, seen_projects, errors)
        if project_id is None:
            return errors

        self._add_participation_relationships(
            row, inv_id, project_id, relationships, seen_rel_keys
        )
        return errors

    def _get_or_create_investigador(
        self,
        row: dict[str, Optional[str]],
        row_number: int,
        seen_investigators: dict[str, Investigador],
        errors: list[dict[str, Any]],
    ) -> Optional[str]:
        pais = self._cell(row.get("pais_documento"))
        tipo = self._cell(row.get("tipo_documento"))
        doc = self._cell(row.get("documento"))

        if not doc or not tipo or not pais:
            errors.append(
                {
                    "type": "MissingResearcherID",
                    "message": f"Fila {row_number}: faltan pais_documento/tipo_documento/documento",
                }
            )
            return None

        inv_id = self._make_researcher_id(pais, tipo, doc)
        if inv_id in seen_investigators:
            return inv_id

        nombres = self._cell(row.get("nombres"))
        apellidos = self._cell(row.get("apellidos"))
        display = self._build_display_name(nombres, apellidos)
        if not display:
            errors.append(
                {
                    "type": "MissingResearcherName",
                    "message": f"Fila {row_number}: investigador {inv_id} sin nombres ni apellidos",
                }
            )
            return None

        sexo = self._cell(row.get("sexo"))
        seen_investigators[inv_id] = Investigador(
            id=inv_id,
            value={
                "name": self._strip_accents_lowercase(display),
                "display_name": display,
                "documento": doc,
                "tipo_documento": tipo,
                "pais_documento": pais,
                "sexo": sexo or "",
            },
        )
        return inv_id

    def _get_or_create_proyecto(
        self,
        row: dict[str, Optional[str]],
        row_number: int,
        seen_projects: dict[str, Proyecto],
        errors: list[dict[str, Any]],
    ) -> Optional[str]:
        id_formulario = self._cell(row.get("id_formulario"))
        anio = self._cell(row.get("anio"))

        if not id_formulario or not anio:
            errors.append(
                {
                    "type": "MissingProjectID",
                    "message": f"Fila {row_number}: faltan id_formulario/anio",
                }
            )
            return None

        project_id = self._make_project_id(anio, id_formulario)
        if project_id not in seen_projects:
            titulo = self._cell(row.get("titulo"))
            project_title = self._normalize_title(titulo or project_id)
            seen_projects[project_id] = Proyecto(id=project_id, value=project_title)
        return project_id

    def _add_participation_relationships(
        self,
        row: dict[str, Optional[str]],
        inv_id: str,
        project_id: str,
        relationships: list[Relationship],
        seen_rel_keys: set[tuple],
    ) -> None:
        calidad = self._cell(row.get("calidad"))
        self._add_relationship(
            self._build_project_rel(inv_id, project_id, calidad),
            relationships,
            seen_rel_keys,
        )

        nombres = self._cell(row.get("nombres"))
        apellidos = self._cell(row.get("apellidos"))
        doc = self._cell(row.get("documento"))
        evidence_text = (
            f"Investigador extraído de tabla: {nombres or ''} {apellidos or ''} "
            f"(doc: {doc or ''}, calidad: {calidad or ''})"
        ).strip()
        table_chunk_id = f"tabular_{project_id}"
        self._add_relationship(
            EXTRAIDO_DE(
                table_chunk_id,
                inv_id,
                properties={"evidence_text": evidence_text},
            ),
            relationships,
            seen_rel_keys,
        )

    def _add_relationship(
        self,
        rel: Relationship,
        relationships: list[Relationship],
        seen_rel_keys: set[tuple],
    ) -> None:
        props = rel.properties or {}
        key = (rel.source_id, rel.target_id, rel.type, props.get("calidad"))
        if key in seen_rel_keys:
            return
        seen_rel_keys.add(key)
        relationships.append(rel)

    def _cell(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def _make_researcher_id(self, pais: str, tipo: str, doc: str) -> str:
        combined = f"{pais}_{tipo}_{doc}"
        s = combined.lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c) or c == "̃")
        s = unicodedata.normalize("NFC", s)
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    def _make_project_id(self, anio: str, id_formulario: str) -> str:
        return build_project_id("proy", anio, id_formulario)

    def _build_display_name(self, nombres: Optional[str], apellidos: Optional[str]) -> str:
        parts = []
        if nombres:
            parts.append(nombres.strip().title())
        if apellidos:
            parts.append(apellidos.strip().title())
        return " ".join(parts)

    @staticmethod
    def _strip_accents_lowercase(text: str) -> str:
        """Lowercase + saca tildes (mantiene ñ). Usado en name e índices de búsqueda."""
        s = text.lower()
        s = "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" or c == "̃"
        )
        return unicodedata.normalize("NFC", s)

    def _normalize_title(self, title: str) -> str:
        return self._strip_accents_lowercase(title)

    def _build_project_rel(
        self, inv_id: str, project_id: str, calidad: Optional[str]
    ) -> Relationship:
        normalized = (calidad or "").lower().strip() or "integrante"
        return PARTICIPO_EN(inv_id, project_id, {"calidad": normalized})
