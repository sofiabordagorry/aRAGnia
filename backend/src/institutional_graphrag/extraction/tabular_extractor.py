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

RESEARCHER_CSV_PATTERN = re.compile(r"^equipos_.*\.csv$", re.IGNORECASE)


@dataclass
class TabularExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


class TabularResearcherExtractor:
    def extract_from_directory(self, table_dir: Path) -> TabularExtractionResult:
        """Find and extract from all researcher CSV files in table_dir.

        Prefers *_clean.csv over the original when both exist.
        """
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        errors: List[Dict[str, Any]] = []

        if not table_dir.exists():
            errors.append({"type": "MissingFolder", "message": f"No existe: {table_dir}"})
            return TabularExtractionResult(entities, relationships, errors)

        csv_files = sorted(
            p for p in table_dir.iterdir() if p.is_file() and RESEARCHER_CSV_PATTERN.match(p.name)
        )

        by_base: dict[str, Path] = {}
        for p in csv_files:
            base = re.sub(r"_clean$", "", p.stem, flags=re.IGNORECASE)
            existing = by_base.get(base)
            if existing is None or "_clean" in p.stem.lower():
                by_base[base] = p

        for csv_path in sorted(by_base.values()):
            result = self.extract_from_csv(csv_path)
            entities.extend(result.entities)
            relationships.extend(result.relationships)
            errors.extend(result.errors)

        return TabularExtractionResult(entities, relationships, errors)

    def extract_from_csv(self, csv_path: Path) -> TabularExtractionResult:
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        errors: List[Dict[str, Any]] = []

        seen_investigators: dict[str, Investigador] = {}
        seen_projects: dict[str, Proyecto] = {}

        try:
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for lineno, row in enumerate(reader, start=2):
                    row_errors = self._process_row(
                        row, lineno, seen_investigators, seen_projects, relationships
                    )
                    errors.extend(row_errors)
        except Exception as exc:
            errors.append({"type": "CSVReadError", "message": f"{csv_path.name}: {exc}"})
            return TabularExtractionResult(entities, relationships, errors)

        entities.extend(seen_investigators.values())
        entities.extend(seen_projects.values())
        return TabularExtractionResult(entities, relationships, errors)

    def _process_row(
        self,
        row: dict[str, Optional[str]],
        lineno: int,
        seen_investigators: dict[str, Investigador],
        seen_projects: dict[str, Proyecto],
        relationships: list[Relationship],
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []

        pais = self._cell(row.get("pais_documento"))
        tipo = self._cell(row.get("tipo_documento"))
        doc = self._cell(row.get("documento"))
        nombres = self._cell(row.get("nombres"))
        apellidos = self._cell(row.get("apellidos"))
        sexo = self._cell(row.get("sexo"))
        calidad = self._cell(row.get("calidad"))
        id_unico = self._cell(row.get("id_unico"))
        anio = self._cell(row.get("anio"))
        programa = self._cell(row.get("programa"))
        titulo = self._cell(row.get("titulo"))

        if not doc or not tipo or not pais:
            errors.append(
                {
                    "type": "MissingResearcherID",
                    "message": (
                        f"Fila {lineno}: faltan pais_documento/tipo_documento/documento"
                    ),
                }
            )
            return errors

        if not id_unico or not anio:
            errors.append(
                {
                    "type": "MissingProjectID",
                    "message": f"Fila {lineno}: faltan id_unico/anio",
                }
            )
            return errors

        inv_id = self._make_researcher_id(pais, tipo, doc)
        project_id = self._make_project_id(programa or "desconocido", anio, id_unico)

        if inv_id not in seen_investigators:
            display = self._build_display_name(nombres, apellidos)
            seen_investigators[inv_id] = Investigador(
                id=inv_id,
                value={
                    "name": display.lower() if display else inv_id,
                    "display_name": display or inv_id,
                    "source": "tabular",
                    "nombre": nombres or "",
                    "apellido": apellidos or "",
                    "documento": doc,
                    "tipo_documento": tipo,
                    "pais_documento": pais,
                    "sexo": sexo or "",
                },
            )

        if project_id not in seen_projects:
            project_title = self._normalize_title(titulo or project_id)
            seen_projects[project_id] = Proyecto(id=project_id, value=project_title)

        rel = self._build_project_rel(inv_id, project_id, calidad)
        relationships.append(rel)

        evidence_text = (
            f"Investigador extraído de tabla: {nombres or ''} {apellidos or ''} "
            f"(doc: {doc}, calidad: {calidad or ''})"
        ).strip()
        table_chunk_id = f"tabular_{project_id}"
        relationships.append(
            EXTRAIDO_DE(
                table_chunk_id,
                inv_id,
                properties={"evidence_text": evidence_text},
            )
        )

        return errors

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

    def _normalize_programa(self, programa: str) -> str:
        s = programa.lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    def _make_project_id(self, programa: str, anio: str, id_unico: str) -> str:
        group = self._normalize_programa(programa)
        return build_project_id(group, anio, id_unico)

    def _build_display_name(self, nombres: Optional[str], apellidos: Optional[str]) -> str:
        parts = []
        if nombres:
            parts.append(nombres.strip().title())
        if apellidos:
            parts.append(apellidos.strip().title())
        return " ".join(parts)

    def _normalize_title(self, title: str) -> str:
        s = title.lower()
        s = "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" or c == "̃"
        )
        return unicodedata.normalize("NFC", s)

    def _build_project_rel(
        self, inv_id: str, project_id: str, calidad: Optional[str]
    ) -> Relationship:
        normalized = (calidad or "").lower().strip() or "integrante"
        return PARTICIPO_EN(inv_id, project_id, {"calidad": normalized})
