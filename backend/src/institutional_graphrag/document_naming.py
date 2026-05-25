from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

DOCUMENT_KIND_ALIASES: dict[str, tuple[str, ...]] = {
    "informe": ("informe",),
    "propuesta": ("propuesta",),
    "resumen": ("resumen",),
}

DOCUMENT_KIND_PRIORITY: dict[str, int] = {
    "resumen": 1,
    "informe": 2,
    "propuesta": 3,
}

GROUP_PREFIXES: tuple[str, ...] = ("gi", "proy")
LEGACY_ROOT_MARKERS: tuple[str, ...] = ("1_GRUPO", "2_PROYECTO")

_KIND_REGEX = "|".join(sorted(DOCUMENT_KIND_ALIASES))
_GROUP_REGEX = "|".join(GROUP_PREFIXES)

PATTERN_DOCUMENT = re.compile(
    rf"^(?P<group>[A-Za-z]+)_(?P<year>\d{{4}})_(?P<doc_id>\d+)_(?P<kind>{_KIND_REGEX})$",
    re.IGNORECASE,
)

PATTERN_DOCUMENT_WITH_OPTIONAL_PDF = re.compile(
    rf"^(?P<group>[A-Za-z]+)_(?P<year>\d{{4}})_(?P<doc_id>\d+)_(?P<kind>{_KIND_REGEX})(?:\.pdf)?$",
    re.IGNORECASE,
)

PATTERN_TABLE = re.compile(r".+", re.IGNORECASE)

PROJECT_KEY_RE = re.compile(rf"((?:{_GROUP_REGEX})_\d{{4}})_\d+", re.IGNORECASE)


def _norm_text(value: str) -> str:
    return value.lower().strip()


def is_group(value: str) -> bool:
    v = _norm_text(Path(value).stem)
    return v.startswith("gi_") or "grupos" in v


def is_project(value: str) -> bool:
    v = _norm_text(Path(value).stem)
    return v.startswith("proy_") or "proyectos" in v


def prefix_from_source(value: str) -> str:
    if is_group(value):
        return "gi"
    if is_project(value):
        return "proy"
    return ""


def document_kind_from_name(value: str) -> Optional[str]:
    text = _norm_text(Path(value).stem)
    for canonical, aliases in DOCUMENT_KIND_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def document_kind_position(value: str) -> int:
    text = _norm_text(value)
    positions: list[int] = []
    for aliases in DOCUMENT_KIND_ALIASES.values():
        for alias in aliases:
            pos = text.find(alias)
            if pos >= 0:
                positions.append(pos)
    return min(positions) if positions else -1


def is_narrative_document(value: str) -> bool:
    return document_kind_from_name(value) is not None


def is_legacy_root_marker(value: str) -> bool:
    text = value.upper().strip()
    return any(marker in text for marker in LEGACY_ROOT_MARKERS)


def build_project_key(group: str, year: str) -> str:
    return f"{group}_{year}"


def build_project_id(group: str, year: str, sub_id: str) -> str:
    return f"{build_project_key(group, year)}_{sub_id}"


def build_table_chunk_id(group: str, year: str, sub_id: str) -> str:
    return f"{build_project_key(group, year)}_table_{sub_id}"


def extract_project_key(value: str) -> Optional[str]:
    match = PROJECT_KEY_RE.search(value)
    if match:
        return match.group(1)
    return None


def parent_doc_from_stem(stem: str) -> str:
    project_key = extract_project_key(stem)
    if project_key:
        return project_key

    # Fallback para stems no estándar: "gi_124_texto" -> "gi_124"
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem
