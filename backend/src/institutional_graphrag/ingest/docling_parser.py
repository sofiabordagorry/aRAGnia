from pathlib import Path
from typing import Any, Optional, cast

from docling.document_converter import DocumentConverter

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
DEFAULT_CORPUS_DIR = DATA_DIR / "corpus"
DEFAULT_DOCLING_DIR = DATA_DIR / "docling"

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"]
GLOB_PATTERNS = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]


class DocumentAlreadyProcessed(Exception):
    """El documento ya estaba en cache."""


def promote_consecutive_section_headers(doc_dict: dict[str, Any]) -> None:
    """
    Si hay section_header consecutivos con el mismo level (típico portada + primera sección),
    sube el level de los siguientes para que no se pisen:
      level 1, level 1, level 2  ->  level 1, level 2, level 3
    Resetea cuando aparece un item que no sea section_header.
    """
    body = doc_dict.get("body")
    if not isinstance(body, dict):
        return
    children = body.get("children")
    if not isinstance(children, list):
        return

    texts = doc_dict.get("texts")
    if not isinstance(texts, list):
        return

    in_streak = False

    for ref in children:
        parsed = _ref_to_index(ref)
        if not parsed:
            in_streak = False
            prev_level = None
            continue

        arr_name, idx = parsed
        if arr_name != "texts" or not (0 <= idx < len(texts)):
            in_streak = False
            prev_level = None
            continue

        item = texts[idx]
        if not isinstance(item, dict):
            in_streak = False
            prev_level = None
            continue

        if item.get("label") != "section_header":
            in_streak = False
            prev_level = None
            continue

        lvl = item.get("level")
        if not isinstance(lvl, int):
            in_streak = False
            prev_level = None
            continue

        if not in_streak:
            # arranca racha
            in_streak = True
            prev_level = lvl
        else:
            # header consecutivo: forzar lvl = prev_level + 1
            new_lvl = (prev_level or lvl) + 1
            item["level"] = new_lvl
            prev_level = new_lvl


def is_already_processed(path: Path) -> bool:
    json_path = DEFAULT_DOCLING_DIR / path.with_suffix(".json").name
    if json_path.exists():
        raise DocumentAlreadyProcessed(f"El documento {path.name} ya había sido convertido.")
    return False


def _ref_to_index(ref: dict[str, Any]) -> Optional[tuple[str, int]]:
    ref_str = ref.get("$ref") if isinstance(ref, dict) else None
    if not isinstance(ref_str, str) or not ref_str.startswith("#/"):
        return None
    parts = ref_str[2:].split("/")
    if len(parts) != 2:
        return None
    name, idx_str = parts
    try:
        return name, int(idx_str)
    except Exception:
        return None


def _best_prov_key_from_item(item: dict[str, Any]) -> Optional[tuple[int, float, float]]:
    prov_list = item.get("prov")
    if not isinstance(prov_list, list) or not prov_list:
        return None

    best: Optional[tuple[int, float, float]] = None
    best_t: Optional[float] = None

    for prov in prov_list:
        if not isinstance(prov, dict):
            continue
        bbox = prov.get("bbox")
        if not isinstance(bbox, dict):
            continue

        page_no = prov.get("page_no")
        top = bbox.get("t")
        left = bbox.get("l")
        if page_no is None or top is None or left is None:
            continue

        top = float(top)
        if best is None or best_t is None or top > best_t:
            best = (int(page_no), top, float(left))
            best_t = top

    return best


def _get_item_bbox_key(
    doc_dict: dict[str, Any], ref: dict[str, Any]
) -> Optional[tuple[int, float, float]]:
    parsed = _ref_to_index(ref)
    if not parsed:
        return None
    arr_name, idx = parsed

    arr = doc_dict.get(arr_name)
    if not isinstance(arr, list) or idx < 0 or idx >= len(arr):
        return None

    item = arr[idx]
    if not isinstance(item, dict):
        return None

    direct = _best_prov_key_from_item(item)
    if direct is not None:
        page_no, top, left = direct
        return (page_no, -top, left)

    if arr_name == "groups":
        children = item.get("children")
        if not isinstance(children, list) or not children:
            return None

        child_keys = [k for ch in children if (k := _get_item_bbox_key(doc_dict, ch)) is not None]
        if not child_keys:
            return None
        child_keys.sort()
        return child_keys[0]

    return None


def reorder_refs_by_bbox(
    doc_dict: dict[str, Any], refs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    keyed = [(k, r) for r in refs if (k := _get_item_bbox_key(doc_dict, r)) is not None]
    unkeyed = [r for r in refs if _get_item_bbox_key(doc_dict, r) is None]
    keyed.sort(key=lambda x: x[0])
    return [r for _, r in keyed] + unkeyed


def _postprocess_doc_dict(doc_dict: dict[str, Any]) -> dict[str, Any]:
    body = doc_dict.get("body")
    if isinstance(body, dict) and isinstance(body.get("children"), list):
        body["children"] = reorder_refs_by_bbox(doc_dict, body["children"])

    groups = doc_dict.get("groups")
    if isinstance(groups, list):
        for g in groups:
            if isinstance(g, dict) and isinstance(g.get("children"), list):
                g["children"] = reorder_refs_by_bbox(doc_dict, g["children"])

    promote_consecutive_section_headers(doc_dict)
    return doc_dict


#### Parsear documento


def parse_single_document(source: Path) -> dict[str, Any] | None:
    if is_already_processed(source):
        print(f"El documento {source.name} ya había sido convertido.")
        return None
    converter = DocumentConverter()
    res = converter.convert(source)
    doc_dict = cast(dict[str, Any], res.document.export_to_dict())
    return _postprocess_doc_dict(doc_dict)


#### Parsear Carpeta


def get_input_paths(corpus_dir: Path, recursive: bool = False) -> list[Path]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Directorio de corpus no encontrado: {corpus_path}")
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"La ruta del corpus no es un directorio: {corpus_path}")

    patterns = (f"**/{p}" if recursive else p for p in GLOB_PATTERNS)
    files = [p for pat in patterns for p in corpus_path.glob(pat) if p.is_file()]
    return sorted(files)


def filter_unprocessed(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if is_already_processed(p):
            print(f"El documento {p.name} ya había sido convertido.")
            continue
        out.append(p)
    return out


def parse_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    recursive: bool = False,
    skip_errors: bool = True,
) -> list[dict[str, Any]]:
    paths = filter_unprocessed(get_input_paths(corpus_dir, recursive))
    if not paths:
        return []

    converter = DocumentConverter()
    results = converter.convert_all(source=paths, raises_on_error=not skip_errors)

    docs: list[dict[str, Any]] = []
    for res in results:
        doc_dict = cast(dict[str, Any], res.document.export_to_dict())
        docs.append(_postprocess_doc_dict(doc_dict))

    return docs
