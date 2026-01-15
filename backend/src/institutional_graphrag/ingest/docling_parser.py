from pathlib import Path
from typing import cast, Any, Optional
from collections import Counter
import re
from docling.document_converter import DocumentConverter

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
DEFAULT_CORPUS_DIR = DATA_DIR / Path("corpus")
DEFAULT_DOCLING_DIR = DATA_DIR / Path("docling")
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"]
GLOB_PATTERNS = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]


class DocumentAlreadyProcessed(Exception):
    """El documento ya estaba en cache."""

    pass


def extract_furniture_lines(doc) -> list[str]:
    from docling_core.types.doc import ContentLayer  # type: ignore

    lines: list[str] = []
    for item, _level in doc.iterate_items(included_content_layers={ContentLayer.FURNITURE}):
        text = getattr(item, "text", None)
        if not text:
            continue
        s = re.sub(r"\s+", " ", str(text)).strip()
        if s:
            lines.append(s)
    return lines

def get_input_paths(corpus_dir: Path, recursive: bool = False) -> list[Path]:
    """Devuelve la lista de archivos de documentos soportados en un directorio."""
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Directorio de corpus no encontrado: {corpus_path}")
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"La ruta del corpus no es un directorio: {corpus_path}")

    files: list[Path] = []

    for pattern in (f"**/{p}" if recursive else p for p in GLOB_PATTERNS):
        files.extend(p for p in corpus_path.glob(pattern) if p.is_file())

    return sorted(files)


def exists_docling(path: Path) -> bool:
    """Devuelve True si la version JSON del documento ya existe en el directiorio data/docling."""
    json_twin = DEFAULT_DOCLING_DIR / path.with_suffix(".json").name
    if json_twin.exists():
        raise DocumentAlreadyProcessed(f"El documento {path.name} ya había sido convertido.")
    else:
        return False
def _ref_to_index(ref: dict[str, Any]) -> Optional[tuple[str, int]]:
    r = ref.get("$ref") if isinstance(ref, dict) else None
    if not isinstance(r, str) or not r.startswith("#/"):
        return None
    parts = r[2:].split("/")
    if len(parts) != 2:
        return None
    name, idx_str = parts
    try:
        return name, int(idx_str)
    except Exception:
        return None


def _best_prov_key_from_item(item: dict[str, Any]) -> Optional[tuple[int, float, float]]:
    """
    Devuelve (page_no, t, l) usando el prov con mayor t (más arriba en BOTTOMLEFT).
    """
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
        t = bbox.get("t")
        l = bbox.get("l")
        if page_no is None or t is None or l is None:
            continue

        t = float(t)
        if best is None or best_t is None or t > best_t:
            best = (int(page_no), t, float(l))
            best_t = t

    return best


def _get_item_bbox_key(doc_dict: dict[str, Any], ref: dict[str, Any]) -> Optional[tuple[int, float, float]]:
    """
    Key de orden: (page_no asc, -t desc, l asc)
    - Para texts: usa su prov/bbox
    - Para groups: calcula bbox “virtual” desde sus children (texts dentro del group)
    """
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

    # 1) Caso normal: el item trae prov
    direct = _best_prov_key_from_item(item)
    if direct is not None:
        page_no, t, l = direct
        return (page_no, -t, l)

    # 2) Caso group sin prov: derivar desde sus children
    if arr_name == "groups":
        children = item.get("children")
        if not isinstance(children, list) or not children:
            return None

        child_keys = []
        for ch in children:
            # ch es {"$ref":"#/texts/10"} etc.
            ck = _get_item_bbox_key(doc_dict, ch)
            if ck is not None:
                # ck es (page_no, -t, l)
                child_keys.append(ck)

        if not child_keys:
            return None

        # Para ubicar el group: usá el primero “visual” de sus hijos
        child_keys.sort()
        return child_keys[0]

    return None

def reorder_refs_by_bbox(doc_dict: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with_key = []
    without_key = []
    for ref in refs:
        key = _get_item_bbox_key(doc_dict, ref)
        if key is None:
            without_key.append(ref)
        else:
            with_key.append((key, ref))

    with_key.sort(key=lambda x: x[0])
    return [r for _, r in with_key] + without_key


def parse_single_document(source: Path) -> dict | None:
    """Para pruebas: intenta parsear un único documento sin importar su formato."""
    try:
        if not exists_docling(path=source):
            # Inicializar el conversor
            converter = DocumentConverter()
            # Convertir el archivo y devolverlo
            result = converter.convert(source)
            doc = result.document
            doc_dict = cast(dict[str, Any], doc.export_to_dict())
            body = doc_dict.get("body")
            if isinstance(body, dict) and isinstance(body.get("children"), list):
                body["children"] = reorder_refs_by_bbox(doc_dict, body["children"])

            groups = doc_dict.get("groups")
            if isinstance(groups, list):
                for g in groups:
                    if isinstance(g, dict) and isinstance(g.get("children"), list):
                        g["children"] = reorder_refs_by_bbox(doc_dict, g["children"])

            return doc_dict
    except DocumentAlreadyProcessed as e:
        print(e)
    return None


def parse_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    recursive: bool = False,
    skip_errors: bool = True,
) -> list[dict]:

    # Obtener todos los archivos válidos
    input_paths = get_input_paths(corpus_dir=corpus_dir, recursive=recursive)
    # Filtrar archivos ya procesados
    filtered_paths = []
    for p in input_paths:
        try:
            if not exists_docling(p):
                filtered_paths.append(p)
        except DocumentAlreadyProcessed as e:
            print(e)
            continue

    input_paths = filtered_paths

    # Inicializar el conversor
    doc_converter = DocumentConverter()

    # Convertir los archivos y devolverlos
    conv_results = doc_converter.convert_all(source=input_paths, raises_on_error=not skip_errors)
    return [res.document.export_to_dict() for res in conv_results]
