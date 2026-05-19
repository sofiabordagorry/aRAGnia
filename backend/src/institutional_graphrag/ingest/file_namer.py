import re
import tempfile
from enum import Enum
from pathlib import Path

from institutional_graphrag.document_naming import (
    document_kind_from_name,
    document_kind_position,
    is_legacy_root_marker,
    is_narrative_document,
    prefix_from_source,
)


class PdfKind(Enum):
    TABULAR = "tabular"
    NARRATIVE = "narrative"


def looks_tabular(fileName: str) -> bool:
    return not is_narrative_document(fileName)


def classify_pdf(fileName: str) -> PdfKind:
    return PdfKind.TABULAR if looks_tabular(fileName) else PdfKind.NARRATIVE


def save_temp_file(content: bytes, filename: str) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "institutional_graphrag"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    path = tmp_dir / filename
    path.write_bytes(content)
    return path


def extract_relevant_year(path):
    path = path.replace("\\", "/")

    folders = path.split("/")[:-1]

    year_pattern = re.compile(r"20\d{2}")

    numeric_folder_pattern = re.compile(r"/\d+/")

    for i, folder in enumerate(reversed(folders)):
        if numeric_folder_pattern.match("/" + folder + "/"):
            i_original = len(folders) - 1 - i
            for j in range(i_original - 1, -1, -1):
                years = year_pattern.findall(folders[j])
                if len(years) == 1:
                    return years[0]
            break

    year = None
    alternative_year = "unknown"
    for folder in reversed(folders):
        years_in_folder = year_pattern.findall(folder)
        if len(years_in_folder) == 1:
            pos_year = folder.find(years_in_folder[0])
            pos_kind = document_kind_position(folder)
            if pos_kind > pos_year:
                year = years_in_folder[0]
                break
            alternative_year = years_in_folder[0]
    return year if year else alternative_year


def generate_new_filename(path_str):
    """
    Parsea el path para definir el nuevo nombre del archivo.
    """
    try:
        # Partir el path en partes
        path_str = path_str.replace("\\", "/")
        parts = path_str.strip("/").split("/")
        folders = parts[:-1]
        filename = parts[-1]

        # Identificar si es un grupo de investigación o un proyecto
        root_folder = folders[0] if folders else ""
        print("PATH", path_str)
        prefix = prefix_from_source(root_folder)

        # Identificar el año correspondiente
        year = "unknown"

        year = extract_relevant_year(path_str)

        # Itera del final del path al comienzo para buscar la ID y el tipo del documento
        type_suffix = ""

        for i in range(len(folders) - 1, -1, -1):
            type_suffix = document_kind_from_name(folders[i]) or type_suffix

            if type_suffix != "":
                potential_id = folders[i - 1].strip()
                match = re.match(r"(\d+)", potential_id)
                if is_legacy_root_marker(potential_id) or match is None:
                    filename_path = Path(filename)
                    return f"{prefix}_{year}_table{filename_path.suffix}"
                potential_id = match.group(1)
                if potential_id.isdigit():
                    file_id = potential_id
                    filename_path = Path(filename)
                    return f"{prefix}_{year}_{file_id}_{type_suffix}{filename_path.suffix}"
                break

        # Fallback para paths que no siguen la estructura esperada: admin_año_archivo
        return f"admin_{year}_{filename}"

    except Exception as e:
        print(f"No se pudo generar el nombre de archivo: {e}")
        return "invalid_path_structure.pdf"
