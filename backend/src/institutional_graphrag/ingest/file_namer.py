import re


def extract_relevant_year(path):
    path = path.replace("\\", "/")

    folders = path.split("/")[:-1]

    year_pattern = re.compile(r"20\d{2}")

    numeric_folder_pattern = re.compile(r"/\d+/")

    for i, folder in enumerate(reversed(folders)):
        if numeric_folder_pattern.match("/" + folder + "/"):
            i_original = len(folders) - 1 - i
            for j in range(i_original, -1, -1):
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
            pos_informe = folder.lower().find("informe")
            if pos_informe > pos_year:
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
        prefix = ""
        root_folder = folders[0].upper() if folders else ""
        if "GRUPOS" in root_folder:
            prefix = "gi"
        elif "PROYECTOS" in root_folder:
            prefix = "proy"

        # Identificar el año correspondiente
        year = "unknown"

        year = extract_relevant_year(path_str)

        # Itera del final del path al comienzo para buscar la ID y si es informe o propuesta
        type_suffix = ""

        for i in range(len(folders) - 1, -1, -1):
            folder_lower = folders[i].lower()
            if "informe" in folder_lower:
                type_suffix = "informe"
            elif "propuesta" in folder_lower:
                type_suffix = "propuesta"
            elif "resumen" in folder_lower:
                type_suffix = "resumen"

            if type_suffix != "":
                potential_id = folders[i - 1].strip()
                match = re.match(r"(\d+)", potential_id)
                if "1_GRUPO" in potential_id or "2_PROYECTO" in potential_id or match is None:
                    return f"{prefix}_{year}_{filename}"
                potential_id = match.group(1)
                if potential_id.isdigit():
                    file_id = potential_id
                    return f"{prefix}_{year}_{file_id}_{type_suffix}.pdf"
                break

        # Rule for 'admin' (Fallback for paths not following the above structure)
        # Logic: admin_year_filename

        return f"admin_{year}_{filename}"

    except Exception as e:
        print(f"No se pudo generar el nombre de archivo: {e}")
        return "invalid_path_structure.pdf"
