import re


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
        for folder in reversed(folders):  # Iterate backwards from the folders
            year_match = re.search(r"(\d{4})", folder)
            if year_match and not folder.strip().isdigit():
                year = year_match.group(0)
                break

        # Itera del final del path al comienzo para buscar la ID y si es informe o propuesta
        type_suffix = ""

        for i in range(len(folders) - 1, -1, -1):
            folder_lower = folders[i].lower()
            if "informe" in folder_lower:
                type_suffix = "informe"
            elif "propuesta" in folder_lower:
                type_suffix = "propuesta"

            if type_suffix != "":
                potential_id = folders[i - 1].strip()
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
