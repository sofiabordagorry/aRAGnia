import os
import re

import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import tempfile
from enum import Enum
from pathlib import Path
from urllib.parse import quote

from institutional_graphrag.ingest.file_namer import generate_new_filename
from institutional_graphrag.ingest.table_extractors import extract_table
from institutional_graphrag.ingest.file_namer import classify_pdf, PdfKind, save_temp_file



BASE_DIR = Path(__file__).resolve().parents[2]

def main():
    backend_dir = BASE_DIR / "backend"
    load_dotenv(backend_dir / ".env")
    token = os.getenv("FING_TOKEN")
    base_url = f"https://nube.fing.edu.uy/index.php/s/{token}/download"
    output_dir = BASE_DIR / "data" / "corpus"
    output_dir_tables = BASE_DIR / "data" / "tables"
    input_dir = BASE_DIR / "data" / "downloads_list.txt"

    if token is None:
        print("Token Invalido")
        return
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_dir, "r", encoding="utf-8") as f:
            lines = f.readlines()

        archive_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            new_filename = generate_new_filename(line)
            file_path = Path(output_dir) / new_filename
            if file_path.exists():
                print(
                    "El archivo a descargar ya existe :",
                    line,
                    "con el nombre: ",
                    new_filename,
                    "en la carpeta ",
                    output_dir,
                )
                print("=" * 60)
                continue
            file_path = Path(output_dir_tables) / f"{Path(new_filename).stem}.parquet"
            if file_path.exists():
                print(
                    "El archivo a descargar ya existe :",
                    line,
                    "con el nombre: ",
                    new_filename,
                    "en la carpeta ",
                    output_dir_tables,
                )
                print("=" * 60)
                continue
            line = line.replace("\\", "/")
            parts = line.rsplit("/", 1)
            path = parts[0]
            original_filename = parts[1]

            path_encoded = quote(path)
            filename_encoded = quote(original_filename)

            url = f"{base_url}?path={path_encoded}&files={filename_encoded}"

            response = requests.get(url, verify=False)

            if response.status_code == 200:
                tmp_path = save_temp_file(response.content, new_filename)
                try:
                    kind = classify_pdf(new_filename)

                    if kind == PdfKind.TABULAR:
                        print(f"Descargando: {original_filename}")
                        extract_table(tmp_path, output_dir_tables)
                    else:
                        print(f"Descargando: {original_filename} -> Guardando como: {new_filename}")
                        with open(os.path.join(output_dir, new_filename), "wb") as out:
                            out.write(response.content)
                    archive_count += 1
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
            else:
                print()
                print(f" Error downloading {original_filename} (status {response.status_code})")
                print()

        print(f"The number of files downloaded was: {archive_count}")
    except FileNotFoundError:
        print(f" Error: The file '{input_dir}' was not found.")


if __name__ == "__main__":
    main()
