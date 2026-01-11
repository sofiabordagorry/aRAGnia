import requests
import os
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import quote
from pathlib import Path

from institutional_graphrag.ingest.file_namer import generate_new_filename

BASE_DIR = Path(__file__).resolve().parents[2]


def main():
    backend_dir = BASE_DIR / "backend"
    load_dotenv(backend_dir / ".env")
    token = os.getenv("FING_TOKEN")
    base_url = f"https://nube.fing.edu.uy/index.php/s/{token}/download"
    output_dir = BASE_DIR / "data" / "corpus"
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

            line = line.replace("\\", "/")
            parts = line.rsplit("/", 1)
            path = parts[0]
            original_filename = parts[1]

            path_encoded = quote(path)
            filename_encoded = quote(original_filename)

            url = f"{base_url}?path={path_encoded}&files={filename_encoded}"

            print(f"Descargando: {original_filename} -> Guardando como: {new_filename}")
            response = requests.get(url, verify=False)

            if response.status_code == 200:
                with open(os.path.join(output_dir, new_filename), "wb") as out:
                    out.write(response.content)
                archive_count += 1
            else:
                print()
                print(f" Error downloading {original_filename} (status {response.status_code})")
                print()

        print(f"The number of files downloaded was: {archive_count}")
    except FileNotFoundError:
        print(f" Error: The file '{input_dir}' was not found.")


if __name__ == "__main__":
    main()
