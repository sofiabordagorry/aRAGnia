import requests
import os
import urllib3
from dotenv import load_dotenv
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import quote
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

def main():
    backend_dir = BASE_DIR / "backend"
    load_dotenv(backend_dir / ".env")
    token = os.getenv("FING_TOKEN")
    base_url = f"https://nube.fing.edu.uy/index.php/s/{token}/download"
    output_dir = BASE_DIR / "data" / "corpus"
    input_dir = BASE_DIR / "data" / "downloads_list.txt"
    if token==None:
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
            line = line.replace("\\", "/")

            parts = line.rsplit("/", 1)
            path = parts[0]
            filename = parts[1]

            path_encoded = quote(path)
            filename_encoded = quote(filename)

            url = f"{base_url}?path={path_encoded}&files={filename_encoded}"

            print(f"Download: {filename}")
            response = requests.get(url, verify=False)

            if response.status_code == 200:
                with open(os.path.join(output_dir, filename), "wb") as out:
                    out.write(response.content)
                archive_count +=1
            else:
                print()
                print(f" Error downloading {filename} (status {response.status_code})")
                print()

        print(f"The number of files downloaded was: {archive_count}")
    except FileNotFoundError:
        print(f" Error: The file '{input_dir}' was not found.")

if __name__ == "__main__":
    main()