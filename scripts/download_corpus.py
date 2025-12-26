import requests
import os
import urllib3
from dotenv import load_dotenv
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import quote
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  
load_dotenv(BASE_DIR / ".env")

TOKEN = os.getenv("FING_TOKEN")
BASE_URL = f"https://nube.fing.edu.uy/index.php/s/{TOKEN}/download"
OUTPUT_DIR = BASE_DIR / "data" / "corpus"
INPUT_ARCHIVE = BASE_DIR / "data" / "downloads_list.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open(INPUT_ARCHIVE, "r", encoding="utf-8") as f:
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

        url = f"{BASE_URL}?path={path_encoded}&files={filename_encoded}"

        print(f"Download: {filename}")
        response = requests.get(url, verify=False)

        if response.status_code == 200:
            with open(os.path.join(OUTPUT_DIR, filename), "wb") as out:
                out.write(response.content)
            archive_count +=1
        else:
            print()
            print(f" Error downloading {filename} (status {response.status_code})")
            print()

    print(f"The number of files downloaded was: {archive_count}")
except FileNotFoundError:
    print(f" Error: The file '{INPUT_ARCHIVE}' was not found.")