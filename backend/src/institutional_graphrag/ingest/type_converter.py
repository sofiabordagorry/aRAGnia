import subprocess
import tempfile
from pathlib import Path


def odt_bytes_to_pdf(odt_bytes: bytes) -> bytes:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".odt") as temp_odt:
        temp_odt.write(odt_bytes)
        temp_odt_path = Path(temp_odt.name)

    temp_pdf_path = temp_odt_path.with_suffix(".pdf")
    try:
        result = subprocess.run(
            [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temp_odt_path.parent),
                str(temp_odt_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if not temp_pdf_path.exists():
            raise FileNotFoundError(
                f"No se generó el PDF en {temp_pdf_path}. "
                f"stdout={result.stdout} stderr={result.stderr}"
            )

        pdf_bytes = temp_pdf_path.read_bytes()

    finally:
        if temp_odt_path.exists():
            temp_odt_path.unlink()
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()
    return pdf_bytes
