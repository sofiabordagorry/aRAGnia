import subprocess
import tempfile
from pathlib import Path

def odt_bytes_to_pdf(odt_bytes: bytes) -> bytes:
    # Crear un archivo temporal para el ODT
    with tempfile.NamedTemporaryFile(delete=False, suffix=".odt") as temp_odt:
        temp_odt.write(odt_bytes)
        temp_odt_path = Path(temp_odt.name)

    # Crear un archivo temporal para el PDF
    temp_pdf_path = temp_odt_path.with_suffix(".pdf")

    try:
        # Ejecutar LibreOffice headless para convertir a PDF
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--nologo",
                "--nofirststartwizard",
                "--convert-to", "pdf",
                str(temp_odt_path)
            ],
            check=True
        )

        # Leer el PDF generado a memoria
        pdf_bytes = temp_pdf_path.read_bytes()

    finally:
        # Limpiar archivos temporales
        if temp_odt_path.exists():
            temp_odt_path.unlink()
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()

    return pdf_bytes
