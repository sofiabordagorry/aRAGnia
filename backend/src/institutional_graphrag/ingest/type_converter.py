import subprocess
import uuid
from pathlib import Path


def odt_bytes_to_pdf(odt_bytes: bytes) -> bytes:
    shared_dir = Path(__file__).resolve().parents[3] / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    temp_odt_path = shared_dir / f"{file_id}.odt"
    temp_pdf_path = shared_dir / f"{file_id}.pdf"

    temp_odt_path.write_bytes(odt_bytes)

    try:
        cmd = [
            "docker",
            "exec",
            "libreoffice-converter",
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            "/data",
            f"/data/{temp_odt_path.name}",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Falló la conversión ODT -> PDF.\n"
                f"CMD: {cmd}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not temp_pdf_path.exists():
            raise FileNotFoundError(
                f"No se generó el PDF en {temp_pdf_path}.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        return temp_pdf_path.read_bytes()

    finally:
        if temp_odt_path.exists():
            temp_odt_path.unlink()
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()
