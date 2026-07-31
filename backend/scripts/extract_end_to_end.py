import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

from institutional_graphrag.services.ingest_service import IngestService

# ==============================
# CONFIGURACIÓN
# ==============================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
ENV_PATH: Optional[Path] = Path(BACKEND_DIR / ".env")


# ==============================
# LECTURA DE ARCHIVOS
# ==============================


def collect_folder_files(input_dir: Path) -> List[Tuple[str, bytes]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    folder_files: List[Tuple[str, bytes]] = []
    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel_inside = file_path.relative_to(input_dir).as_posix()
        if any(part.startswith(".") for part in rel_inside.split("/")):
            continue
        folder_files.append((f"{input_dir.name}/{rel_inside}", file_path.read_bytes()))

    return folder_files


# ==============================
# MAIN ASYNC
# ==============================


async def main(input_dir: Path, csv_path: Optional[Path], keep_debug_artifacts: bool):
    print("Inicializando IngestService...")
    try:
        service = IngestService(
            data_dir=DATA_DIR,
            env_path=ENV_PATH,
            keep_debug_artifacts=keep_debug_artifacts,
        )

        print("Ejecutando ingest...")

        result = await service.ingest_from_uploads(
            folder_files=collect_folder_files(input_dir),
            csv_bytes=csv_path.read_bytes() if csv_path else None,
            csv_filename=csv_path.name if csv_path else None,
        )
        print("Resultado:")
        print(result)
    except Exception as e:
        print(f"Error durante el ingest: {e}")


# ==============================
# ENTRYPOINT
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar end to end")

    parser.add_argument(
        "input_path",
        type=Path,
        help="Carpeta local con la estructura de proyectos/grupos a ingestar",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV de proyectos a cargar junto con los archivos",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mantiene archivos intermedios para debugging (no limpia)",
    )

    args = parser.parse_args()

    keep_debug_artifacts = args.debug

    print(f"[CONFIG] MODO DEBUG: {'ACTIVO' if keep_debug_artifacts else 'DESACTIVADO'}")

    asyncio.run(main(args.input_path, args.csv, keep_debug_artifacts))
