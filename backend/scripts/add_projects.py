import asyncio
from pathlib import Path
from typing import List, Optional

from institutional_graphrag.services.ingest_service import IngestService, collect_folder_files

# ==============================
# CONFIGURACIÓN
# ==============================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
ENV_PATH: Optional[Path] = Path(BACKEND_DIR / ".env")

# ==============================
# MAIN ASYNC
# ==============================


async def main(input_dirs: List[Path], csv_path: Optional[Path]):
    print("Inicializando IngestService...")

    service = IngestService(
        data_dir=DATA_DIR,
        env_path=ENV_PATH,
        keep_debug_artifacts=True,
    )

    print("Ejecutando ingest...")
    for input_dir in input_dirs:
        service.cleanup()
        result = await service.ingest_from_uploads(
            folder_files=collect_folder_files(input_dir),
            csv_bytes=csv_path.read_bytes() if csv_path else None,
            csv_filename=csv_path.name if csv_path else None,
        )
        print("Resultados de la carpeta ", input_dir, ":")
        print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestar varias carpetas de proyectos")

    parser.add_argument(
        "input_paths",
        type=Path,
        nargs="+",
        help="Carpetas locales con la estructura de proyectos/grupos a ingestar",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV de proyectos a cargar junto con los archivos",
    )

    args = parser.parse_args()

    asyncio.run(main(args.input_paths, args.csv))
